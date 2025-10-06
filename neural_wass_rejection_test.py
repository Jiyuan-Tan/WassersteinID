import numpy as np
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import time

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as e:  # pragma: no cover
    raise ImportError("PyTorch is required for neural_wass.py. Please install with: pip install torch --extra-index-url https://download.pytorch.org/whl/cpu") from e


# ==========================
# Utility: set random seeds
# ==========================
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover (GPU optional)
        torch.cuda.manual_seed_all(seed)


# ===========================================
# Nuisance estimation + debiased ATE gradients
# ===========================================
class DMLNuisance:
    """A light, self-contained version of the dml functionality needed.
    Sample splitting: first half used for test/statistic, second half for nuisance training.
    """
    def __init__(self, W: np.ndarray, D: np.ndarray, Y: np.ndarray, method_propensity: str = 'logistic'):
        assert W.ndim == 2, "W must be 2D"
        n_full = W.shape[0]
        self.n = n_full // 2
        if self.n < 5:
            warnings.warn("Very small sample after split; results may be unstable.")
        self.W = W[:self.n, :]
        # Ensure D and Y are 1-dimensional arrays so boolean indexing matches rows
        self.D = np.ravel(D[:self.n])
        self.Y = np.ravel(Y[:self.n])
        self.W_est = W[self.n:]
        self.D_est = np.ravel(D[self.n:])
        self.Y_est = np.ravel(Y[self.n:])
        self.method_propensity = method_propensity
        self._fitted = False

    # ---- Fit nuisance models ----
    def fit_propensity(self):
        if self.method_propensity == 'kernel':
            kr = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.5)
            kr.fit(self.W_est, self.D_est.reshape(-1, 1))
            self._propensity_model = kr
            self.propensity = lambda w: np.clip(self._propensity_model.predict(w.reshape(1, -1)).reshape(-1, 1), 1e-2, 1 - 1e-2)
        elif self.method_propensity == 'logistic':
            log_reg = LogisticRegression()
            log_reg.fit(self.W_est, self.D_est.ravel())
            self._propensity_model = log_reg
            self.propensity = lambda w: np.clip(self._propensity_model.predict_proba(w.reshape(1, -1))[:, 1].reshape(-1, 1), 1e-2, 1 - 1e-2)
        else:
            raise ValueError("Unsupported propensity method.")

    def fit_outcomes(self):
        kr1 = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.5)
        kr0 = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.5)
        # Use a 1-D boolean mask to avoid shape mismatch when D_est is a column vector
        mask_treated = (np.ravel(self.D_est) == 1)
        mask_control = (np.ravel(self.D_est) == 0)
        W1 = self.W_est[mask_treated]
        Y1 = self.Y_est[mask_treated]
        W0 = self.W_est[mask_control]
        Y0 = self.Y_est[mask_control]
        if len(W1) == 0 or len(W0) == 0:
            raise RuntimeError("Need both treated and control samples for outcome models.")
        kr1.fit(W1, Y1)
        kr0.fit(W0, Y0)
        self._outcome1 = kr1
        self._outcome0 = kr0
        self.outcome1 = lambda w: self._outcome1.predict(w.reshape(1, -1))
        self.outcome0 = lambda w: self._outcome0.predict(w.reshape(1, -1))

    def fit(self):
        self.fit_propensity()
        self.fit_outcomes()
        self.set_ate_grad()
        self.debiased_ate = lambda w, d, y: self.outcome1(w) - self.outcome0(w) + d * (y - self.outcome1(w)) / self.propensity(w) - (1 - d) * (y - self.outcome0(w)) / (1 - self.propensity(w))
        self._fitted = True

    # ---- Gradients of nuisance pieces ----
    def propensity_gradient(self, w, gamma=0.5):
        w = np.array(w).reshape(1, -1)
        if self.method_propensity == 'kernel':
            W_train = self.W_est
            diff = np.tile(w, (len(W_train), 1)) - W_train
            K = np.exp(-gamma * np.sum(diff ** 2, axis=1))
            grad = -2 * gamma * diff * K[:, None]
            alpha = self._propensity_model.dual_coef_.reshape(-1)
            return np.dot(grad.T, alpha)
        elif self.method_propensity == 'logistic':
            p = self.propensity(w)
            return self._propensity_model.coef_.reshape(-1) * p * (1 - p)
        else:  # pragma: no cover
            raise ValueError("Unsupported propensity method")

    def outcome_gradient(self, w, gamma=0.5):
        w = np.array(w).reshape(1, -1)
        W1 = self.W_est[self.D_est == 1]
        W0 = self.W_est[self.D_est == 0]
        diff1 = np.tile(w, (len(W1), 1)) - W1
        K1 = np.exp(-gamma * np.sum(diff1 ** 2, axis=1))
        grad1 = -2 * gamma * diff1 * K1[:, None]
        a1 = self._outcome1.dual_coef_.reshape(-1)
        g1 = np.dot(grad1.T, a1)
        diff0 = np.tile(w, (len(W0), 1)) - W0
        K0 = np.exp(-gamma * np.sum(diff0 ** 2, axis=1))
        grad0 = -2 * gamma * diff0 * K0[:, None]
        a0 = self._outcome0.dual_coef_.reshape(-1)
        g0 = np.dot(grad0.T, a0)
        return g1, g0

    def set_ate_grad(self):
        self.ate_gd = lambda w, d, y: (y - self.outcome1(w)) / self.propensity(w) + (y - self.outcome0(w)) / (1 - self.propensity(w))
        self.ate_gy = lambda w, d, y: d / self.propensity(w) - (1 - d) / (1 - self.propensity(w))

    def ate_gw(self, w, d, y):
        g1, g0 = self.outcome_gradient(w)
        g_prop = self.propensity_gradient(w)
        p = self.propensity(w)
        o1 = self.outcome1(w)
        o0 = self.outcome0(w)
        # derivative from original code logic
        return g1 - g0 + d * (-g1 / p + (o1 - y) / (p ** 2) * g_prop) - (1 - d) * (-g0 / (1 - p) - (y - o0) / ((1 - p) ** 2) * g_prop)


# ==========================
# Neural network definition
# ==========================
class PerturbNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: List[int]):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: (batch, input_dim)
        return self.net(x)


@dataclass
class TrainingConfig:
    n_outer: int = 200
    inner_epochs: int = 10  # moderate fixed number per spec
    batch_size: Optional[int] = None  # default -> full batch
    lr_theta: float = 1e-2
    lr_lambda: float = 1e-3
    print_every: int = 20
    grad_clip: Optional[float] = 5.0


class NeuralWassCI:
    """Neural implementation of the finite-sum min (theta) / max (lambda) problem described.

    Objective per iteration for fixed lambda:
        L(θ; λ) = (1/n) Σ_i [ λ * sqrt(n) * debiased_ate( (W_i,D_i,Y_i) + f_θ(...) / sqrt(n) ) + ||f_θ(...)||^2 ]

    λ update (gradient ascent):
        λ ← λ + lr_λ * (1/n) Σ_i sqrt(n) * debiased_ate( perturbed_i )
    """
    def __init__(self, W: np.ndarray, D: np.ndarray, Y: np.ndarray, method_propensity: str = 'logistic', alpha: float = 0.05,
                 hidden: List[int] = [32, 32], seed: int = 42, device: Optional[str] = None):
        set_seed(seed)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.alpha = alpha
        self.nuisance = DMLNuisance(W, D, Y, method_propensity=method_propensity)
        self.n = self.nuisance.n
        self.d = self.nuisance.W.shape[1]
        self.input_dim = self.d + 2  # (W, D, Y)
        self.output_dim = self.d + 2  # perturbations for (W, D, Y)
        self.model = PerturbNet(self.input_dim, self.output_dim, hidden).to(self.device)
        self.lambda_param = 0.0  # unrestricted scalar λ
        self.sqrt_n = math.sqrt(self.n)
        self.history = {"lambda": [], "loss": [], "lam_grad": []}
        self.radius: Optional[float] = None

    # ---- Radius (reuse Wasserstein style) ----
    def compute_radius(self):
        m = [self.nuisance.debiased_ate(self.nuisance.W[i, :], self.nuisance.D[i], self.nuisance.Y[i]) for i in range(self.n)]
        ate_est = np.mean(m)
        moment_var = np.var(m)
        d_m = self.d
        grad_d = np.array([self.nuisance.ate_gd(self.nuisance.W[i, :], self.nuisance.D[i], self.nuisance.Y[i]) for i in range(self.n)])
        grad_y = np.array([self.nuisance.ate_gy(self.nuisance.W[i, :], self.nuisance.D[i], self.nuisance.Y[i]) for i in range(self.n)])
        grad_w = np.zeros((self.n, d_m))
        for i in range(self.n):
            grad_w[i, :] = self.nuisance.ate_gw(self.nuisance.W[i, :], self.nuisance.D[i], self.nuisance.Y[i])
        m_g = (np.linalg.norm(grad_w) ** 2 + np.linalg.norm(grad_d) ** 2 + np.linalg.norm(grad_y) ** 2) / self.n
        from scipy.stats import chi2
        self.radius = chi2.ppf(1 - self.alpha, 1) * moment_var / (m_g)
        return ate_est

    # ---- Vector helpers ----
    def _forward_full(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass on all n samples, return (z, Wp, Dp, Yp)."""
        with torch.no_grad():
            inputs = torch.from_numpy(np.concatenate([
                self.nuisance.W,
                self.nuisance.D.reshape(-1, 1),
                self.nuisance.Y.reshape(-1, 1)
            ], axis=1)).float().to(self.device)
            z = self.model(inputs).cpu().numpy()  # (n, d+2)
        z_w = z[:, :self.d]
        z_d = z[:, self.d]
        z_y = z[:, self.d + 1]
        Wp = self.nuisance.W + z_w / self.sqrt_n
        Dp = self.nuisance.D + z_d / self.sqrt_n
        Yp = self.nuisance.Y + z_y / self.sqrt_n
        return z, Wp, Dp, Yp

    def _compute_debiased_vectorized(self, Wp, Dp, Yp):
        # Batch predictions for outcomes and propensity
        # (KernelRidge / logistic don't expose batch probability for arbitrary arrays with reshape tweaks)
        out1 = self.nuisance._outcome1.predict(Wp).reshape(-1)
        out0 = self.nuisance._outcome0.predict(Wp).reshape(-1)
        if self.nuisance.method_propensity == 'logistic':
            p = self.nuisance._propensity_model.predict_proba(Wp)[:, 1]
        else:
            p = self.nuisance._propensity_model.predict(Wp).reshape(-1)
        p = np.clip(p, 1e-2, 1 - 1e-2)
        g = (out1 - out0 + Dp * (Yp - out1) / p - (1 - Dp) * (Yp - out0) / (1 - p))
        return g

    # ---- Training loop ----
    def fit(self, config: TrainingConfig = TrainingConfig(), verbose: bool = False):
        if not self.nuisance._fitted:
            self.nuisance.fit()
        self.compute_radius()
        batch_size = config.batch_size or self.n  # full batch default
        optimizer = optim.Adam(self.model.parameters(), lr=config.lr_theta)

        indices = np.arange(self.n)
        for outer in range(1, config.n_outer + 1):
            # Inner minimization over theta
            for _ in range(config.inner_epochs):
                batch_idx = np.random.choice(indices, size=batch_size, replace=False)
                Wb = self.nuisance.W[batch_idx]
                Db = self.nuisance.D[batch_idx]
                Yb = self.nuisance.Y[batch_idx]
                inp = torch.from_numpy(np.concatenate([Wb, Db.reshape(-1, 1), Yb.reshape(-1, 1)], axis=1)).float().to(self.device)
                inp.requires_grad_(False)
                z = self.model(inp)  # (batch, d+2)

                # Detach to numpy for analytical gradients
                z_np = z.detach().cpu().numpy()
                z_w = z_np[:, :self.d]
                z_d = z_np[:, self.d]
                z_y = z_np[:, self.d + 1]
                Wp = Wb + z_w / self.sqrt_n
                Dp = Db + z_d / self.sqrt_n
                Yp = Yb + z_y / self.sqrt_n

                # Compute debiased ATE values and gradients sample-wise
                g_vals = np.zeros(len(batch_idx))
                grad_z = np.zeros_like(z_np)
                lam = self.lambda_param
                for j in range(len(batch_idx)):
                    wj = Wp[j]
                    dj = Dp[j]
                    yj = Yp[j]
                    # value
                    g_val = self.nuisance.debiased_ate(wj, dj, yj).item()
                    g_vals[j] = g_val
                    # gradients of g wrt w,d,y
                    gw = self.nuisance.ate_gw(wj, dj, yj).reshape(-1)
                    gd = self.nuisance.ate_gd(wj, dj, yj).item()
                    gy = self.nuisance.ate_gy(wj, dj, yj).item()
                    # Chain rule (perturbation scaling 1/sqrt(n))
                    grad_z[j, :self.d] = lam * gw + 2 * z_w[j]
                    grad_z[j, self.d] = lam * gd + 2 * z_d[j]
                    grad_z[j, self.d + 1] = lam * gy + 2 * z_y[j]

                # Surrogate loss whose gradient wrt network params is grad_z / n
                grad_z_tensor = torch.from_numpy(grad_z).float().to(self.device)
                loss_surrogate = (z * grad_z_tensor).sum() / self.n  # scaling by n for objective definition
                optimizer.zero_grad()
                loss_surrogate.backward()
                if config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_clip)
                optimizer.step()

            # After inner epochs compute lambda gradient on FULL data (fresh forward)
            z_full, Wp_full, Dp_full, Yp_full = self._forward_full()
            g_full = np.array([self.nuisance.debiased_ate(Wp_full[i], Dp_full[i], Yp_full[i]).item() for i in range(self.n)])
            lam_grad = np.mean(g_full) * self.sqrt_n  # (1/n) Σ √n g_i
            self.lambda_param += config.lr_lambda * lam_grad  # unrestricted

            # Track final objective with current theta & lambda
            objective = (self.lambda_param * self.sqrt_n * g_full.mean() + np.mean(np.sum(z_full ** 2, axis=1)))
            self.history["lambda"].append(self.lambda_param)
            self.history["loss"].append(objective)
            self.history["lam_grad"].append(lam_grad)

            if verbose and (outer % config.print_every == 0 or outer == 1):
                print(f"[Outer {outer:04d}] lambda={self.lambda_param:.4f} obj={objective:.6f} lam_grad={lam_grad:.6f}", end='\r', flush=True)

        return self

    # ---- Final statistic ----
    def test_statistic(self) -> float:
        z_full, Wp_full, Dp_full, Yp_full = self._forward_full()
        g_full = np.array([self.nuisance.debiased_ate(Wp_full[i], Dp_full[i], Yp_full[i]).item() for i in range(self.n)])
        T = (self.lambda_param * self.sqrt_n * g_full.mean() + np.mean(np.sum(z_full ** 2, axis=1)))
        return float(T)

    def reject(self) -> Tuple[bool, float, float]:
        if self.radius is None:
            self.compute_radius()
        T = self.test_statistic()
        return T > self.radius, T, self.radius


def generate_data(n: int, theta: float = 0.0, beta: float = 0.5, seed: int = None, noise = "normal"):
    """Generate synthetic data for testing."""
    if seed is not None:
        np.random.seed(seed)
    
    d = 5
    W = np.random.uniform(-1, 1, (n, d))
    c0 = np.random.uniform(0, 1, (d,))
    c1 = np.random.uniform(0, 1, (d,))
    D = np.random.binomial(1, 1 / (1 + beta * np.exp(-np.matmul(W, c0) / d)))
    
    if noise == "normal":
        noise = np.random.normal(0, 1, size=n)
    elif noise == "uniform":
        noise = np.random.uniform(-1, 1, size=n)
    elif noise == "pareto":
        noise = np.random.pareto(a=2.2, size=n)
    else:
        raise ValueError("Unsupported noise type.")
    
    Y = (np.matmul(W, c1) + theta) * D  + noise
    return W, D, Y


def run_rejection_test(sample_sizes: List[int], n_repetitions: int = 50, theta: float = 0.0, 
                      alpha: float = 0.05, verbose: bool = True, hypothesis: str = 'null') -> pd.DataFrame:
    """
    Run rejection test for multiple sample sizes and repetitions.
    
    Args:
        sample_sizes: List of sample sizes to test
        n_repetitions: Number of repetitions for each sample size
        theta: True treatment effect (0.0 for null hypothesis)
        alpha: Significance level
        verbose: Whether to print progress
    
    Returns:
        DataFrame with results
    """
    results = []
    noise = "pareto"  # consistent noise type for all runs
    for n in sample_sizes:
        if verbose:
            print(f"\nTesting sample size n={n}")
        
        rejections = 0
        test_stats = []
        radii = []
        ate_estimates = []
        computation_times = []
        theta = 0 if hypothesis == 'null' else 1 / np.sqrt(n)  # Local alternative scaling
        
        # Initialize progress tracking
        if verbose:
            print(f"Running {n_repetitions} experiments for sample size n={n}...")
        
        total_time = 0
        
        for rep in range(n_repetitions):
            start_time = time.time()
            
            try:
                # Generate data with unique seed for each repetition  seed=42 + rep * 1000 + n
                W, D, Y = generate_data(n, theta=theta, noise = noise)
                
                # Create and fit model
                ci = NeuralWassCI(W, D, Y, method_propensity='logistic', alpha=alpha, seed=42 + rep)
                
                # Use faster training config for repetitive testing
                config = TrainingConfig(
                    n_outer=150,  # Reduced from 200
                    inner_epochs= 30,
                    lr_theta=5e-3,
                    lr_lambda=1e-3,
                    print_every=40,  # Reduced printing
                    batch_size=min(10, n//4)  # Smaller batch size for efficiency
                )
                
                ci.fit(config, verbose=False)
                
                # Get test results
                decision, T, radius = ci.reject()
                
                # Get ATE estimate
                ci.nuisance.fit()
                ate_est = ci.compute_radius()
                
                # Record results
                if decision:
                    rejections += 1
                
                test_stats.append(T)
                radii.append(radius)
                ate_estimates.append(ate_est)
                
                experiment_time = time.time() - start_time
                total_time += experiment_time
                computation_times.append(experiment_time)
                
                # Dynamic update: calculate current rejection rate
                current_rejection_rate = rejections / (rep + 1)
                
                # Refreshing display (overwrites previous line)
                if verbose:
                    decision_str = "REJECT" if decision else "ACCEPT"
                    progress_msg = (f"\rExperiment {rep+1:>3d}/{n_repetitions} | "
                                   f"Rejection Rate: {current_rejection_rate:>6.3f} | "
                                   f"Last: {decision_str:>6s} | "
                                   f"Test Stat: {T:>7.4f} | "
                                   f"Time: {total_time:>6.1f}s")
                    print(progress_msg, end='', flush=True)
                
            except Exception as e:
                raise
        
        # Print newline after progress is complete
        if verbose:
            print()  # Move to next line after progress display
        
        # Calculate final rejection rate
        rejection_rate = rejections / n_repetitions
        
        # Store results
        result = {
            'sample_size': n,
            'rejection_rate': rejection_rate,
            'mean_test_stat': np.nanmean(test_stats),
            'mean_radius': np.nanmean(radii),
            'mean_ate_estimate': np.nanmean(ate_estimates),
            'mean_computation_time': np.nanmean(computation_times),
            'std_test_stat': np.nanstd(test_stats),
            'std_ate_estimate': np.nanstd(ate_estimates),
            'n_successful_runs': n_repetitions - np.sum(np.isnan(test_stats)),
            'noise_type': noise
        }
        
        results.append(result)
        
        if verbose:
            print("-" * 80)
            print(f"FINAL RESULTS for n={n}:")
            print(f"  Final rejection rate: {rejection_rate:.3f} ({rejections}/{n_repetitions} rejections)")
            print(f"  Mean test statistic: {result['mean_test_stat']:.4f} ± {result['std_test_stat']:.4f}")
            print(f"  Mean radius: {result['mean_radius']:.4f}")
            print(f"  Mean ATE estimate: {result['mean_ate_estimate']:.4f} ± {result['std_ate_estimate']:.4f}")
            print(f"  Average computation time: {result['mean_computation_time']:.2f}s")
            print(f"  Total time for n={n}: {total_time:.2f}s")
            print(f"  Successful runs: {result['n_successful_runs']}/{n_repetitions}")
            print("=" * 80)
    
    return pd.DataFrame(results)


def plot_rejection_rates(results_df: pd.DataFrame, title: str = "Neural Wasserstein Test: Rejection Rates vs Sample Size"):
    """Plot rejection rates versus sample sizes."""
    plt.figure(figsize=(10, 6))
    
    # Plot rejection rates
    plt.plot(results_df['sample_size'], results_df['rejection_rate'], 'bo-', linewidth=2, markersize=8)
    
    # Add horizontal line at alpha=0.05
    plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
    
    plt.xlabel('Sample Size', fontsize=12)
    plt.ylabel('Rejection Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add text annotations for each point
    for i, row in results_df.iterrows():
        plt.annotate(f'{row["rejection_rate"]:.3f}', 
                    (row['sample_size'], row['rejection_rate']),
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    return plt.gcf()




# ==============
# Main execution
# ==============
if __name__ == "__main__":
    print("Neural Wasserstein CI Rejection Rate Analysis")
    print("=" * 60)
    np.random.seed(42)


    # Configuration
    sample_sizes = [500, 600, 700, 800]
    n_repetitions = 100
    theta = 0.0  # Null hypothesis: no treatment effect
    alpha = 0.05
    hypothesis = 'alternative'  # or 'alternative' for the alternative hypothesis

    print(f"Sample sizes: {sample_sizes}")
    print(f"Number of repetitions: {n_repetitions}")
    print(f"True treatment effect (theta): {theta}")
    print(f"Significance level (alpha): {alpha}")
    print("=" * 60)
    
    # Run the rejection test with dynamic updates
    start_time = time.time()
    results_df = run_rejection_test(sample_sizes, n_repetitions, theta, alpha, verbose=True, hypothesis=hypothesis)
    total_time = time.time() - start_time
    
    print(f"\nTOTAL COMPUTATION TIME: {total_time:.2f} seconds")
    print("\nFINAL RESULTS SUMMARY:")
    print("=" * 60)
    print(results_df.to_string(index=False, float_format='%.4f'))
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"results/neural_wass_rejection_results_{timestamp}_theta{theta}_hypothesis{hypothesis}_{results_df['noise_type'].iloc[0]}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    # Create plots
    print("\nGenerating plots...")
    
    # Plot 1: Rejection rates
    fig1 = plot_rejection_rates(results_df)
    plot1_file = f"results/neural_wass_rejection_rates_{timestamp}_theta{theta}_hypothesis{hypothesis}_{results_df['noise_type'].iloc[0]}.png"
    fig1.savefig(plot1_file, dpi=300, bbox_inches='tight')
    print(f"Rejection rate plot saved to: {plot1_file}")
    


    # Show plots
    plt.show()
    
    print("\nAnalysis complete!")
    print("=" * 60)