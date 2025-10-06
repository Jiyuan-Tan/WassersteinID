import numpy as np
import os
from scipy.stats import chi2
from scipy.optimize import minimize
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
import pandas as pd
import cvxpy as cp
from datetime import datetime
from sklearn.linear_model import LogisticRegression
import random

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility across all random number generators."""
    np.random.seed(seed)
    random.seed(seed)
    # For any future tensorflow/torch usage
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

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

class dml:
    '''This class is used to estimate the nuisnace function and calculate the pseduo-outcome'''
    def __init__(self, dataW: np.ndarray, dataD: np.ndarray, dataY: np.ndarray, alpha: float = 0.05, method_propensity = 'logistic'):
        # The data is of the form (W,D,Y)
        # Sample spliting: use half of the same for the estimation of the propensity score and the outcome function and the other half for the estimation of the average treatment effect.
        self.n = dataW.shape[0] // 2

        self.W = dataW[:self.n,:]
        self.D = dataD[:self.n]
        self.Y = dataY[:self.n]

        self.W_est = dataW[self.n:]
        self.D_est = dataD[self.n:]
        self.Y_est = dataY[self.n:]       
        
        self.alpha = alpha
        self.method_propensity = method_propensity
        
    def set_ate_grad(self):
        '''This function computes the gradient of the average treatment effect with respect to w, d and y.'''
        self.ate_gd = lambda w, d, y: (y- self.outcome1(w))/self.propensity(w) + (y - self.outcome0(w))/(1 - self.propensity(w))
        self.ate_gy = lambda w, d, y: d / self.propensity(w) - (1 - d)/(1 - self.propensity(w))

    def ate_gw(self, w, d, y):
        g1, g2 = self.outcome_gradient(w)
        g3 = self.propensity_gradient(w)
        p = self.propensity(w)
        o1 = self.outcome1(w)
        o0 = self.outcome0(w)
        return  g1 - g2 + d * (-g1/p + (o1 - y)/p**2 * g3) - (1 - d) * (-g2/(1 - p) - (y - o0)/(1 - p)**2 * g3)

    def est_propensity(self, method = 'kernel'):
        '''This function estimates the propensity score (nuisance) using the kernel methods.'''

        if method == 'kernel':
            kr_propensity = KernelRidge(kernel = 'rbf', alpha = 0.1, gamma = 0.5)
            kr_propensity.fit(self.W_est.reshape(self.n,-1), self.D_est.reshape(-1,1))
            self.kr_propensity = kr_propensity
            self.propensity = lambda w: np.clip(self.kr_propensity.predict(w.reshape(1,-1)).reshape(-1,1), 1e-2, 1-1e-2)
        elif method == 'logistic':
            # run logistic regression to estimate the propensity score 
            log_reg = LogisticRegression()
            log_reg.fit(self.W_est.reshape(self.n,-1), self.D_est.ravel())  # Use ravel() to ensure 1D array
            self.log_reg = log_reg
            self.propensity = lambda w: np.clip(self.log_reg.predict_proba(w.reshape(1,-1))[:,1].reshape(-1,1), 1e-2, 1-1e-2)
        else:
            raise ValueError('The method is not supported.')


    def propensity_gradient(self, w, gamma = 0.5):
        '''This function computes the gradient of the propensity score with respect to w.'''
        w = np.array(w).reshape(1,-1)
        W_train = self.W_est.reshape(self.n, -1)
        if self.method_propensity == 'kernel':
            diff = np.tile(w, (len(W_train), 1)) - W_train
            K = np.exp(-gamma * np.sum(diff**2, axis=1))
            grad = -2 * gamma * diff * K[:, np.newaxis]
            alpha = self.kr_propensity.dual_coef_.reshape(-1)
            return np.dot(grad.T, alpha)
        elif self.method_propensity == 'logistic':
            # Return the gradient of logistic function
            p = self.propensity(w)
            return self.log_reg.coef_.reshape(-1) * p * (1 - p)
        else:
            raise ValueError('The method is not supported.')

    def est_outcome(self):
        '''This function estimates the outcome (nuisnace) using kernel methods.'''
        kr_outcome1 = KernelRidge(kernel = 'rbf', alpha = 0.1, gamma = 0.5)
        kr_outcome0 = KernelRidge(kernel = 'rbf', alpha = 0.1, gamma = 0.5)
        data_wd = self.W_est
        # Extract all the rows where D = 1 and D = 0
        W_train1 = data_wd[self.D_est == 1, :]
        Y_train1 = self.Y_est[self.D_est == 1].reshape(-1)
        W_train0 = data_wd[self.D_est == 0, :]
        Y_train0 = self.Y_est[self.D_est == 0].reshape(-1)

        # Fit the Kernel Ridge models
        kr_outcome1.fit(W_train1, Y_train1)
        kr_outcome0.fit(W_train0, Y_train0)
        
        self.kr_outcome1 = kr_outcome1
        self.kr_outcome0 = kr_outcome0

        self.outcome1 = lambda w: self.kr_outcome1.predict(w.reshape(1,-1))
        self.outcome0 = lambda w: self.kr_outcome0.predict(w.reshape(1,-1))

    def outcome_gradient(self, w, gamma=0.5):
        '''This function computes the gradient of the outcome functions with respect to w.
        
        For RBF kernel ridge regression: f(w) = Σᵢ αᵢ K(w, wᵢ)
        Gradient: ∇f(w) = Σᵢ αᵢ ∇K(w, wᵢ) = Σᵢ αᵢ (-2γ(w - wᵢ)) exp(-γ ||w - wᵢ||²)
        '''
        w = np.array(w).reshape(1, -1)
        W_train1 = self.W_est[self.D_est == 1, :]
        W_train0 = self.W_est[self.D_est == 0, :]
        
        # Compute gradients for outcome1 (treated group)
        diff1 = np.tile(w, (len(W_train1), 1)) - W_train1
        K_1 = np.exp(-gamma * np.sum(diff1**2, axis=1))
        grad1 = -2 * gamma * diff1 * K_1[:, np.newaxis]
        alpha1 = self.kr_outcome1.dual_coef_.reshape(-1)
        grad_outcome1 = np.dot(grad1.T, alpha1)

        # Compute gradients for outcome0 (control group)  
        diff0 = np.tile(w, (len(W_train0), 1)) - W_train0
        K_0 = np.exp(-gamma * np.sum(diff0**2, axis=1))
        grad0 = -2 * gamma * diff0 * K_0[:, np.newaxis]
        alpha0 = self.kr_outcome0.dual_coef_.reshape(-1)
        grad_outcome0 = np.dot(grad0.T, alpha0)
        
        return grad_outcome1, grad_outcome0
    
class WassCI(dml):
    '''This class is used to compute the confidence interval of the average treatment effect.'''
    def __init__(self, dataW: np.ndarray, dataD: np.ndarray, dataY: np.ndarray, alpha: float = 0.05, method_propensity = 'logistic'):
        # The data is of the form (W,D,Y)
        # Sample spliting: use half of the same for the estimation of the propensity score and the outcome function and the other half for the estimation of the average treatment effect.
        self.n = dataD.shape[0] // 2

        self.W = dataW[:self.n,:]
        self.D = dataD[:self.n]
        self.Y = dataY[:self.n]

        self.W_est = dataW[self.n:]
        self.D_est = dataD[self.n:]
        self.Y_est = dataY[self.n:]       
        
        self.alpha = alpha
        self.method_propensity = method_propensity



    def set_radius(self, alpha = 0.05):
        '''This function set the given radius for the confidence interval '''
        m = [self.debiased_ate(self.W[i, :], self.D[i], self.Y[i]) for i in range(self.n)]

        self.ate_est = np.mean(m)

        moment_var = np.var(m)
        d_m = len(self.W[0,:])

        # Calculate the gradient of moment
        grad_d = np.array([self.ate_gd(self.W[i, :], self.D[i], self.Y[i]) for i in range(self.n)])

        grad_y = np.array([self.ate_gy(self.W[i, :], self.D[i], self.Y[i]) for i in range(self.n)])

        grad_w = np.zeros((self.n, d_m))

        for i in range(self.n):
            grad_w[i,:] = self.ate_gw(self.W[i, :], self.D[i], self.Y[i])

        m_g = (np.linalg.norm(grad_w) ** 2 + np.linalg.norm(grad_d)** 2 + np.linalg.norm(grad_y)** 2) / self.n

        self.radius = chi2.ppf(1 - self.alpha, 1) * moment_var / (m_g)
        

        
    def opt_sub_min_efficient(self, x_0, indices, lam):
        '''Efficient version with cached gradient computations to avoid duplicate calculations.'''
        n = self.n
        d_m = len(self.W[0,:])
        sqrt_n = np.sqrt(n)
        lam_der_scaled = 0
        fun = 0

        for index in indices:
            # Pre-compute baseline value
            baseline_ate = self.debiased_ate(self.W[index], self.D[index], self.Y[index])
            
            # Cache for gradient computations
            _cache = {}
            
            def subobj_and_grad(x):
                '''Compute both objective and gradient simultaneously for efficiency'''
                cache_key = tuple(x)
                if cache_key in _cache:
                    return _cache[cache_key]
                
                pertub_w = x[0:d_m]/sqrt_n + self.W[index]
                pertub_d = x[d_m]/sqrt_n + self.D[index]
                pertub_y = x[d_m + 1]/sqrt_n + self.Y[index]
                
                # Objective computation
                current_ate = self.debiased_ate(pertub_w, pertub_d, pertub_y)
                ate_diff = current_ate - baseline_ate
                obj = sqrt_n * lam * ate_diff + np.linalg.norm(x)**2
                
                # Gradient computation (reusing ATE computation context)
                grad_w = self.ate_gw(pertub_w, pertub_d, pertub_y)
                grad_d = self.ate_gd(pertub_w, pertub_d, pertub_y).item()
                grad_y = self.ate_gy(pertub_w, pertub_d, pertub_y).item()
                
                gradient = np.zeros_like(x)
                gradient[0:d_m] = lam * grad_w + 2 * x[0:d_m]
                gradient[d_m] = lam * grad_d + 2 * x[d_m]
                gradient[d_m + 1] = lam * grad_y + 2 * x[d_m + 1]
                
                result = (obj, gradient, current_ate)
                _cache[cache_key] = result
                return result
            
            def obj_func(x):
                return subobj_and_grad(x)[0]
            
            def grad_func(x):
                return subobj_and_grad(x)[1]

            # Use minimize with cached objective and gradient
            bounds = [(None, None)] * (d_m + 2)
            result = minimize(obj_func, x_0, jac=grad_func, bounds=bounds, method='L-BFGS-B', tol=1e-3)
            x_out = result.x

            # Get final ATE value from cache or compute it
            final_cache_key = tuple(x_out)
            if final_cache_key in _cache:
                final_ate = _cache[final_cache_key][2]
            else:
                final_ate = self.debiased_ate(x_out[0:d_m] / sqrt_n + self.W[index], 
                                             x_out[d_m] / sqrt_n + self.D[index], 
                                             x_out[d_m + 1] / sqrt_n + self.Y[index])

            lam_der_scaled += final_ate
            fun += result.fun

        return lam_der_scaled / len(indices), fun / len(indices)

    def opt_sub_min(self, x_0, indices, lam):
        '''The function solves the optimization sub-problem for the subproblem with analytical gradients.'''
        n = self.n
        d_m = len(self.W[0,:])
        sqrt_n = np.sqrt(n)
        lam_der_scaled = 0
        fun = 0

        for index in indices:
            # Pre-compute baseline value to avoid repeated computation
            baseline_ate = self.debiased_ate(self.W[index], self.D[index], self.Y[index])
            
            def subobj_scale(x):
                pertub_w = x[0:d_m]/sqrt_n + self.W[index]
                pertub_d = x[d_m]/sqrt_n + self.D[index]
                pertub_y = x[d_m + 1]/sqrt_n + self.Y[index]
                
                ate_diff = self.debiased_ate(pertub_w, pertub_d, pertub_y) - baseline_ate
                return sqrt_n * lam * ate_diff + np.linalg.norm(x)**2
            
            def subobj_grad(x):
                '''Analytical gradient of the objective function'''
                pertub_w = x[0:d_m]/sqrt_n + self.W[index]
                pertub_d = x[d_m]/sqrt_n + self.D[index]
                pertub_y = x[d_m + 1]/sqrt_n + self.Y[index]
                
                # Get gradients from ATE functions
                grad_w = self.ate_gw(pertub_w, pertub_d, pertub_y)
                grad_d = self.ate_gd(pertub_w, pertub_d, pertub_y).item()
                grad_y = self.ate_gy(pertub_w, pertub_d, pertub_y).item()
                
                # Chain rule: d/dx = d/d(perturbation) * d(perturbation)/dx
                # Since perturbation = x/sqrt(n) + original, d(perturbation)/dx = 1/sqrt(n)
                gradient = np.zeros_like(x)
                gradient[0:d_m] = lam * grad_w + 2 * x[0:d_m]  # sqrt_n * (1/sqrt_n) = 1
                gradient[d_m] = lam * grad_d + 2 * x[d_m]
                gradient[d_m + 1] = lam * grad_y + 2 * x[d_m + 1]
                
                return gradient

            # Use the scipy.optimize to solve the optimization problem with gradient
            bounds = [(None, None)] * (d_m + 2)
            result = minimize(subobj_scale, x_0, jac=subobj_grad, bounds=bounds, method='L-BFGS-B', tol=1e-3)
            x_out = result.x

            # Compute the lambda derivative
            lam_der_scaled += self.debiased_ate(x_out[0:d_m] / sqrt_n + self.W[index], 
                                               x_out[d_m] / sqrt_n + self.D[index], 
                                               x_out[d_m + 1] / sqrt_n + self.Y[index])

            fun += result.fun

        return lam_der_scaled / len(indices), fun / len(indices)

    def opt_lambda_min(self, lam, iter = 1000, lr = 1e-3, batch_size = 5, beta1=0.9, beta2=0.999, epsilon=1e-8, method = 'SGD'):
        '''This function solves the minimization problem for the lambda.'''
        n = self.n
        
        m_t = 0
        v_t = 0
        t = 0
        for i in range(iter):
            t += 1
            indices = np.random.choice(n, batch_size, replace=False).reshape(-1)
            #x_0 = [*self.W[index, :], self.D[index], self.Y[index]]
            x_0 = [0] * (len(self.W[0,:]) + 2)
            lam_der = self.opt_sub_min(x_0, indices, lam)[0].item()

            if method == 'SGD':
                # SGD
                 lam = lam + lr * lam_der
            else:
                # Adam
                # Update biased first moment estimate
                m_t = beta1 * m_t + (1 - beta1) * lam_der
                # Update biased second raw moment estimate
                v_t = beta2 * v_t + (1 - beta2) * (lam_der ** 2)
                # Compute bias-corrected first moment estimate
                m_hat = m_t / (1 - beta1 ** t)
                # Compute bias-corrected second raw moment estimate
                v_hat = v_t / (1 - beta2 ** t)
                # Update lambda
                lam = lam - lr * m_hat / (np.sqrt(v_hat) + epsilon)

        m = self.opt_sub_min(x_0, range(n), lam)[1] + sum(self.debiased_ate(self.W[i, :], self.D[i], self.Y[i]) for i in range(n)) / np.sqrt(n)
        return m
    

    def batch_predict_outcomes(self, W_batch):
        """Batch prediction for outcomes - much faster than individual predictions"""
        outcomes1 = self.kr_outcome1.predict(W_batch).reshape(-1)
        outcomes0 = self.kr_outcome0.predict(W_batch).reshape(-1) 
        return outcomes1, outcomes0
    
    def batch_predict_propensity(self, W_batch):
        """Batch prediction for propensity scores"""
        if self.method_propensity == 'logistic':
            props = self.log_reg.predict_proba(W_batch)[:, 1]
        else:  # kernel method
            props = self.kr_propensity.predict(W_batch).reshape(-1)
        return np.clip(props, 1e-2, 1-1e-2)
    
    def max_min_lambda(self, lam, lr=1e-3, iter = 1000, batch = None):
        # Precompute constants outside the inner function
        n = self.n
        d_m = len(self.W[0,:])
        sqrt_n = np.sqrt(n)
        inv_sqrt_n = 1.0 / sqrt_n
        min_step = 50
        if batch is None:
            batch = n  # Default batch size if not provided

        def debiased_ate_avg(x):
            # Extract perturbations for all samples at once
            w_pert = x[:, :d_m] * inv_sqrt_n + self.W
            d_pert = x[:, d_m] * inv_sqrt_n + self.D
            y_pert = x[:, d_m + 1] * inv_sqrt_n + self.Y

            # Batch predictions - single call instead of n calls!
            outcomes1, outcomes0 = self.batch_predict_outcomes(w_pert)
            propensities = self.batch_predict_propensity(w_pert)
            
            # Vectorized debiased ATE calculation
            debiased_ates = (outcomes1 - outcomes0 + 
                           d_pert * (y_pert - outcomes1) / propensities - 
                           (1 - d_pert) * (y_pert - outcomes0) / (1 - propensities))
            
            return np.mean(debiased_ates)
        
        def min_obj(lam, x):
            # Vectorized approach with batch predictions - fastest!
            x_reshaped = x.reshape(n, d_m + 2)
            # Vectorized sum
            f = sqrt_n * lam * debiased_ate_avg(x_reshaped) + np.linalg.norm(x)**2 / n
            return f

        def min_grad_optimized(lam):
            """Ultra-optimized version of min_grad with minimal redundant computations"""
            d_z = d_m + 2
            z_agg = np.zeros(d_z * n)
            
            # Precompute all constants - FIXED: use lam directly, not lam/n
            lam_scalar = lam
            if hasattr(lam, 'item'):
                lam_scalar = lam_scalar.item()
            two = 2.0
            
            for j in range(min_step):
                # Sample batch indices once per iteration
                batch_indices = np.random.choice(n, size=batch, replace=False)
                
                # Process each sample in batch
                for k in batch_indices:
                    # Direct indexing without repeated calculations
                    start_idx = k * d_z
                    end_w = start_idx + d_m
                    d_idx = end_w
                    y_idx = end_w + 1
                    
                    # Extract current z values (avoid repeated slicing)
                    z_w = z_agg[start_idx:end_w]
                    z_d = z_agg[d_idx]
                    z_y = z_agg[y_idx]
                    
                    # Compute perturbations (reuse inv_sqrt_n)
                    pertub_w = z_w * inv_sqrt_n + self.W[k, :]
                    pertub_d = z_d * inv_sqrt_n + self.D[k]
                    pertub_y = z_y * inv_sqrt_n + self.Y[k]
                    
                    # Compute all gradients at once
                    grad_w = self.ate_gw(pertub_w, pertub_d, pertub_y).reshape(-1)
                    grad_d = self.ate_gd(pertub_w, pertub_d, pertub_y).item()
                    grad_y = self.ate_gy(pertub_w, pertub_d, pertub_y).item()
                    
                    # FIXED: Vectorized updates using lam directly (not lam/n)
                    z_agg[start_idx:end_w] -= lr * (lam_scalar * grad_w + two * z_w)
                    z_agg[d_idx] -= lr * (lam_scalar * grad_d + two * z_d)
                    z_agg[y_idx] -= lr * (lam_scalar * grad_y + two * z_y)
            
            # Ultra-fast final computation using vectorized operations
            z_reshaped = z_agg.reshape(n, d_z)
            
            # Batch compute final perturbations
            w_final = z_reshaped[:, :d_m] * inv_sqrt_n + self.W
            d_final = z_reshaped[:, d_m] * inv_sqrt_n + self.D
            y_final = z_reshaped[:, d_m + 1] * inv_sqrt_n + self.Y
            
            # Single batch prediction for all samples
            outcomes1, outcomes0 = self.batch_predict_outcomes(w_final)
            propensities = self.batch_predict_propensity(w_final)
            
            # Vectorized debiased ATE computation
            debiased_ates = (outcomes1 - outcomes0 + 
                           d_final * (y_final - outcomes1) / propensities - 
                           (1 - d_final) * (y_final - outcomes0) / (1 - propensities))
            
            # Fast norm computation using vectorized operations
            z_norms_squared = np.sum(z_reshaped**2, axis=1)
            
            # Final result
            m = np.mean(debiased_ates) * sqrt_n * lam + np.mean(z_norms_squared)
            return -m
        
        def min_obj_with_gradient(x):
                """Compute both objective and gradient for min_obj"""
                # Reshape x to (n, d_m + 2) for vectorized operations
                x_reshaped = x.reshape(n, d_m + 2)
                
                # Compute objective value
                obj_val = sqrt_n * lam * debiased_ate_avg(x_reshaped) + np.linalg.norm(x)**2 / n
                
                # Compute gradient
                gradient = np.zeros_like(x)
                
                # Gradient w.r.t. norm term: 2x/n
                gradient += 2 * x / n
                
                # Gradient w.r.t. debiased ATE term
                # This is more complex - we need gradients of debiased_ate_avg w.r.t. x
                ate_gradient = compute_debiased_ate_gradient(x_reshaped)
                gradient += sqrt_n * lam * ate_gradient / n
                
                return obj_val, gradient
            
        def compute_debiased_ate_gradient(x_reshaped):
            """Compute gradient of debiased_ate_avg w.r.t. x"""
            # Extract perturbations
            w_pert = x_reshaped[:, :d_m] * inv_sqrt_n + self.W
            d_pert = x_reshaped[:, d_m] * inv_sqrt_n + self.D
            y_pert = x_reshaped[:, d_m + 1] * inv_sqrt_n + self.Y
            
            # Compute gradients for each sample
            gradients = np.zeros_like(x_reshaped)
            
            for i in range(n):
                # Get gradients from ate_gw, ate_gd, ate_gy
                grad_w = self.ate_gw(w_pert[i], d_pert[i], y_pert[i])
                grad_d = self.ate_gd(w_pert[i], d_pert[i], y_pert[i]).item()
                grad_y = self.ate_gy(w_pert[i], d_pert[i], y_pert[i]).item()
                
                # Chain rule: d(debiased_ate)/dx = d(debiased_ate)/d(perturbation) * d(perturbation)/dx
                # Since perturbation = x * inv_sqrt_n + original, d(perturbation)/dx = inv_sqrt_n
                gradients[i, :d_m] = grad_w * inv_sqrt_n
                gradients[i, d_m] = grad_d * inv_sqrt_n  
                gradients[i, d_m + 1] = grad_y * inv_sqrt_n
            
            # Average across all samples and flatten
            return gradients.flatten()
        
        def obj_func(x):
            return min_obj_with_gradient(x)[0]
            
        def grad_func(x):
            return min_obj_with_gradient(x)[1]
        
        def max_obj(lam, x_0):
            n = self.n
            try:
                result = minimize(obj_func, x_0, jac=grad_func, method='L-BFGS-B',  tol=1e-3)
                lam_der = (result.fun - np.linalg.norm(result.x)**2 / n) / (lam)
                
                """# Perform Hessian analysis at the solution
                print(f"\n=== FAST HESSIAN ANALYSIS AT SOLUTION (lambda = {lam}) ===")
                
                # Check if the solution is a saddle point using fast analysis
                hessian_results = fast_hessian_analysis(self, result.x, lam, check_blocks=True)
                
                # Print results
                print_fast_analysis_results(hessian_results)
                
                # Check for saddle point warning
                if hessian_results['overall_analysis']['is_saddle']:
                    print("⚠️  WARNING: Solution is a SADDLE POINT!")
                    print("   Consider using different initialization or regularization.")
                else:
                    print("✅ Solution appears to be a proper critical point.")"""
                
                return -result.fun, lam_der, result.x
            except:
                print(f"Error in minimize for lambda = {lam}")
                raise
        # result = minimize(min_grad_optimized, lam, method='L-BFGS-B', tol = 1e-1)

        # Try to use gradient ascent to solve the maximization problem
        f = 0
        x_0 = [0] * n * (len(self.W[0,:]) + 2)
        thres = 1e-2

        for i in range(iter):
            f, lam_der, x_0 = max_obj(lam, x_0)  
            if np.abs(lam_der) < thres:
                break
            lam = lam + lr * lam_der
        return -f

    def ate_test(self, iter = 1000, lr = 1e-3, method = 'SGD'):
        '''This function estimates the average treatment effect.'''
        
        n = self.n
        self.est_propensity(method= self.method_propensity)
        self.est_outcome()
        self.set_ate_grad()
        self.debiased_ate = lambda w, d, y: self.outcome1(w) - self.outcome0(w) + d * (y - self.outcome1(w)) / self.propensity(w) - (1-d) * (y - self.outcome0(w)) / (1 - self.propensity(w))

        self.set_radius()
        start_time = datetime.now()
        if method == 'SGD':
            wpf = self.opt_lambda_min(0, iter = iter, lr = lr, method = method, batch_size = 1).item()
        elif method == 'scipy':
            wpf = self.max_min_lambda(-1e-3, lr = lr, iter = iter)
        
        end_time = datetime.now()
        rej = 1 if wpf > self.radius else 0
        # optimization time
        print(f"Result: {rej}, Optimization time (seconds): {(end_time - start_time).total_seconds():.2f}, wpf: {wpf:.4f}, radius: {self.radius:.4f}") 

        return rej

class ELCI(dml):
    
    def set_radius(self, alpha = 0.05):   
        self.radius = chi2.ppf(1 - self.alpha, 1)
    
    def opt(self):
        ''''The function solve the optimization for the lower bound using scipt.optimize.'''
        n = self.n

        pseudo_outcome = np.array([self.debiased_ate(self.W[i, :], self.D[i], self.Y[i]) for i in range(n)]).reshape(-1)

        reject = 0
        x_max = np.max(pseudo_outcome)
        x_min = np.min(pseudo_outcome)

        # Define the optimization variables (Langrange multipliers)
        lam = cp.Variable(2)

        # Define the constraints
        constraints = [lam[0] * x_max + lam[1] >= 0,
                    lam[0] * x_min + lam[1] >= 0]
        
        # Define the objective
        obj = cp.Maximize(1 - lam[1] + 1/n * cp.sum(cp.log(pseudo_outcome * lam[0] + lam[1])))

        # Define the problem
        prob = cp.Problem(obj, constraints)

        # Solve the problem: Use Mosek for better performance
        prob.solve(solver=cp.MOSEK, warm_start=True)  # I don't know if specify the solver is necessary

        # Compute the test statistic
        EL = prob.value * n

        return EL

    def test(self):
        '''The function is used for hypothesis testing for the average treatment effect using EL function.'''
        n = self.n
        self.est_propensity(method= self.method_propensity)
        self.est_outcome()

        self.debiased_ate = lambda w, d, y: self.outcome1(w) - self.outcome0(w) + d * (y - self.outcome1(w)) / self.propensity(w) - (1-d) * (y - self.outcome0(w)) / (1 - self.propensity(w))

        self.set_radius()

        # solve the EL problem
        EL = self.opt()

        rej = 1 if EL > self.radius else 0

        return rej

class normal_ci(dml):
    def ate_ci(self, iter = 1000, lr = 1e-3, method = 'SGD'):
        '''This function estimates the average treatment effect.'''
        
        n = self.n
        self.est_propensity(method= self.method_propensity)
        self.est_outcome()
        self.set_ate_grad()
        self.debiased_ate = lambda w, d, y: self.outcome1(w) - self.outcome0(w) + d * (y - self.outcome1(w)) / self.propensity(w) - (1-d) * (y - self.outcome0(w)) / (1 - self.propensity(w))


        m = [self.debiased_ate(self.W[i, :], self.D[i], self.Y[i]) for i in range(self.n)]

        self.ate_est = np.mean(m)
        self.radius = chi2.ppf(1 - self.alpha, 1)
        var = np.var(m)

        # quantile of the chi-square distribution
        lower_bound = self.ate_est - np.sqrt(self.radius * var / n)
        upper_bound = self.ate_est + np.sqrt(self.radius * var / n)

        return lower_bound, upper_bound

def run_single_test(n, theta, beta, alpha, iter, lr, method, method_propensity, estimators=['wasserstein', 'el', 'normal'], noise = "normal"):
    """
    Run a single hypothesis test for selected methods.
    
    Parameters:
    - estimators: list of estimators to run ['wasserstein', 'el', 'normal']
    
    Returns:
    - reject_wass: 1 if Wasserstein CI rejects H0, 0 otherwise (or None if not run)
    - reject_el: 1 if EL CI rejects H0, 0 otherwise (or None if not run)
    - reject_normal: 1 if Normal CI rejects H0, 0 otherwise (or None if not run)
    """
    # Generate data
    W, D, Y = generate_data(n, theta, beta, noise=noise)
    
    # Initialize results
    reject_wass = None
    reject_el = None
    reject_normal = None
    
    # Wasserstein CI
    if 'wasserstein' in estimators:
        try:
            wass_ci = WassCI(W, D, Y, alpha, method_propensity)
            wass_ci.est_propensity()
            wass_ci.est_outcome()
            wass_ci.set_ate_grad()
            reject_wass = wass_ci.ate_test(iter, lr, method)
        except:
            raise
    
    # Empirical Likelihood CI
    if 'el' in estimators:
        try:
            el_ci = ELCI(W, D, Y, alpha, method_propensity)
            el_ci.est_propensity()
            el_ci.est_outcome()
            reject_el = el_ci.test()
        except:
            reject_el = 0
    
    # Normal CI
    if 'normal' in estimators:
        try:
            normal_ci_obj = normal_ci(W, D, Y, alpha, method_propensity)
            normal_ci_obj.est_propensity()
            normal_ci_obj.est_outcome()
            lower, upper = normal_ci_obj.ate_ci()
            reject_normal = 1 if 0 < lower or 0 > upper else 0
        except:
            reject_normal = 0
    
    return reject_wass, reject_el, reject_normal


def run_simulation_study(param_name, param_values, fixed_params, test_n, alpha, iter, lr, method, method_propensity, estimators=['wasserstein', 'el', 'normal']):
    """
    Run simulation study varying one parameter.
    
    Parameters:
    - param_name: 'n' or 'theta'
    - param_values: array of values to test
    - fixed_params: dict with fixed parameter values
    - test_n: number of repetitions per parameter value
    - estimators: list of estimators to run ['wasserstein', 'el', 'normal']
    """
    results = {
        'param_values': param_values,
        'wasserstein_reject_rate': [],
        'el_reject_rate': [],
        'normal_reject_rate': []
    }
    
    for param_val in param_values:
        print(f"Testing {param_name} = {param_val}")
        
        # Set parameters for this test
        if param_name == 'n':
            n = param_val
            theta = fixed_params['theta']
            beta = fixed_params['beta']
        elif param_name == 'theta':
            n = fixed_params['n']
            theta = param_val
            beta = fixed_params['beta']
        
        # Run tests
        wass_successes = 0
        el_successes = 0
        normal_successes = 0
        
        for i in range(test_n):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{test_n}")
            
            reject_wass, reject_el, reject_normal = run_single_test(
                n, theta, beta, alpha, iter, lr, method, method_propensity, estimators
            )
            
            if reject_wass is not None:
                wass_successes += reject_wass
            if reject_el is not None:
                el_successes += reject_el
            if reject_normal is not None:
                normal_successes += reject_normal
        
        # Calculate rejection rates (only for selected estimators)
        wass_rate = wass_successes / test_n if 'wasserstein' in estimators else None
        el_rate = el_successes / test_n if 'el' in estimators else None
        normal_rate = normal_successes / test_n if 'normal' in estimators else None
        
        results['wasserstein_reject_rate'].append(wass_rate)
        results['el_reject_rate'].append(el_rate)
        results['normal_reject_rate'].append(normal_rate)
        
        # Print results only for selected estimators
        if 'wasserstein' in estimators:
            print(f"  Wasserstein rejection rate: {wass_rate:.3f}")
        if 'el' in estimators:
            print(f"  EL rejection rate: {el_rate:.3f}")
        if 'normal' in estimators:
            print(f"  Normal rejection rate: {normal_rate:.3f}")
        print()
    
    return results


def plot_results(results, param_name, fixed_params, save_path=None, estimators=['wasserstein', 'el', 'normal']):
    """Plot rejection rates vs parameter values for selected estimators."""
    plt.figure(figsize=(10, 6))
    
    param_values = results['param_values']
    
    # Plot only selected estimators
    if 'wasserstein' in estimators and any(r is not None for r in results['wasserstein_reject_rate']):
        wass_rates = [r for r in results['wasserstein_reject_rate'] if r is not None]
        wass_x = [param_values[i] for i, r in enumerate(results['wasserstein_reject_rate']) if r is not None]
        plt.plot(wass_x, wass_rates, 'o-', 
                 label='Wasserstein CI', linewidth=2, markersize=6, color='blue')
    
    if 'el' in estimators and any(r is not None for r in results['el_reject_rate']):
        el_rates = [r for r in results['el_reject_rate'] if r is not None]
        el_x = [param_values[i] for i, r in enumerate(results['el_reject_rate']) if r is not None]
        plt.plot(el_x, el_rates, 's-', 
                 label='Empirical Likelihood CI', linewidth=2, markersize=6, color='red')
    
    if 'normal' in estimators and any(r is not None for r in results['normal_reject_rate']):
        normal_rates = [r for r in results['normal_reject_rate'] if r is not None]
        normal_x = [param_values[i] for i, r in enumerate(results['normal_reject_rate']) if r is not None]
        plt.plot(normal_x, normal_rates, '^-', 
                 label='Normal CI', linewidth=2, markersize=6, color='green')
    
    plt.xlabel(f'{param_name.title()}')
    plt.ylabel('Rejection Rate')
    
    if param_name == 'n':
        title = f'Rejection Rates vs Sample Size (θ={fixed_params["theta"]}, β={fixed_params["beta"]})'
    else:
        title = f'Rejection Rates vs Treatment Effect (n={fixed_params["n"]}, β={fixed_params["beta"]})'
    
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to: {save_path}')
    
    plt.show()


def save_results_csv(results, param_name, filename, estimators=['wasserstein', 'el', 'normal']):
    """Save results to CSV file for selected estimators."""
    df_data = {param_name: results['param_values']}
    
    # Only add columns for selected estimators
    if 'wasserstein' in estimators:
        df_data['Wasserstein_rejection_rate'] = results['wasserstein_reject_rate']
    if 'el' in estimators:
        df_data['EL_rejection_rate'] = results['el_reject_rate']
    if 'normal' in estimators:
        df_data['Normal_rejection_rate'] = results['normal_reject_rate']
    
    df = pd.DataFrame(df_data)
    
    filepath = os.path.join('results', filename)
    df.to_csv(filepath, index=False)
    print(f'Results saved to: {filepath}')


def run_local_alternative_study(n_values, test_n, alpha, iter, lr, method, method_propensity, date_str, estimators=['wasserstein', 'el', 'normal'], hypothesis = "alternative", noise = "normal"):
    """
    Run study with local alternative theta = 1/√n.
    """
    estimator_names = [est.replace('_', ' ').title() for est in estimators]
    print(f"\nStudy 1: Rejection rates vs Sample Size (Local Alternative: θ=1/√n)")
    print(f"Running estimators: {', '.join(estimator_names)}")
    print("-" * 50)
    
    results = {
        'param_values': n_values,
        'wasserstein_reject_rate': [],
        'el_reject_rate': [],
        'normal_reject_rate': []
    }
    
    for n_val in n_values:
        if hypothesis == "null":
            theta_local = 0.0  # Null hypothesis: theta = 0
        elif hypothesis == "alternative":
            theta_local = 1.0 / np.sqrt(n_val)  # Local alternative: theta = 1/√n
        else:
            raise ValueError("Hypothesis must be 'null' or 'alternative'.")
        print(f"Testing n = {n_val}, θ = 1/√n = {theta_local:.4f}")
        
        # Run tests with local alternative
        wass_successes = 0
        el_successes = 0
        normal_successes = 0
        
        for i in range(test_n):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{test_n}")
            
            reject_wass, reject_el, reject_normal = run_single_test(
                n_val, theta_local, 0.5, alpha, iter, lr, method, method_propensity, estimators, noise=noise
            )
            
            if reject_wass is not None:
                wass_successes += reject_wass
            if reject_el is not None:
                el_successes += reject_el
            if reject_normal is not None:
                normal_successes += reject_normal
        
        # Calculate rejection rates
        wass_rate = wass_successes / test_n if 'wasserstein' in estimators else None
        el_rate = el_successes / test_n if 'el' in estimators else None
        normal_rate = normal_successes / test_n if 'normal' in estimators else None
        
        results['wasserstein_reject_rate'].append(wass_rate)
        results['el_reject_rate'].append(el_rate)
        results['normal_reject_rate'].append(normal_rate)
        
        # Print results only for selected estimators
        if 'wasserstein' in estimators:
            print(f"  Wasserstein rejection rate: {wass_rate:.3f}")
        if 'el' in estimators:
            print(f"  EL rejection rate: {el_rate:.3f}")
        if 'normal' in estimators:
            print(f"  Normal rejection rate: {normal_rate:.3f}")
        print()
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    if 'wasserstein' in estimators and any(r is not None for r in results['wasserstein_reject_rate']):
        plt.plot(n_values, [r for r in results['wasserstein_reject_rate'] if r is not None], 'o-', 
                 label='Wasserstein CI', linewidth=2, markersize=6, color='blue')
    
    if 'el' in estimators and any(r is not None for r in results['el_reject_rate']):
        plt.plot(n_values, [r for r in results['el_reject_rate'] if r is not None], 's-', 
                 label='Empirical Likelihood CI', linewidth=2, markersize=6, color='red')
    
    if 'normal' in estimators and any(r is not None for r in results['normal_reject_rate']):
        plt.plot(n_values, [r for r in results['normal_reject_rate'] if r is not None], '^-', 
                 label='Normal CI', linewidth=2, markersize=6, color='green')
    
    plt.xlabel('Sample Size (n)')
    plt.ylabel('Rejection Rate')
    plt.title('Rejection Rates vs Sample Size (Local Alternative: θ=1/√n, β=0.5)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = f'results/rejection_vs_n_{date_str}_{hypothesis}_{noise}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Plot saved to: {save_path}')
    plt.show()
    
    # Save results to CSV
    df_data = {
        'n': results['param_values'],
        'theta_1_over_sqrt_n': [1.0/np.sqrt(n) for n in results['param_values']]
    }
    
    if 'wasserstein' in estimators:
        df_data['Wasserstein_rejection_rate'] = results['wasserstein_reject_rate']
    if 'el' in estimators:
        df_data['EL_rejection_rate'] = results['el_reject_rate']
    if 'normal' in estimators:
        df_data['Normal_rejection_rate'] = results['normal_reject_rate']
    
    df = pd.DataFrame(df_data)
    filepath = os.path.join('results', f'rejection_vs_n_{date_str}_{hypothesis}_{noise}.csv')
    df.to_csv(filepath, index=False)
    print(f'Results saved to: {filepath}')
    
    return results


def run_theta_sensitivity_study(theta_values, fixed_n, test_n, alpha, iter, lr, method, method_propensity, date_str, estimators=['wasserstein', 'el', 'normal']):
    """
    Run study varying theta values around null hypothesis.
    """
    estimator_names = [est.replace('_', ' ').title() for est in estimators]
    print(f"\nStudy 2: Rejection rates vs Treatment Effect θ (n={fixed_n})")
    print(f"Running estimators: {', '.join(estimator_names)}")
    print("-" * 50)
    
    fixed_params = {'n': fixed_n, 'beta': 0.5}
    results = run_simulation_study(
        'theta', theta_values, fixed_params, test_n, alpha, iter, lr, method, method_propensity, estimators
    )
    
    # Plot and save results
    plot_results(results, 'theta', fixed_params, f'results/rejection_vs_theta_{date_str}.png', estimators)
    save_results_csv(results, 'theta', f'rejection_vs_theta_{date_str}.csv', estimators)
    
    return results


def run_power_analysis_study(theta_values, fixed_n, test_n, alpha, iter, lr, method, method_propensity, date_str, estimators=['wasserstein', 'el', 'normal']):
    """
    Run power analysis study for positive treatment effects.
    """
    estimator_names = [est.replace('_', ' ').title() for est in estimators]
    print(f"\nStudy 3: Power Analysis - Rejection rates vs θ (n={fixed_n})")
    print(f"Running estimators: {', '.join(estimator_names)}")
    print("-" * 50)
    
    fixed_params = {'n': fixed_n, 'beta': 0.5}
    results = run_simulation_study(
        'theta', theta_values, fixed_params, test_n, alpha, iter, lr, method, method_propensity, estimators
    )
    
    # Plot and save results
    plot_results(results, 'theta', fixed_params, f'results/power_vs_theta_{date_str}.png', estimators)
    save_results_csv(results, 'theta', f'power_vs_theta_{date_str}.csv', estimators)
    
    return results


if __name__ == '__main__':
    # Set random seed for reproducibility
    seed = 42
    np.random.seed(seed)
    
    # Simulation parameters
    test_n = 100  # Number of repetitions per parameter value
    alpha = 0.05  # Significance level
    iter = 1000  # Iterations for optimization
    lr = 1e-2  # Learning rate
    method = 'scipy'  # Optimization method
    method_propensity = 'logistic'  # Propensity score estimation method
    
    # CHOOSE WHICH ESTIMATORS TO COMPARE
    # Options: 'wasserstein', 'el', 'normal'
    # Examples:
    # estimators = ['wasserstein', 'el', 'normal']  # Run all three
    # estimators = ['wasserstein', 'el']          # Only Wasserstein and EL
    # estimators = ['wasserstein']                 # Only Wasserstein
    estimators = [ 'el', 'normal']  # Default: run all
    
    # Setup
    date_str = datetime.today().strftime("%Y%m%d")
    os.makedirs('results', exist_ok=True)
    
    estimator_names = [est.replace('_', ' ').title() for est in estimators]
    print("Causal Inference CI Comparison")
    print(f"Selected estimators: {', '.join(estimator_names)}")
    print("=" * 60)
    
    # Study 1: Local alternative (theta = 1/√n)
    n_values = np.array([500, 600, 700, 800])
    
    # results_local = run_local_alternative_study(
    #     n_values, test_n, alpha, iter, lr, method, method_propensity, date_str, estimators, hypothesis="null"
    # )

    results_local = run_local_alternative_study(
        n_values, test_n, alpha, iter, lr, method, method_propensity, date_str, estimators, hypothesis="null", noise="pareto"
    )
    
    # # Study 2: Theta sensitivity around null
    # theta_values = np.linspace(0, 1, 11)
    # results_theta = run_theta_sensitivity_study(
    #     theta_values, 3000, test_n, alpha, iter, lr, method, method_propensity, date_str, estimators
    # )
    
    # # Study 3: Power analysis
    # theta_power_values = np.linspace(0, 2, 11)
    # results_power = run_power_analysis_study(
    #     theta_power_values, 3000, test_n, alpha, iter, lr, method, method_propensity, date_str, estimators
    # )
    
    print("\n" + "=" * 60)
    print("All studies completed! Check the 'results' folder for plots and CSV files.")
    print(f"Results saved with date: {date_str}")
    print(f"Estimators compared: {', '.join(estimator_names)}")
    print("=" * 60)