import numpy as np
import time
import concurrent.futures
import queue
import os
from scipy.stats import chi2
from scipy.optimize import minimize
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
import pandas as pd
import cvxpy as cp
from datetime import datetime

def generate_data(n: int, theta = 1, beta = 0.5):
    '''This function generates data of the standard causal inference model with n samples.'''
    ''' The data is of the form (W_i,D_i,Y_i), where W_i is the covariates, D_i is the treatment and Y_i is the outcome.'''
    # TBD
    d = 1 #10
    W = np.random.uniform(-2, 2, (n, d))  # np.random.normal(0, 1, (n, 1))
    # generate the treatment by expit(W)
    D = np.random.binomial(1, 1/( 1+ beta * np.exp(-np.sum(W, axis = 1)/d)))
    # generate the outcome
    Y = 2 * W + theta * D.reshape(-1,1) + np.random.normal(0, 1, (n,1)) #(np.sum(W, axis=1) + theta)* D + np.random.normal(0, 1, (n, ))
    return W, D, Y

class dml:
    '''This class is used to estimate the nuisnace function and calculate the pseduo-outcome'''
    def __init__(self, dataW: np.ndarray, dataD: np.ndarray, dataY: np.ndarray, alpha: float = 0.05):
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
        
    def set_ate_grad(self):
        '''This function computes the gradient of the average treatment effect with respect to w, d and y.'''
        self.ate_gd = lambda w, d, y: (y- self.outcome1(w))/self.propensity(w) + (y - self.outcome0(w))/(1 - self.propensity(w))
        self.ate_gy = lambda w, d, y: d / self.propensity(w) - (1 - d)/(1 - self.propensity(w))

    def ate_gw(self, w, d, y):
        g1, g2 = self.outcome_gradient(w, d)
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
            # run logistic regression TBD
            pass

    def propensity_gradient(self, w, gamma = 0.5):
        '''This function computes the gradient of the propensity score with respect to w.'''
        w = np.array(w).reshape(1,-1)
        W_train = self.W_est.reshape(self.n, -1)
        diff = np.tile(w, (len(W_train), 1)) - W_train
        K = np.exp(-gamma * np.sum(diff**2, axis=1))
        grad = -2 * gamma * diff * K[:, np.newaxis]
        alpha = self.kr_propensity.dual_coef_.reshape(-1)
        return np.dot(grad.T, alpha)

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

    def outcome_gradient(self, w, d, gamma = 0.5):
        '''This function computes the gradient of the outcome function with respect to w and d.'''
        w = np.array(w).reshape(1,-1)
        d = np.array(d).reshape(-1)
        W_train1 = self.W_est[self.D_est == 1,:]
        W_train0 = self.W_est[self.D_est == 0,:]
        
        # compute the graident of the outcome1 and outcome0
        diff1 = np.tile(w, (len(W_train1), 1)) - W_train1
        K_1 = np.exp(-gamma * np.sum(diff1**2, axis=1))

        diff0 = np.tile(w, (len(W_train0), 1)) - W_train0
        K_0 = np.exp(-gamma * np.sum(diff0**2, axis=1))
        
        grad1 = -2 * gamma * diff1 * K_1[:, np.newaxis]
        grad0 = -2 * gamma * diff0 * K_0[:, np.newaxis]

        alpha1 = self.kr_outcome1.dual_coef_.reshape(-1)
        alpha0 = self.kr_outcome0.dual_coef_.reshape(-1)
        
        return np.dot(grad1.T, alpha1), np.dot(grad0.T, alpha0)
    
class WassCI(dml):
    '''This class is used to compute the confidence interval of the average treatment effect.'''
    def __init__(self, dataW: np.ndarray, dataD: np.ndarray, dataY: np.ndarray, alpha: float = 0.05):
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



    def set_radius(self, alpha = 0.05):
        '''This function set the given radius for the confidence interval '''
        m = [self.debiased_ate(self.W[i, :], self.D[i], self.Y[i]) for i in range(self.n)]

        self.ate_est = np.mean(m)

        moment_var = np.var(m)
        d_m = len(self.W[0,:])

        # Calculate the gradient of moment
        grad_d = np.array([(self.Y[i] - self.outcome1(self.W[i, :]))/self.propensity(self.W[i, :]) + (self.Y[i] - self.outcome0(self.W[i, :]))/(1 - self.propensity(self.W[i, :])) for i in range(self.n)])
        grad_y = np.array([self.D[i] / self.propensity(self.W[i, :]) -(1 - self.D[i])/(1 - self.propensity(self.W[i, :])) for i in range(self.n)])
        grad_w = np.zeros((self.n, d_m))

        for i in range(self.n):
            grad_w[i,:] = self.ate_gw(self.W[i, :], self.D[i], self.Y[i])

        m_g = (np.linalg.norm(grad_w) ** 2 + np.linalg.norm(grad_d)** 2 + np.linalg.norm(grad_y)** 2) / self.n

        self.radius = chi2.ppf(1 - self.alpha, 1) * moment_var / (m_g * self.n) 
        
    
    # def obj_min(self, x):
    #     '''This function computes the objective function for the optimization problem.'''
    #     # x[0]: lambda, x[1:n+1] t_i, x[n+1:2n+1] w_i, x[2n+1:3n+1] d_i, x[3n+1:4n+1] y_i
    #     res = - x[0] * self.radius
    #     n  = self.n
    #     res += sum([self.debiased_ate(x[i + 1], x[i + 1 + n], x[i + 1 + 2 * n]) + x[0] * ((x[ 1 + i]-self.W[i, :])**2 + (x[n + 1 + i] - self.D[i]) ** 2 + (x[2 * n + 1 + i] - self.Y[i])**2)  for i in range(n)]) / n
    #     return -res

    # def obj_max(self, x):
    #     '''This function computes the objective function for the optimization problem.'''
    #     # x[0]: lambda, x[1:n+1] t_i, x[n+1:2n+1] w_i, x[2n+1:3n+1] d_i, x[3n+1:4n+1] y_i
    #     res = - x[0] * self.radius
    #     n  = self.n
    #     res += sum([-self.debiased_ate(x[i + 1], x[i + 1 + n], x[i + 1 + 2 * n]) + x[0] * ((x[ 1 + i]-self.W[i, :])**2 + (x[n + 1 + i] - self.D[i]) ** 2 + (x[2 * n + 1 + i] - self.Y[i])**2)  for i in range(n)]) / n
    #     return -res 
    
        
    def opt_sub_min(self, x_0, index, lam):
        '''The function solves the optimization problem for the subproblem.'''
        n = self.n
        #subobj = lambda x: self.debiased_ate(x[0], x[1], x[2]) + lam * ((x[0] - self.W[index])**2 + (x[1] - self.D[index])**2 + (x[2] - self.Y[index])**2)
        d_m = len(self.W[0,:])

        subobj_scale = lambda x: np.sqrt(n) * (self.debiased_ate(x[0:d_m]/np.sqrt(n) + self.W[index], x[d_m]/np.sqrt(n) + self.D[index], x[d_m+ 1 ]/np.sqrt(n) + self.Y[index]) - self.debiased_ate(self.W[index], self.D[index], self.Y[index])) \
        + lam * np.linalg.norm(x)**2

        # Use the scipy.optimize to solve the optimization problem

        bounds = [(None, None)] * d_m + [(-self.D[index] * np.sqrt(n), (1 - self.D[index]) * np.sqrt(n))] + [(None, None)]
        result = minimize(subobj_scale, x_0, bounds = bounds, method='L-BFGS-B', tol = 1e-1)
        x_out = result.x

        # Use the gradient descent to solve the optimization problem

        # lam_der = ((w - self.W[index])**2 + (d - self.D[index])**2 + (y - self.Y[index])**2)
        lam_der_scale = np.linalg.norm(x_out)**2

        fun = result.fun
        return lam_der_scale, fun


    def opt_lambda_min(self, lam, iter = 10000, lr = 1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, method = 'SGD'):
        '''This function solves the optimization problem for the lambda.'''
        n = self.n
        
        m_t = 0
        v_t = 0
        t = 0
        for i in range(iter):
            t += 1
            index = np.random.randint(n)
            #x_0 = [*self.W[index, :], self.D[index], self.Y[index]]
            x_0 = [0] * (len(self.W[0,:]) + 2)
            lam_der= self.opt_sub_min(x_0, index, lam)[0] - self.radius * n
            
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

        m = (-lam * self.radius * n + sum(self.opt_sub_min(x_0, i, lam)[1] for i in range(n)) / n)/np.sqrt(n) + sum(self.debiased_ate(self.W[i, :], self.D[i], self.Y[i]) for i in range(n)) / n
        return m

    def opt_sub_max(self, x_0, index, lam):
        '''The function solves the optimization problem for the subproblem.'''
        n = self.n
        d_m = len(self.W[0,:])

        #subobj = lambda x: -self.debiased_ate(x[0], x[1], x[2]) + lam * ((x[0] - self.W[index])**2 + (x[1] - self.D[index])**2 + (x[2] - self.Y[index])**2)
        
        subobj_scale = lambda x: np.sqrt(n) * (-self.debiased_ate(x[0:d_m]/np.sqrt(n) + self.W[index], x[d_m]/np.sqrt(n) + self.D[index], x[1 + d_m]/np.sqrt(n) + self.Y[index]) + self.debiased_ate(self.W[index], self.D[index], self.Y[index])) \
        + lam * (np.linalg.norm(x)**2)

        bounds = [(None, None)] * d_m + [(-self.D[index] * np.sqrt(n), (1 - self.D[index]) * np.sqrt(n))] + [(None, None)]
        result = minimize(subobj_scale, x_0, bounds = bounds, method='L-BFGS-B',tol = 1e-1)
        x_out = result.x
        lam_der = np.linalg.norm(x_out)**2
        fun = result.fun
        return lam_der, fun

    def opt_lambda_max(self, lam, iter = 10000, lr = 1e-3, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, method = 'SGD'):
        '''This function solves the optimization problem for the lambda.'''
        n = self.n
        m_t = 0
        v_t = 0
        t = 0

        for i in range(iter):
            t += 1
            index = np.random.randint(n)

            #x_0 = [*self.W[index, :], self.D[index], self.Y[index]]

            x_0 = [0] * (len(self.W[0,:]) + 2)
            lam_der= self.opt_sub_max(x_0, index, lam)[0] - self.radius * n
            
            if method == 'SGD':
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

        m = (-lam * self.radius * n + sum(self.opt_sub_min(x_0, i, lam)[1] for i in range(n)) / n)/np.sqrt(n) - sum(self.debiased_ate(self.W[i, :], self.D[i], self.Y[i]) for i in range(n)) / n
        return -m


    def ate_ci(self, iter = 1000, lr = 1e-3, method = 'SGD'):
        '''This function estimates the average treatment effect.'''
        
        n = self.n
        self.est_propensity()
        self.est_outcome()
        self.set_ate_grad()
        self.debiased_ate = lambda w, d, y: self.outcome1(w) - self.outcome0(w) + d * (y - self.outcome1(w)) / self.propensity(w) - (1-d) * (y - self.outcome0(w)) / (1 - self.propensity(w))

        self.set_radius()

        '''# Lower bound using scipy.optimize. Results: failed due to numerical issues.
        
        constraints = [{'type': 'eq', 'fun': lambda x: (self.ate_gd(x[i+1], x[i+1+n], x[i+1+ 2 * n]) + 2*x[0] * (x[i+1+n] - self.D[i])).item()} for i in range(n)] + \
        [{'type': 'eq', 'fun': lambda x: (self.ate_gy(x[i+1], x[i+1+n], x[i+1+ 2 * n]) + 2*x[0] * (x[i+1+2*n] - self.Y[i])).item()} for i in range(n)] + \
        [{'type': 'eq', 'fun': lambda x: (self.ate_gw(x[i+1], x[i+1+n], x[i+1+ 2 * n]) + 2*x[0] * (x[i+1] - self.W[i, :])).item()} for i in range(n)]

        bounds = [(0, None)] * (3 * n + 1) # + [(None, None)] * (n) + [(0, 1)] * (n) + [(None, None)] * (n)

        x0 = np.ones((3 * n + 1,)) / 2

        result = minimize(self.obj_min, x0, method='COBYQA', constraints=constraints, bounds=bounds)

        lower_bound = -result.fun

        # Upper bound

        constraints = [{'type': 'eq', 'fun': lambda x: (self.ate_gd(x[i+1], x[i+1+n], x[i+1+ 2 * n]) - 2*x[0] * (x[i+1+n] - self.D[i])).item()} for i in range(n)] + \
        [{'type': 'eq', 'fun': lambda x: (self.ate_gy(x[i+1], x[i+1+n], x[i+1+ 2 * n]) - 2*x[0] * (x[i+1+2*n] - self.Y[i])).item()} for i in range(n)] + \
        [{'type': 'eq', 'fun': lambda x: (self.ate_gw(x[i+1], x[i+1+n], x[i+1+ 2 * n]) - 2*x[0] * (x[i+1] - self.W[i, :])).item()} for i in range(n)] 

        result = minimize(self.obj_max, x0, method='SLSQP', constraints = constraints, bounds=bounds)

        unpper_bound = result.fun '''
        lower_bound = self.opt_lambda_min(1e2, iter = iter, lr = lr, method = method)
        upper_bound = self.opt_lambda_max(1e2, iter = iter, lr = lr, method = method)
        return lower_bound, upper_bound

class ELCI(dml):
    
    def set_radius(self, alpha = 0.05):   
        self.radius = chi2.ppf(1 - self.alpha, 1)
    
    def opt(self):
        ''''The function solve the optimization for the lower bound using scipt.optimize.'''
        n = self.n

        pseduo_outcome = np.array([self.debiased_ate(self.W[i, :], self.D[i], self.Y[i]) for i in range(n)]).reshape(-1)

        # solve the minimization problem for the lower bound
        # set the initial value
        p = cp.Variable(n)

        # set constraints
        constraints = [
            p >= 0, 
            cp.sum(p) == 1,
            2 * cp.sum(cp.log(p)) + 2 * n * cp.log(n) + self.radius >= 0]

        # set the objective function (dual)
        #obj_min = cp.Minimize(lam[0] + (2 * n * cp.log(n) + self.radius - 2 * n + 2 * cp.sum(cp.log(2 * lam[1]) - 2 * cp.log(lam[0] + pseduo_outcome))) * lam[1])
        # primal objective 
        obj_min = cp.Minimize(cp.sum(cp.multiply(p, pseduo_outcome)))

        # Define the problem for minimization
        problem_min = cp.Problem(obj_min, constraints)

        # Solve the problem
        problem_min.solve(solver=cp.SCS, max_iters = 1000000, warm_start=True, verbose = False, eps = 1e-2)
        result_min = problem_min.value

        # solve the maximization problem for the upper bound
        obj_max = cp.Maximize(cp.sum(cp.multiply(p, pseduo_outcome)))

       # Define the problem for maximization
        problem_max = cp.Problem(obj_max, constraints)

        # Solve the problem
        problem_max.solve(solver=cp.SCS, max_iters = 1000000, warm_start=True,  eps = 1e-2)

        # Get the results for maximization
        result_max = problem_max.value

        return result_min, result_max

    def ate_ci(self):
        '''The function constructs the confidence interval for the average treatment effect using EL function.'''
        n = self.n
        self.est_propensity()
        self.est_outcome()

        self.debiased_ate = lambda w, d, y: self.outcome1(w) - self.outcome0(w) + d * (y - self.outcome1(w)) / self.propensity(w) - (1-d) * (y - self.outcome0(w)) / (1 - self.propensity(w))

        self.set_radius()

        # solve the EL problem
        lower_bound, upper_bound = self.opt()

        return lower_bound, upper_bound


class normal_ci(dml):
    def ate_ci(self, iter = 1000, lr = 1e-3, method = 'SGD'):
        '''This function estimates the average treatment effect.'''
        
        n = self.n
        self.est_propensity()
        self.est_outcome()
        self.set_ate_grad()
        self.debiased_ate = lambda w, d, y: self.outcome1(w) - self.outcome0(w) + d * (y - self.outcome1(w)) / self.propensity(w) - (1-d) * (y - self.outcome0(w)) / (1 - self.propensity(w))


        m = [self.debiased_ate(self.W[i, :], self.D[i], self.Y[i]) for i in range(self.n)]

        self.ate_est = np.mean(m)
        self.radius = chi2.ppf(1 - self.alpha, 1)
        var = np.var(m)

        # quantile of the chi-square distribution
        lower_bound = self.ate_est - np.sqrt(self.radius * var/n)
        upper_bound = self.ate_est + np.sqrt(self.radius * var/n)

        return lower_bound, upper_bound

# set random seed
np.random.seed(1) 
#5

# # plot the propensity score [-2,2]
# w = np.linspace(-2, 2, 100)
# p = ci_wass.propensity(w)
# p_true = 1/(1 + 0.5 * np.exp(-w))
# plt.plot(w, p, label = 'Estimated Propensity score', color = 'red')
# plt.plot(w, p_true, label = 'True Propensity score', color = 'blue')
# plt.legend()
# plt.show()

test_n = 50
success_wass = 0
success_el = 0


iter = 4000 
lr = 1e-1
method = 'SGD'
alpha = 0.05


# create a pandas dataframe to store the results
results_wass = pd.DataFrame(columns = ['Lower Bound', 'Upper Bound', 'radius', 'ATE','Sample Size'])
results_el = pd.DataFrame(columns = ['Lower Bound', 'Upper Bound', 'radius', 'ATE','Sample Size'])
result_normal = pd.DataFrame(columns = ['Lower Bound', 'Upper Bound', 'radius', 'ATE'])

date_str = datetime.today().strftime("%Y%m%d")
os.makedirs('results', exist_ok=True)

'''
def process_test(i, theta, beta, iter, lr, method, date_str, test_n, result_queue):
    data = generate_data(5000, theta, beta)
    

    ci_wass = WassCI(dataW=data[:, 0], dataD=data[:, 1], dataY=data[:, 2])
    l, u = ci_wass.ate_ci(iter=iter, lr=lr, method=method)

    l = l.item()
    u = u.item()
    wass_result = (i, l, u, ci_wass.radius, theta, l <= 1 and u >= 1)
    print(f"The confidence interval of the average treatment effect using Wasserstein method is [{l}, {u}].")

    # use EL method
    ci_el = ELCI(dataW=data[:, 0], dataD=data[:, 1], dataY=data[:, 2])
    l, u = ci_el.ate_ci()
    el_result = (i, l, u, ci_el.radius, theta, l <= 1 and u >= 1)
    print(f"The confidence interval of the average treatment effect using EL method is [{l}, {u}].")

    result_queue.put((wass_result, el_result))

for theta in [-1, 1]:
    for beta in [0.5, 1]:
        success_wass = 0
        success_el = 0
        result_queue = queue.Queue()
        with concurrent.futures.ThreadPoolExecutor(max_workers = 50) as executor:
            futures = [executor.submit(process_test, i, theta, beta, iter, lr, method, date_str, test_n, result_queue) for i in range(test_n)]
            for future in concurrent.futures.as_completed(futures):
                wass_result, el_result = result_queue.get()
                i, l, u, radius, theta, wass_success = wass_result
                results_wass.loc[i] = [l, u, radius, theta]
                success_wass += wass_success

                i, l, u, radius, theta, el_success = el_result
                results_el.loc[i] = [l, u, radius, theta]
                success_el += el_success

        # Save results to CSV files
        results_wass.to_csv(f'results/Wass_CI_{date_str}_{theta}_{beta}_{iter}_{lr}_parallel.csv', index=False)
        results_el.to_csv(f'results/EL_CI_{date_str}_{theta}_{beta}_parallel.csv', index=False)

        print(f"The success rate of Wasserstein approach is {success_wass / test_n}.")
        print(f"The success rate of EL approach is {success_el / test_n}.")

'''
for samplesize in [5000]:
    for theta in [-1,1]:
        for beta in [0.5, 1]:
            for i in range(test_n):
                W, D, Y = generate_data(samplesize, theta, beta)
                
                # start_time = time.time()
                # ci_wass = WassCI(dataW = W, dataD = D, dataY = Y, alpha = alpha)
                # l, u = ci_wass.ate_ci(iter = iter, lr = lr, method = method)
                # end_time = time.time()
                # elapsed_time = end_time - start_time
                # print(f"Elapsed time: {elapsed_time:.2f} seconds")

                # l = l.item()
                # u = u.item()
                # # save the results to the dataframe
                # results_wass.loc[i] = [l, u, ci_wass.radius, theta, samplesize]
                # if l <= 1 and u >= 1:
                #     success_wass += 1
                # print(f"The confidence interval of the average treatment effect using Wasserstein method is [{l}, {u}].")

                # use EL method
                # start_time = time.time()
                # ci_el = ELCI(dataW = W, dataD = D, dataY = Y, alpha = alpha)
                # l, u = ci_el.ate_ci()
                # end_time = time.time()
                # elapsed_time = end_time - start_time
                # print(f"Elapsed time of EL : {elapsed_time:.2f} seconds")
                
                # results_el.loc[i] = [l, u, ci_el.radius, theta, samplesize]
                # if l <= 1 and u >= 1:
                #     success_el += 1
                # print(f"The confidence interval of the average treatment effect using EL method is [{l}, {u}].")

                
                # use normal method
                ci_normal = normal_ci(dataW = W, dataD = D, dataY = Y, alpha = alpha)
                l, u = ci_normal.ate_ci()
                result_normal.loc[i] = [l, u, ci_normal.radius, theta]
                print(f"The confidence interval of the average treatment effect using normal method is [{l}, {u}].")

                print(rf"Test {i} \ {test_n} is done.")
                

                # Append the results to the CSV files after each iteration
                # results_wass.iloc[[i]].to_csv(f'results/Wass_CI_{date_str}_{test_n}_{samplesize}_{alpha}_{iter}_{lr}_{method}_{theta}_{beta}_0.1.csv', mode='a', header=False, index=False)
                #results_el.iloc[[i]].to_csv(f'results/EL_CI_{date_str}_{test_n}_{samplesize}_{alpha}_{theta}_{beta}.csv', mode='a', header=False, index=False)
                result_normal.iloc[[i]].to_csv(f'results/Normal_CI_1D_{date_str}_{test_n}_{samplesize}_{alpha}_{theta}_{beta}.csv', mode='a', header=False, index=False)


            #print(f"The success rate of Wasserstein approach is {success_wass / test_n}.")
            #print(f"The success rate of EL approach is {success_el / test_n}.")


# # save the results to a csv file add date to the file name

# results_wass.to_csv(f'Wass_CI_f_{date_str}_{test_n}_{iter}_{lr}.csv')
# results_el.to_csv(f'EL_CI_f_{date_str}_{test_n}.csv')

