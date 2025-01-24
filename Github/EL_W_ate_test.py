import numpy as np
import os
from datetime import datetime
import pandas as pd
from scipy.stats import chi2
import cvxpy as cp
import matplotlib.pyplot as plt
'''This file compares the Wasserstein test and the Empirical likelhood test for causal inference. The data is of the form (D_i,Y_i), where D_i is the treatment and Y_i is the outcome. The null test is that the treatment has no effect on the outcome. The alternative is that the treatment has an effect on the outcome. '''

def gen_data(n, p1, p2, q):
    '''Generates data for the causal inference problem. The data is of the form (D_i,Y_i), where D_i is the treatment and Y_i is the outcome. The null test is that the treatment has no effect on the outcome. The alternative is that the treatment has an effect on the outcome. '''
    D = np.random.binomial(1, p1, (n,1))
    Y = np.zeros((n,1))
    for i in range(n):
        if D[i] == 0:
            Y[i] = 2 * np.random.binomial(1, p2, 1) - 1
        else:
            if p2  + q > 1:
                raise ValueError('Error: p2 + q > 1')
            Y[i] = 2 * np.random.binomial(1, p2 + q, 1) - 1
    Y += np.random.normal(0, 0.1, (n,1))
    return np.concatenate((D,Y), axis=1)

def Wasserstein_test(data, alpha):
    '''Performs the Wasserstein test for the causal inference problem. The data is of the form T_i = Y_i(D_i/p1 - (1-D_i)/(1-p1)), where D_i is the treatment and Y_i is the outcome. The null test is that the treatment has no effect on the outcome. The alternative is that the treatment has an effect on the outcome. '''
    n = data.shape[0]
    reject = 0
    RW = (np.sum(data)/np.sqrt(n))**2
    var_hat = np.var(data)
    if RW >= var_hat * chi2.ppf(1 - alpha, df=1):
        reject = 1
    return reject

def EL_test(data, alpha):
    n = data.shape[0]
    reject = 0
    x_max = np.max(data)
    x_min = np.min(data)
    
    # Define the optimization variables (Langrange multipliers)
    lam = cp.Variable(2)

    # Define the constraints
    constraints = [lam[0] * x_max + lam[1] >= 0,
                   lam[0] * x_min + lam[1] >= 0]
    
    # Define the objective
    obj = cp.Maximize(1 - lam[1] + 1/n * cp.sum(cp.log(data * lam[0] + lam[1])))

    # Define the problem
    prob = cp.Problem(obj, constraints)

    # Solve the problem
    prob.solve(verbose=True)  # I don't know if specify the solver is necessary 

    # Compute the test statistic
    EL = prob.value * n

    '''# Verify if the strong duaity holds by solving the primal problem.
    # The primal problem is to project the empirical distribution to the target space using KL divergence. The decision variable are probability on the data points.

    # Define the optimization variables
    p = cp.Variable(n)

    # Define the constraints
    constraints = [p >= 0,
                   cp.sum(p) == 1,
                   cp.sum(cp.multiply(p, data)) == 0]
    
    # Define the objective
    obj = cp.Minimize(-cp.sum(cp.entr(n*p)))

    # Define the problem
    prob = cp.Problem(obj, constraints)

    # Solve the problem
    prob.solve()

    # Compute the test statistic
    EL_prime = prob.value
    print('EL: {:.3f}, EL_prime: {:.3f}'.format(EL, EL_prime))'''

    if EL >= chi2.ppf(1 - alpha, df=1):
        reject = 1
    return reject


# Main code
#n = [500, 1000, 2000, 3000 , 5000]
n = 5000
p1 = 0.1
p2 = 0.2
q = np.linspace(1/np.sqrt(n), min(0.5 - p2, 10/np.sqrt(n)), 20)

alpha = 0.05
test_n = 5000
if not os.path.exists('results_HT'):
    os.mkdir('results_HT')

# rej_W = pd.DataFrame(index = range(test_n + 1), columns = q)
# rej_EL = pd.DataFrame(index = range(test_n + 1), columns = q)
rej_W = pd.DataFrame(index = range(test_n + 1), columns = q)
rej_EL = pd.DataFrame(index = range(test_n + 1), columns = q)
date_str = datetime.today().strftime("%Y%m%d")

# for i in range(len(n)):
#     for j in range(test_n):
#         q = 1/np.sqrt(n[i])
#         data_raw = gen_data(n[i], p1, p2, q)
#         data = data_raw[:,1] * (data_raw[:,0]/p1 - (1 - data_raw[:,0])/(1 - p1))
#         rej_W.iloc[j,i] = Wasserstein_test(data, alpha)
#         rej_EL.iloc[j,i] = EL_test(data, alpha)
#     print('n = {:d}'.format(n[i]))
#     rej_W.iat[test_n,i] = np.mean(rej_W.iloc[:,i])
#     rej_EL.iat[test_n,i] = np.mean(rej_EL.iloc[:,i])
#     print('Wasserstein test reject rate: {:.3f}'.format(rej_W.iat[test_n,i]))
#     print('Empirical likelihood test reject rate : {:.3f}'.format(rej_EL.iat[test_n,i]))

for i in range(len(q)):
    for j in range(test_n):
        data_raw = gen_data(n, p1, p2, q[i])
        data = data_raw[:,1] * (data_raw[:,0]/p1 - (1 - data_raw[:,0])/(1 - p1))
        rej_W.iat[j,i] = Wasserstein_test(data, alpha)
        rej_EL.iat[j,i] = EL_test(data, alpha)
    print('n = {:d}, q2 = {:3f}'.format(n, q[i] + p2))
    print('Wasserstein test reject rate: {:.3f}'.format(np.mean(rej_W.iloc[i])))
    print('Empirical likelihood test reject rate : {:.3f}'.format(np.mean(rej_EL.iloc[i])))
    rej_W.iat[test_n,i] = np.mean(rej_W.iloc[:,i])
    rej_EL.iat[test_n,i] = np.mean(rej_EL.iloc[:,i])



rej_W.to_csv(f'results_HT/rej_W_{date_str}.csv')
rej_EL.to_csv(f'results_HT/rej_EL_{date_str}.csv')



# Plot the results
# plt.figure()
# plt.plot(q, reject_W, label='Wasserstein test')
# plt.plot(q, reject_EL, label='Empirical likelihood test')
# plt.xlabel('Magnitude of the perturbation')
# plt.ylabel('Reject rate under alternative')
# plt.title(f'Comparison of Wasserstein test and Empirical likelihood test, Sample size = {n}')
# plt.legend()
# plt.show()
