from sensitivity import *
from variance import *
from message_dependence import *
from math import log, sqrt, pi

# this is a standard deviation, aka not squared

def rho_from_epsilon_var_only(delta, sensitivity, epsilon):
    c = sqrt(2*log(1.25/delta))
    return c * sensitivity / epsilon

# In the variance only setting, the epsilon for which a given rho, sensitivity guarantees (epsilon,delta) differential privacy.
# corresponds to Theorem 3.1

def epsilon_var_only_DP(delta, rho, sensitivity):
    c = sqrt(2*log(1.25/delta))
    rho_ = sqrt(rho)
    return c * sensitivity / rho_

# given delta, dimension d, variance rho, sensitivity, and squared message dependence, return a lower bound on the epsilon required to guarantee (epsilon, delta) differential privacy.
# corresponds to Corollary 3.4

def epsilon_DP(delta, d, rho, sensitivity, msg_dep):
    K_squared = (sensitivity ** 2) / rho
    D = 2*log(1/delta)
    epsilon = sqrt(D*(0.5*d*(msg_dep - 1)+msg_dep**2 * K_squared))
    epsilon += 0.5*msg_dep*K_squared
    epsilon += 0.5*(msg_dep - 1) * (D + d)
    epsilon += 0.5*d*log(msg_dep)
    return epsilon

def plot_min_max_variance(max_k, c, n, d, N, Delta, sigma, h, lamb, alpha):
    for i in c:
        print('\\addplot+[mark = none] coordinates{')
        [print('(' + str(k + 1) + ',' + str(log(rho_k(k + 1, i, n, alpha,
               lamb, d, N, Delta, sigma, h, True), 2))+')') for k in range(max_k)]
        print('};\\addlegendentry{Minimum, $c = ' + str(i) + '$}')
    print('\\addplot+[mark = none] coordinates{')
    [print('(' + str(k + 1) + ',' + str(log(rho_k(k + 1, c, n, alpha,
           lamb, d, N, Delta, sigma, h, False), 2))+')') for k in range(max_k)]
    print('};\\addlegendentry{Maximum}')


def plot_sensitivity(max_k, n, d, alpha, lamb):
    print('\\addplot+[mark = none] coordinates{')
    [print('(' + str(k + 1) + ',' + str(log(delta_k(k + 1, alpha, n, d, lamb), 2))+')')
     for k in range(max_k)]
    print('};\\addlegendentry{Sensitivity}')


def plot_var_only_epsilon(max_k, delta, c, n, d, N, Deltas, sigma, h, lamb, alpha):
    for y in Deltas:
        print('\\addplot+[mark = none] coordinates{')
        for k in range(max_k):
            delta_k_ = delta_k(k + 1, alpha, n, d, lamb)
            rho_k_ = rho_k(k + 1, c, n, alpha, lamb, d, N, y, sigma, h, True)
            epsilon_k_ = epsilon_var_only_DP(delta, rho_k_, delta_k_)
            if (epsilon_k_ <= 1):
                print('(' + str(k + 1) + ', ' + str(log(epsilon_k_, 2)) + ')')
        print('};\\addlegendentry{$\log\Delta= ' + str(round(log(y, 2))) + '$}')

def plot_var_only_iteration(max_k, epsilons, delta, c, n, d, N, logDeltas, sigma, h, lamb, alpha):
    for e in epsilons:
        print('\\addplot+[mark = none] coordinates{')
        for y in logDeltas:
            Delta = pow(2, y)
            k_ = 0
            for k in range(max_k):
                delta_k_ = delta_k(k + 1, alpha, n, d, lamb)
                rho_k_ = rho_k(k + 1, c, n, alpha, lamb, d, N, Delta, sigma, h, True)
                epsilon_k_ = epsilon_var_only_DP(delta, rho_k_, delta_k_)
                if(epsilon_k_ <= e and k_ == 0): # crossed threshold for the first time, record iteration number            
                    k_ = (k + 1)
                if(epsilon_k_ > e and k_ != 0): # exceeded threshold again, set back to zero
                    k_ = 0
            if(k_ != 0):
                print('(' + str(round(log(Delta, 2))) + ', ' + str(k_) + ')')
        print('};\\addlegendentry{$\\varepsilon= ' + str(e) + '$}')

def plot_T(max_k, c, lamb, alpha, n, d, N, Deltas, h, sigma):
    for Delta in Deltas:
        print('\\addplot+[mark = none] coordinates{')
        [print('(' + str(k + 1) + ',' + str(T_k(k + 1, c, lamb, alpha, n, d, N, Delta, h, sigma)) +')') for k in range(max_k)]
        print('};\\addlegendentry{$\\log\Delta = ' + str(round(log(Delta,2))) + '$}')

def plot_epsilon(max_k, delta, c, n, d, N, Deltas, sigma, h, lamb, alpha):
    for Delta in Deltas:
        print('\\addplot+[mark = none] coordinates{')
        for k in range(max_k):
            delta_k_ = delta_k(k + 1, alpha, n, d, lamb)
            rho_k_ = rho_k(k + 1, c, n, alpha, lamb, d, N, Delta, sigma, h, True)
            T_k_ = T_k(k + 1, c, lamb, alpha, n, d, N, Delta, h, sigma)
            epsilon_k_ = epsilon_DP(delta, d, rho_k_, delta_k_, T_k_)
            print('(' + str(k + 1) + ', ' + str(log(epsilon_k_, 2)) + ')')
        print('};\\addlegendentry{$\log\Delta= ' + str(round(log(Delta, 2))) + '$}')
        
def worst_case_error(k, c, n, d, N, Delta, sigma, h, lamb, alpha):
    rho_k_upper_ = rho_k(k + 1, c, n, alpha, lamb, d, N, Delta, sigma, h, False)
    print("Worst Case Error = " + str(log(6*sqrt(rho_k_upper_), 2)) + ' bits')

def plot_expected_noise(max_k, delta, c, n, d, N, Delta, sigma, h, lamb, alpha):
    DP_for_free_ = []
    DP_ = []
    c = 2*log(1.25/delta)
    for k in range(max_k):
        delta_k_ = delta_k(k + 1, alpha, n, d, lamb) # sensitivity
        rho_k_ = rho_k(k + 1, c, n, alpha, lamb, d, N, Delta, sigma, h, True) # upper bound on variance
        T_k_ = T_k(k + 1, c, lamb, alpha, n, d, N, Delta, h, sigma)
        epsilon_k_ = epsilon_DP(delta, d, rho_k_, delta_k_, T_k_)
        DP_for_free_.append(sqrt(rho_k_ * 2 / pi))
        # in the ordinary case, total variance is given by rho_k^2 + c * sensitivity^2 / epsilon_k^2
        rho_ = (c * delta_k_ ** 2) / (epsilon_k_ ** 2)
        rho_ += rho_k_
        DP_.append(sqrt(rho_ * 2 / pi))
    print('\\addplot+[mark = none] coordinates {')
    for k in range(max_k):
        print('(', str(k + 1),',',str(log(DP_for_free_[k], 2)), ')')
    print('};\\addlegendentry{DP for Free}')
    print('\\addplot+[mark = none] coordinates {')
    for k in range(max_k):
        print('(', str(k + 1),',',str(log(DP_[k], 2)), ')')
    print('};\\addlegendentry{Postprocessing DP}')