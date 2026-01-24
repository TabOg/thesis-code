from experiments import *

# DP parameters
delta = 1 / (4096)  # n = 4096 -> want failure probability <= 1/n
epsilon = pow(10, -2)

# database parameters
n = 4096
d = 10
lamb = 1

# HE parameters
N = pow(2, 16)
Delta = pow(2, 25)
Deltas = [pow(2,15),pow(2,20),pow(2,25), pow(2,28)]
sigma = 3.2
h = 2 * N / 3

# other parameters
lamb = 1
def alpha(i): return 1/i

# x axis
max_k = 50

# plot_min_max_variance(max_k, [0, 1024], n, d, N, Delta, sigma, h, lamb, alpha) # Figure 1
# plot_sensitivity(max_k, n, d, alpha, lamb) # Figure 2
plot_var_only_epsilon(max_k, delta, 0, n, d, N, Deltas, sigma, h, lamb, alpha) # Figure 3
# plot_var_only_iteration(2*max_k, [0.001,0.01,0.1,1], delta, 0, n, d, N, range(10, 30), sigma, h, lamb, alpha) # Figure 4
# plot_T(max_k, 0, lamb, alpha, n, d, N, Deltas, h, sigma) # Figure 5
# plot_epsilon(max_k, delta, 1024, n, d, N, Deltas,sigma, h, lamb, alpha) # Figure 6
# worst_case_error(max_k, 0, n, d, N, Deltas[3], sigma, h, lamb, alpha) # worst case error quote
# plot_expected_noise(max_k, delta, 0, n, d, N, Deltas[2], sigma, h, lamb, alpha)

