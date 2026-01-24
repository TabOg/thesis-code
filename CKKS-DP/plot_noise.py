import math
from matplotlib import pyplot as plt
from scipy.stats import norm
import numpy as np

import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})
mpl.rcParams["font.size"] = 12 

num_rows = 10
num_features = 5
num_iterations = 20
num_trials = 1000

poly_modulus_degree = 512
Delta = 35
sigma_fresh = 10.5
sigma_round = 1 / 12 + poly_modulus_degree / 18

folder_name = str(num_rows) + "_" + str(num_features) + "_" + str(num_iterations) + "_" + str(Delta) + "_" + str(num_trials)
print(f"{folder_name=}")

for line in open(folder_name + "/y.txt"):
    y = line.strip().split()
    y = [float(i) for i in y]

X = []
for line in open(folder_name + "/X.txt"):
    X_i = line.strip().split()
    X_i = [float(i) for i in X_i]
    X.append(X_i)

Y = []
sigma_Y = []
for j in range(num_features):
    Y_ = 0
    sigma_ = 0
    for i in range(num_rows):
        Y_ += y[i] * X[i][j]
        sigma_ += X[i][j] ** 2 + y[i] ** 2
    Y.append(Y_ / num_rows)
    
    sigma_ = num_rows * poly_modulus_degree * sigma_fresh ** 2 / pow(2, 2 * Delta) + sigma_fresh * sigma_ + sigma_round
    sigma_ /= num_rows ** 2
    sigma_Y.append(sigma_)

M = []
sigma_M = []

for j in range(num_features):
    sigma_M_j = []
    M_j = []
    for k in range(num_features):
        M_ = 0
        sigma_ = 0
        for i in range(num_rows):
            M_ += X[i][j] * X[i][k]
            sigma_ += X[i][j] ** 2 + X[i][k] ** 2
        M_j.append(M_ / num_rows)
        
        if j == k:
            sigma_ = 2 * num_rows * poly_modulus_degree * sigma_fresh ** 2 / pow(2, 2 * Delta) + 2 * sigma_fresh * sigma_ + sigma_round
        else:
            sigma_ = sigma_fresh * sigma_ + sigma_round + num_rows * poly_modulus_degree * sigma_fresh ** 2 / pow(2, 2 * Delta)
        sigma_ /= num_rows ** 2
        sigma_M_j.append(sigma_)
    M.append(M_j)
    sigma_M.append(sigma_M_j)
    
lamb = 1
beta = [0] * num_features

for iteration in range(num_iterations):
    alpha = 1.0 / (iteration + 1)
    new_beta = [0] * num_features
    new_sigma_beta = [alpha ** 2 * sigma_Y_i for sigma_Y_i in sigma_Y]
    
    for j in range(num_features):
        new_beta[j] = (1 - lamb * alpha) * beta[j] + alpha * Y[j] - alpha * sum([beta_k * M_jk for (beta_k, M_jk) in zip(beta, M[j])])
    
    beta = new_beta
    if iteration != 0:
        for j in range(num_features):
            sigma_1 = alpha ** 2 * poly_modulus_degree * pow(2, -2* Delta) *sigma_M[j][j] * sigma_beta[j] + sigma_M[j][j] * (alpha * beta[j]) ** 2 + sigma_beta[j] * (1 - lamb * alpha - alpha * M[j][j]) ** 2
            sigma_3 = 0
            for k in range(num_features):
                if k == j:
                    continue
                else:
                    sigma_3 += alpha ** 2 * poly_modulus_degree * pow(2, -2* Delta) *sigma_M[j][k] * sigma_beta[k] + sigma_M[j][k] * (alpha * beta[k]) ** 2 + sigma_beta[j] * (alpha * M[j][k]) ** 2
            new_sigma_beta[j] += sigma_1 + sigma_3 + sigma_round
    sigma_beta = new_sigma_beta

print(beta)
print()

rho_beta = [poly_modulus_degree * sigma_beta_j / pow(2, 2 * Delta + 1) for sigma_beta_j in sigma_beta]

for line in open(folder_name + "/beta_msg.txt"):
    beta_msg = line.strip().split()
    beta_msg = [float(i) for i in beta_msg]
    
print(beta_msg)

beta_noisy = []

for line in open(folder_name + "/beta_ctxt.txt"):
    beta_ = line.strip().split()
    beta_ = [float(i) for i in beta_]
    beta_noisy.append(beta_)

beta_cols = [list(beta_i) for beta_i in zip(*beta_noisy)]

import pandas as pd

for (i, col) in enumerate(beta_cols):
    mu = sum(col) / len(col)
    print(f"{mu=}")
    var = sum([(j - mu) ** 2 for j in col]) / len(col)
    print(f"{var=}")
    print(f"{beta[i]=}")
    print(f"{rho_beta[i]=}")
    print(f"out by a factor of {var / rho_beta[i]}")
    
    counts, bin_edges, _ = plt.hist(col, bins="auto", density=True, histtype="stepfilled",
        color=(1.0, 0.0, 0.0),
         alpha=0.3,
         linewidth=1.0, label="Observed Density")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    x = np.linspace(bin_edges[0], bin_edges[-1], 1000)
    pdf = norm.pdf(x, loc=beta[i], scale=math.sqrt(rho_beta[i]))
    plt.plot(x, pdf, linewidth=2,  color=(0.0, 0.0, 1.0),label="Predicted Density")
    
    plt.xlabel(f"Value of $\\beta_{i + 1}$")
    plt.ylabel("Density")
    plt.legend()
    
    plt.savefig(folder_name + f"/beta_plot_{i}.pdf", bbox_inches="tight")
    plt.savefig(folder_name + f"/beta_plot_{i}.png", bbox_inches="tight")
    plt.clf()
    print()