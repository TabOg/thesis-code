from variance import sigma_fresh, sigma_k, sigma_M_jk, sigma_M_jj
from sensitivity import delta_k
from math import sqrt

# For further details, see section 4.4 and appendix C.4

def d_Y(sigma_fresh, n):
    return 2*sigma_fresh / (n**2)


def d_M_jk(sigma_fresh, n):
    return 2*sigma_fresh / (n**2)


def d_M_jj(sigma_fresh, n):
    return 4*sigma_fresh / (n**2)

# Given a bound on |\sigma_beta^2 - \sigma_beta'^2| = d_beta at iteration (k - 1), return a bound on |\sigma_beta^2 - \sigma_beta'^2| at iteration k.


def update_d_beta(k, d_beta, lamb, alpha, n, d, N, Delta, h, sigma):
    # upper bounds on the variances of M_jk, M_jj, and sigma_beta at iteration k - 1
    sigma_beta_ = sigma_k(k - 1, 0, n, alpha, lamb, d, N, Delta, sigma, h, False)
    sigma_M_jk_ = sigma_M_jk(n, 0, N, Delta, 1, sigma, h, False)
    sigma_M_jj_ = sigma_M_jj(n, 0, N, Delta, 1, sigma, h, False)

    # upper bounds on the difference |sigma^2_x - sigma^2_x'| for relevant x
    d_Y_ = d_Y(sigma_fresh(h, sigma), n)
    d_M_jk_ = d_M_jk(sigma_fresh(h, sigma), n)
    d_M_jj_ = d_M_jj(sigma_fresh(h, sigma), n)

    d_1 = d_beta*((d - 1)*sigma_M_jk_ + sigma_M_jj_)
    d_1 += sigma_beta_*((d-1) * d_M_jk_ + d_M_jj_)
    d_1 *= N*alpha(k)**2 / (Delta ** 2)

    d_2 = ((d-1)*sigma_M_jk_ + sigma_M_jj_) * 2 * \
        delta_k(k-1, alpha, n, d, lamb)/(sqrt(lamb))
    d_2 += sigma_beta_*((d-1)*d_M_jk_ + d_M_jj_)
    d_2 *= alpha(k)**2

    d_3 = (d-1)*(4/n)*sigma_beta_ + (d-1)*d_beta
    d_3 *= alpha(k)**2

    d_4 = sigma_beta_ * 2*alpha(k)*max(2, 2*alpha(k)*(lamb + 1))/n
    d_4 += d_beta * max((lamb*alpha(k) - 1)**2, (lamb*alpha(k))**2)

    d_5 = alpha(k)**2 *d_Y_

    return d_1 + d_2 + d_3 + d_4 + d_5

def d_beta_k(k, lamb, alpha, n, d, N, Delta, h, sigma):
    if k == 1:
        return alpha(1)**2 * d_Y(sigma_fresh(h, sigma), n)

    # d from the previous iteration
    d_beta_ = d_beta_k(k - 1, lamb, alpha, n, d, N, Delta, h, sigma)
    return update_d_beta(k, d_beta_, lamb, alpha, n, d, N, Delta, h, sigma)

def T_k(k, c, lamb, alpha, n, d, N, Delta, h, sigma):
    d_ = d_beta_k(k, lamb, alpha, n, d, N, Delta, h, sigma)
    s_min = sigma_k(k, c, n, alpha, lamb, d, N, Delta, sigma, h, True)

    return 1 + d_ / s_min