# This code provides upper and lower bounds on the precision loss variance when evaluating ridge regression updates homomorphically, with the following assumptions:
# - sum_i y^2_i > cy, sum_i x^2_ij > cx
# - alpha < 1
# As a consequence, we have
# (\alpha*M_jj + \alpha\lambda - 1)^2  = (1 - \alpha*M_jj - \alpha\lambda)^2 \geq (1 - \alpha - \alpha\lambda)^2.
# For further details, see the heuristics in appendix A.2, as well as sections 4.2, 4.3, and appendices C.2 and C.3.

# sigma^2 -> rho^2 (ring variance to real variance)
def sigma_to_rho(sigma, N, Delta):
    return sigma * N / (2*Delta**2)


def sigma_fresh(h, sigma):
    return sigma**2 # return (2*h + 1)*sigma**2


def sigma_round(h):
    return h/12 + 1/12


def sigma_keyswitch(N, q_over_P, sigma, h):
    sigma_ks_ = (q_over_P**2 * N * sigma ** 2)/12
    if (q_over_P != 1):
        sigma_ks_ += sigma_round(h)
    return sigma_ks_


def sigma_Y(n, c, N, Delta, q_over_P, sigma, h, lower):
    fresh = sigma_fresh(h, sigma)
    sigma_Y = n*N*fresh**2 / \
        (Delta**2) + sigma_keyswitch(N, q_over_P, sigma, h) + sigma_round(h)
    # if lower bound, add 2*c*fresh
    if (lower):
        sigma_Y += + 2*c * fresh
    # if upper bound, add 2*n*fresh
    if (not lower):
        sigma_Y += 2*n*fresh
    sigma_Y *= (1 / n) ** 2
    return sigma_Y


def sigma_M_jk(n, c, N, Delta, q_over_P, sigma, h, lower):
    fresh = sigma_fresh(h, sigma)
    sigma_M_jk = n*N*fresh**2 / \
        (Delta**2) + sigma_keyswitch(N, q_over_P, sigma, h) + sigma_round(h)
    # if lower bound, add 2*c*fresh
    if (lower):
        sigma_M_jk += + 2 * c * fresh
    # if upper bound, add 2*n*fresh
    if (not lower):
        sigma_M_jk += 2*n*fresh
    sigma_M_jk *= (1 / n) ** 2
    return sigma_M_jk


def sigma_M_jj(n, c, N, Delta, q_over_P, sigma, h, lower):
    fresh = sigma_fresh(h, sigma)
    sigma_M_jj = 2*n*N*fresh**2 / \
        (Delta**2) + sigma_keyswitch(N, q_over_P, sigma, h) + sigma_round(h)
    # if lower bound, add 4*cx*fresh
    if (lower):
        sigma_M_jj += + 4 * c * fresh
    # if upper bound, add 4*n*fresh
    if (not lower):
        sigma_M_jj += 4 * n * fresh
    sigma_M_jj *= (1 / n) ** 2
    return sigma_M_jj


def update_sigma_beta(sigma_beta, sigma_M_jk, sigma_Y, sigma_M_jj, M_jj_, alpha, lamb, d, N, Delta, sigma_ks, sigma_round, lower):
    sigma = N * (alpha**2) * sigma_beta * ((d - 1) * sigma_M_jk + sigma_M_jj)
    sigma /= (Delta ** 2)
    sigma += (alpha**2) * sigma_Y
    sigma += sigma_beta * M_jj_
    sigma += sigma_ks + sigma_round  # only ks once per iteration
    # if lower bounding, these are the only terms.
    if (lower):
        return sigma
    # if upper bounding we additionally have \alpha**2 *sum \sigma_M_jk * \beta_k^2
    if (not lower):
        sigma += (alpha ** 2) * (max(sigma_M_jk, sigma_M_jj)) / lamb
        sigma += (d - 1) * alpha ** 2 * sigma_beta
        return sigma


def sigma_k(k, c, n, alpha, lamb, d, N, Delta, sigma, h, lower):
    # for the first keyswitches, q_over_P = 1
    # these are either all lower bounds or all upper bounds
    sigma_Y_ = sigma_Y(n, c, N, Delta, 1, sigma,
                       h, True if lower else False)
    sigma_M_jk_ = sigma_M_jk(n, c, N, Delta, 1, sigma,
                             h, True if lower else False)
    sigma_M_jj_ = sigma_M_jj(n, c, N, Delta, 1, sigma,
                             h, True if lower else False)
    if k == 1:
        return alpha(1)**2 * sigma_Y_
    # we consume log Delta bits of q each iteration, while P remains constant. Therefore q_over_P =  Delta ** -(k - 1).
    q_over_P_ = pow(Delta, -(k - 1))
    sigma_ks_ = sigma_keyswitch(N, q_over_P_, sigma, h)
    sigma_round_ = sigma_round(h)
    # the sigma of the previous iteration
    sigma_beta_ = sigma_k(
        k - 1, c, n, alpha, lamb, d, N, Delta, sigma, h, lower)
    # calculate the necessary bound on (1 - alpha*M_jj - lambda*alpha)**2
    if (lower):
        M_jj_ = (1 - lamb*alpha(k) - alpha(k)) ** 2
    if (not lower):
        M_jj_ = max((lamb * alpha(k) - 1) ** 2, (lamb * alpha(k)) ** 2)
    return update_sigma_beta(sigma_beta_, sigma_M_jk_, sigma_Y_, sigma_M_jj_, M_jj_, alpha(k), lamb, d, N, Delta, sigma_ks_, sigma_round_, lower)


def rho_k(k, c, n, alpha, lamb, d, N, Delta, sigma, h, lower):
    return sigma_to_rho(sigma_k(k, c, n, alpha, lamb, d, N, Delta, sigma, h, lower), N, Delta)
