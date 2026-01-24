# This code calculates the sensitivity of ridge regression gradient descent updates under the following assumptions:
# |x_ij|,|y_i| \leq i
# Each update reduces the loss


from math import sqrt

# corresponds to Corollary 4.2
def update_delta(delta, alpha, n, d, lamb):
    return 2*alpha*(sqrt(d) + d*sqrt(lamb))/n + (abs(1 - lamb) + alpha*d)*delta


def delta_k(k, alpha, n, d, lamb):
    if k == 0:
        return 0
    if k == 1:
        return 2*alpha(1)*sqrt(d)/n
    return update_delta(delta_k(k - 1, alpha, n, d, lamb), alpha(k), n, d, lamb)
