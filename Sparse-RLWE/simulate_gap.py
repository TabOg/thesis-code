from itertools import accumulate
from sage.all import binomial
from math import log
from matplotlib import pyplot as plt

def size_against_probability_plain(n, h, zeta, W=2):
    h_gs = range(0, min(h, zeta) + 1)
    
    # number of s_g \in [-W/2, ..., W/2]^zeta of weight == j
    fixed_hwt_size = lambda j: binomial(zeta, j) * W ** j
    # probability s_g has weight exactly j
    fixed_hwt_prob = lambda j: binomial(n - zeta, h - j) * binomial(zeta, j) / binomial(n, h)
    
    # number of s_g of weight <= h_g
    size_h_g = list(accumulate(map(fixed_hwt_size, h_gs)))
    # probability s_g has weight <= h_g
    prob_h_g = list(accumulate(map(fixed_hwt_prob, h_gs)))
    
    # once the probability becomes 1, truncate list
    prob_h_g = [prob for prob in prob_h_g]
    # log the search space size & truncate
    size_h_g = [log(x, 2) for x in size_h_g]
    plain_coords = list(zip(size_h_g, prob_h_g))
    
    return plain_coords

def size_against_probability_rotated(n, h, zeta, W=2):
    h_gs = range(0, min(h, zeta) + 1)
    
    # number of s_g \in [-W/2, ..., W/2]^zeta of weight == j
    fixed_hwt_size = lambda j: binomial(zeta, j) * W ** j
    # probability s_g has weight exactly j
    fixed_hwt_prob = lambda j: binomial(n - zeta, h - j) * binomial(zeta, j) / binomial(n, h)
    
    # number of s_g of weight <= h_g
    size_h_g = list(accumulate(map(fixed_hwt_size, h_gs)))
    # probability s_g has weight <= h_g
    prob_h_g = list(accumulate(map(fixed_hwt_prob, h_gs)))

    # when we rotate, the guessing set size increases by a factor of n, while the probability increases to 1 - (1 - p)^n
    size_h_g_rot = [log(n, 2) + log(size, 2) for size in size_h_g]
    prob_h_g_rot = [1 - (1 - p) ** n for p in prob_h_g]
    
    # log the search space size & truncate
    size_h_g_rot = [x for x in size_h_g_rot]
    rot_coords = list(zip(size_h_g_rot, prob_h_g_rot))
    
    return rot_coords

def plot_simulations(n, h, zeta, plain_coords, rot_coords):
    x_plain, y_plain = zip(*plain_coords)
    x_rot, y_rot = zip(*rot_coords)
    
    plt.plot(x_plain, y_plain, label="without rotation")
    plt.plot(x_rot, y_rot, label="with rotation")
    plt.legend()
    plt.savefig(f"{n}_{h}_{zeta}.png")
    plt.clf()
    
if __name__ == "__main__":
    for (n, h, zeta) in [(512, 32, 128), (2048, 32, 512), (2 ** 15, 64, 2 ** 13), (2 ** 16, 128, 2 ** 14)]:
        print(f"{n=}\n{h=}\n{zeta=}\n")
        plain_coords = size_against_probability_plain(n, h, zeta)
        rot_coords = size_against_probability_rotated(n, h, zeta)
        
        # truncate: to keep the x axis short. can skip
        max_x = plain_coords[-1][0]
        rot_coords = [rot_coord for rot_coord in rot_coords if rot_coord[0] <= max_x]
        
        print(f"{plain_coords=}")
        print()
        print(f"{rot_coords=}")
        
        # print(*rot_coords, sep="\n")
        # print()
        # print(*plain_coords, sep="\n")
        # print()
        # print()
        plot_simulations(n, h, zeta, plain_coords, rot_coords)
    
    
    
    

