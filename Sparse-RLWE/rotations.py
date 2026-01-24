import random
from itertools import combinations

def negacyclic_rotate(key, j):
    if j == 0:
        return key
    rotated_key = tuple([-x for x in key[-j:]]) + key[:-j]
    return rotated_key

def probability_with_negacyclic_rotation_enumeration_ternary(n, h, zeta, J):
    assert zeta == len(J)
    # make a (very large) set of all keys
    remaining_keys = set()
    for non_zeroes in combinations(range(n), h):
        for choices in range(2 ** h):
            choices = bin(choices + 2 ** h)[3:]
            s = [0] * n
            for sign, index in zip(choices, non_zeroes):
                s[index] = (1 if sign == '1' else -1)
            remaining_keys.add(tuple(s))
            
    print("# finished enumerating all keys")
    
    hit_keys = 0
    enumerated = 0
    total_keys = len(remaining_keys)
    counts_with_hg = {}
    proportion_keys = {}
    h_g = 0
    while h_g <= min(h, zeta) and len(remaining_keys) > 0:
        for guess_non_zero_indices in combinations(range(zeta), h_g):
            for choices in range(2 ** h_g):
                choices = bin(choices + 2 ** h_g)[3:]
                s_g = [0] * zeta
                for sign, index in zip(choices, guess_non_zero_indices):
                    s_g[index] = 1 if sign == '1' else -1
                s_g = tuple(s_g)
                new_remaining_keys = remaining_keys.copy()
                r = list(range(n))
                random.shuffle(r)
                for j in r:
                    for key in remaining_keys:
                        rot_key = negacyclic_rotate(key, j)
                        if all(rot_key[j] == s_g_j for j, s_g_j in zip(J, s_g)):
                            new_remaining_keys.remove(key)
                            hit_keys += 1
                    remaining_keys = new_remaining_keys.copy()
                    enumerated += 1
                    proportion_keys[enumerated] = hit_keys
                    if len(remaining_keys) == 0:
                        break
                if len(remaining_keys) == 0:
                    break
            if len(remaining_keys) == 0:
                break
        counts_with_hg[h_g] = hit_keys
        h_g += 1
    return proportion_keys, counts_with_hg, total_keys

def probability_no_rotation_ternary(n, h, zeta, J):
    assert zeta == len(J)
    # make a (very large) set of all keys
    remaining_keys = set()
    for non_zeroes in combinations(range(n), h):
        for choices in range(2 ** h):
            choices = bin(choices + 2 ** h)[3:]
            s = [0] * n
            for sign, index in zip(choices, non_zeroes):
                s[index] = (1 if sign == '1' else -1)
            remaining_keys.add(tuple(s))
            
    print("# finished enumerating all keys")
    
    hit_keys = 0
    enumerated = 0
    total_keys = len(remaining_keys)
    proportion_keys = {}
    counts_with_hg = {}
    h_g = 0
    while h_g <= min(h, zeta) and len(remaining_keys) > 0:
        for guess_non_zero_indices in combinations(range(zeta), h_g):
            for choices in range(2 ** h_g):
                choices = bin(choices + 2 ** h_g)[3:]
                s_g = [0] * zeta
                for sign, index in zip(choices, guess_non_zero_indices):
                    s_g[index] = 1 if sign == '1' else -1
                s_g = tuple(s_g)
                new_remaining_keys = remaining_keys.copy()
                for key in remaining_keys:
                    if all(key[j] == s_g_j for j, s_g_j in zip(J, s_g)):
                        new_remaining_keys.remove(key)
                        hit_keys += 1
                remaining_keys = new_remaining_keys.copy()
                enumerated += 1
                proportion_keys[enumerated] = hit_keys
                if len(remaining_keys) == 0:
                    break
            if len(remaining_keys) == 0:
                break
        counts_with_hg[h_g] = hit_keys
        h_g += 1
    return proportion_keys, counts_with_hg, total_keys
    
def plot_cumulative(rotation_proportion_keys, no_rotation_proportion_keys, n, h, zeta, total_keys, prefix="ternary"):
    x1, y1 = zip(*sorted(rotation_proportion_keys.items()))
    x2, y2 = zip(*sorted(no_rotation_proportion_keys.items()))
    
    y1 = [number / total_keys for number in y1]
    y2 = [number / total_keys for number in y2]

    import matplotlib.pyplot as plt
    plt.plot(x1, y1, label="with rotation")
    plt.plot(x2, y2, label="without rotation")
    
    plt.xlabel("number of keys guessed (|S|)")
    plt.ylabel("success probability (P[s in S])")
    plt.legend()

    plt.savefig(f"{prefix}_{n}_{h}_{zeta}.png")


if __name__ == "__main__":
    n = 32
    h = 4
    zeta = 12
    J = set(random.sample(range(n), zeta))

    print(f"n = {n}\nh = {h}\nzeta = {zeta}")
    print("J =", J)
    print()
    rotation_proportion_keys, rotation_hwt_counts, total_keys = probability_with_negacyclic_rotation_enumeration_ternary(n, h, zeta, J)
    print("rotation_proportion_keys =", rotation_proportion_keys)
    print("rotation_hwt_counts =", rotation_hwt_counts)
    print("total_keys =", total_keys)
    print()

    no_rotation_proportion_keys, no_rotation_hwt_counts, total_keys = probability_no_rotation_ternary(n, h, zeta, J)
    print("no_rotation_proportion_keys =", no_rotation_proportion_keys)
    print("no_rotation_hwt_counts =", no_rotation_hwt_counts)

    plot_cumulative(rotation_proportion_keys, no_rotation_proportion_keys, n, h, zeta, total_keys)
        
    