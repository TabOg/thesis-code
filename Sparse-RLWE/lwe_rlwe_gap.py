from estimator import *
from sage.all import oo

from lwe_rot_primal import rot_primal_hybrid

# lambda = 128
h_64_params = [(11, 25, 64, 3.2), (12, 52, 64, 3.2), (13, 99, 64, 3.2), (14, 219, 64, 3.2), (15, 431, 64, 3.2), (16, 930, 64, 3.2), (17, 2022, 64, 3.2)] 

# lambda = 128
h_128_params =  [(13, 165, 128, 3.2), (14, 337, 128, 3.2), (15, 700, 128, 3.2), (16, 1450, 128, 3.2), (17, 2900, 128, 3.2)] # 

# lambda = 128
h_192_params = [(11, 46, 192, 3.19), (12, 92, 192, 3.19), (13, 186, 192, 3.19), (14, 377, 192, 3.19), (15, 767, 192, 3.19)]

for (logn, logq, h, sigma) in h_192_params:
    
    h_half = h // 2
    
    print(f"{logn=}\n{logq=}\n{h=}\n{sigma=}")
    print()
    params = LWE.Parameters(n=2**logn, q = 2**logq, Xs=ND.SparseTernary(p=h_half, m=h_half, n=2 ** logn), Xe=ND.DiscreteGaussian(stddev=sigma), m=oo)
    
    # run all estimating scripts
    if logn < 14:
        plain_estimate = LWE.estimate(params)
        print("plain lwe estimate:")
        print(plain_estimate)
        print()
        
    # estimator doesn't terminate for large values: just run the two best performing attacks (dual hybrid, bdd mitm hybrid)
    else:
        primal_hybrid_estimate = LWE.primal_hybrid(params, babai=True, mitm=True)
        print("plain primal hybrid estimate:")
        print(primal_hybrid_estimate)
        print()
        dual_hybrid_estimate = LWE.dual_hybrid(params)
        print("plain dual hybrid estimate:")
        print(dual_hybrid_estimate)
        print()
    
    ternary_search_ = False if logn < 14 else True
    ring_estimate = rot_primal_hybrid(params, babai=True, mitm=True, mitm_heuristic="estimator", ternary_search_=ternary_search_)
    print("ring estimate (mitm heuristic = estimator):")
    print(ring_estimate)
    print()
    print()
    
    ring_estimate = rot_primal_hybrid(params, babai=True, mitm=True, mitm_heuristic="square root", ternary_search_=ternary_search_)
    print("ring estimate (mitm heuristic = square root):")
    print(ring_estimate)
    print()
    print()
    

