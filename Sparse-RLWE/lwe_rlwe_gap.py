from multiprocessing import cpu_count
from pprint import pprint

from lattice_estimator.estimator import *
from sage.all import oo

from lwe_rot_primal import rot_primal_hybrid

# lambda = 128
h_64_params = [(11, 25, 64, 3.2), (12, 52, 64, 3.2), (13, 99, 64, 3.2), (14, 219, 64, 3.2), (15, 431, 64, 3.2), (16, 930, 64, 3.2), (17, 2022, 64, 3.2)] 

# lambda = 128
h_128_params =  [(11, 42, 128, 3.2), (12, 82, 128, 3.2), (13, 165, 128, 3.2), (14, 337, 128, 3.2), (15, 700, 128, 3.2), (16, 1450, 128, 3.2), (17, 2900, 128, 3.2)] # 

# lambda = 128
h_192_params = [(11, 46, 192, 3.19), (12, 92, 192, 3.19), (13, 186, 192, 3.19), (14, 377, 192, 3.19), (15, 767, 192, 3.19)]

for (logn, logq, h, sigma) in h_64_params + h_128_params + h_192_params:
    
    h_half = h // 2
    
    print(f"{logn=}, {logq=}, {h=}, {sigma=}")
    print()
    params = LWE.Parameters(n=2**logn, q = 2**logq, Xs=ND.SparseTernary(p=h_half, m=h_half, n=2 ** logn), Xe=ND.DiscreteGaussian(stddev=sigma), m=oo)
    
    # run all estimating scripts
    if logn < 14:
        plain_estimate = LWE.estimate(params, quiet=True, jobs=cpu_count() // 4)
        print("\tplain lwe estimate:")
        print(f"\t{plain_estimate}")
        print()
        
    # estimator doesn't terminate for large values: just run the two best performing attacks (dual hybrid, bdd mitm hybrid)
    else:
        plain_primal_hybrid_estimate = LWE.primal_hybrid(params, babai=True, mitm=True)
        print(f"\t{plain_primal_hybrid_estimate=}")
        plain_dual_hybrid_estimate = LWE.dual_hybrid(params)
        print(f"\t{plain_dual_hybrid_estimate=}")
        print()
        
    # these are all RLWE; poly_degree = params.n
    poly_degree = params.n
    
    ring_no_mitm_estimate = rot_primal_hybrid(params, babai=False, mitm=False, poly_degree=poly_degree)
    print(f"\t{ring_no_mitm_estimate=}")
    
    ring_babai_mitm_estimate = rot_primal_hybrid(params, babai=True, mitm=True, mitm_heuristic="square root", poly_degree=poly_degree)
    print(f"\t{ring_babai_mitm_estimate=}")
    
    ring_svp_mitm_estimate = rot_primal_hybrid(params, babai=False, mitm=True, mitm_heuristic="square root", poly_degree=poly_degree)
    print(f"\t{ring_svp_mitm_estimate=}")
    print()
    print()
    

