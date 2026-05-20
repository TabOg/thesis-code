import sage
from lattice_estimator.estimator import *
from sage.all import oo
from lwe_rot_primal import rot_primal_hybrid

# Gaussian secret
# log n, log q, h, sigma
gaussian_params = [
    (13, 55, 31, 3.2),
    (14, 100, 31, 3.2),
    (15, 105, 31, 3.2),
    (11, 52, 256, 3.2),
    (12, 64, 256, 3.2),
    (12, 104, 256, 3.2),
    (13, 117, 256, 3.2),
    (13, 178, 256, 3.2),
    (15, 767, 192, 3.19),
    (16, 1553, 192, 3.19),
    (17, 3104, 192, 3.19),
    (16, 1518, 192, 3.2),
    (16, 104, 32, 3.2),
    (15, 679, 192, 3.2),
    (15, 780, 192, 3.2),
    (16, 1555, 192, 3.2),
    (16, 1533, 192, 3.2),
    (16, 118, 30, 3.2),
    (16, 300, 128, 3.2),
    (15, 934, 64, 3.2),
    (16, 1496, 64, 3.2),
    
]

# ternary secret
# (logn, logq, h)
ternary_params = [
    (14, 404, 256)
]

cost_model = RC.MATZOV

for (logn, logq, h, sigma) in gaussian_params:
    print(f"{logn=}, {logq=}, {h=}, {sigma=}")
    h_half = h // 2
    
    params = LWE.Parameters(n=2**logn, q = 2**logq, Xs=ND.SparseTernary(p=h_half, m=h-h_half, n=2 ** logn), Xe=ND.DiscreteGaussian(stddev=sigma), m=oo)

    # these are all RLWE; poly_degree = params.n
    poly_degree = params.n
    
    ring_no_mitm_estimate = rot_primal_hybrid(params, babai=False, mitm=False, poly_degree=poly_degree)
    print(f"\t{ring_no_mitm_estimate=}")
    
    ring_babai_mitm_estimate = rot_primal_hybrid(params, babai=True, mitm=True, mitm_heuristic="square root", poly_degree=poly_degree)
    print(f"\t{ring_babai_mitm_estimate=}")
    
    ring_svp_mitm_estimate = rot_primal_hybrid(params, babai=False, mitm=True, mitm_heuristic="square root", poly_degree=poly_degree)
    print(f"\t{ring_svp_mitm_estimate=}")
    print()
    

for (logn, logq, h) in ternary_params:
    print(f"{logn=}, {logq=}, {h=}, ternary error")
    h_half = h // 2
    params = LWE.Parameters(n=2**logn, q = 2**logq, Xs=ND.SparseTernary(p=h_half, m=h_half, n=2 ** logn), Xe=ND.Uniform(-1, 1), m=oo)
    
    # these are all RLWE; poly_degree = params.n
    poly_degree = params.n
    
    ring_no_mitm_estimate = rot_primal_hybrid(params, babai=False, mitm=False, poly_degree=poly_degree)
    print(f"\t{ring_no_mitm_estimate=}")
    
    ring_babai_mitm_estimate = rot_primal_hybrid(params, babai=True, mitm=True, mitm_heuristic="square root", poly_degree=poly_degree)
    print(f"\t{ring_babai_mitm_estimate=}")
    
    ring_svp_mitm_estimate = rot_primal_hybrid(params, babai=False, mitm=True, mitm_heuristic="square root", poly_degree=poly_degree)
    print(f"\t{ring_svp_mitm_estimate=}")