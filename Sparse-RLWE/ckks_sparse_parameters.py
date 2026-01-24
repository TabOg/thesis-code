import sage
from estimator import *
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
    print(f"{logn=}\n{logq=}\n{h=}\n{sigma=}")
    h_half = h // 2
    
    parameters = LWE.Parameters(n=2**logn, q = 2**logq, Xs=ND.SparseTernary(p=h_half, m=h_half, n=2 ** logn), Xe=ND.DiscreteGaussian(stddev=sigma), m=oo)

    rot_primal_estimate_no_mitm = rot_primal_hybrid(parameters, babai=False, mitm=False, red_cost_model=cost_model, ternary_search_=True)
    print("rotated, no mitm:")
    print(rot_primal_estimate_no_mitm)
    print()

    rot_primal_estimate_babai_mitm = rot_primal_hybrid(parameters, babai=True, mitm=True, mitm_heuristic="square root", red_cost_model=cost_model, ternary_search_=True)
    print("rotated, babai, mitm (square root):")
    print(rot_primal_estimate_babai_mitm)
    print()
    
    rot_primal_estimate_mitm = rot_primal_hybrid(parameters, babai=False, mitm=True, mitm_heuristic="square root", red_cost_model=cost_model, ternary_search_=True)
    print("rotated, mitm (square root):")
    print(rot_primal_estimate_mitm)
    print()
    

for (logn, logq, h) in ternary_params:
    print(f"{logn=}\n{logq=}\n{h=}\nternary error")
    h_half = h // 2
    parameters = LWE.Parameters(n=2**logn, q = 2**logq, Xs=ND.SparseTernary(p=h_half, m=h_half, n=2 ** logn), Xe=ND.Uniform(-1, 1), m=oo)
    
    rot_primal_estimate_no_mitm = rot_primal_hybrid(parameters, babai=False, mitm=False, red_cost_model=cost_model, ternary_search_=False)
    print("rotated, no mitm:")
    print(rot_primal_estimate_no_mitm)
    print()

    rot_primal_estimate_babai_mitm = rot_primal_hybrid(parameters, babai=True, mitm=True, mitm_heuristic="square root", red_cost_model=cost_model, ternary_search_=True)
    print("rotated, babai, mitm (square root):")
    print(rot_primal_estimate_babai_mitm)
    print()
    
    rot_primal_estimate_mitm = rot_primal_hybrid(parameters, babai=False, mitm=True, mitm_heuristic="square root", red_cost_model=cost_model, ternary_search_=True)
    print("rotated, mitm (square root):")
    print(rot_primal_estimate_mitm)
    print()