import sage
from estimator import *
from sage.all import oo
from lwe_rot_primal import rot_primal_hybrid

# log n, log q, h, sigma
sparse_params = [
    # (15, 767, 192, 3.19),
    # (16, 1553, 192, 3.19),
    # (17, 3104, 192, 3.19),
    # (16, 1518, 192, 3.2),
    (14, 420, 256, 3.2),
    # (13, 55, 31, 3.2),
    # (14, 100, 31, 3.2),
    # (15, 105, 31, 3.2),
    # (11, 52, 256, 3.2),
    # (12, 64, 256, 3.2),
    # (12, 104, 256, 3.2),
    # (13, 117, 256, 3.2),
    # (13, 178, 256, 3.2),
    # (15, 1332, 120, 3.2),
    # (16, 1585, 256, 3.2),
    # (15, 679, 192, 3.2),
    # (15, 780, 192, 3.2),
    # (16, 1555, 192, 3.2),
    # (16, 1345, 192, 3.2),
    (11, 64, 35, 2 ** 49),
    (11, 64, 38, 2 ** 47),
    (12, 64, 30, 2 ** 43),
    (13, 64, 25, 2 ** 43),
    (12, 64, 32, 2 ** 40),
    (13, 64, 26, 2 ** 41),
    # (17, 2365, 192, 3.2),
    # (15, 802, 256, 3.2)
]

cost_model = RC.MATZOV
for (logn, logq, h, sigma) in sparse_params:
    if sigma < 10:
        print(f"{logn=}\n{logq=}\n{h=}\n{sigma=}")
    else:
        log_sigma = sage.all.log(sigma, 2) // 1
        print(f"{logn=}\n{logq=}\n{h=}\n{log_sigma=}")
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
    