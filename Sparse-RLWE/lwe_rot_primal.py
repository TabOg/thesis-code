# this code is adapted from the lattice estimator, https://github.com/malb/lattice-estimator

# in particular, we use the class `RotPrimalHybrid` which corresponds to the class `PrimalHybrid`

# everywhere we have made an edit to this class, we make use the flag #EDIT

from functools import partial

from sage.all import ceil, ZZ, oo, RR, sqrt, log, binomial, cached_function
from estimator.reduction import cost as costf
from estimator.reduction import delta as deltaf
from estimator.cost import Cost
from estimator.lwe_parameters import LWEParameters
from estimator.util import local_minimum, ternary_search
from estimator.simulator import normalize as simulator_normalize

from estimator.prob import amplify as prob_amplify
from estimator.prob import babai as prob_babai
from estimator.prob import mitm_babai_probability

from estimator.conf import red_cost_model as red_cost_model_default
from estimator.conf import red_shape_model as red_shape_model_default

from estimator.io import Logging

from estimator.lwe_primal import PrimalUSVP, PrimalHybrid, primal_usvp

class RotPrimalHybrid:
    @classmethod
    def babai_cost(cls, d):
        return Cost(rop=max(d, 1) ** 2)

    @classmethod
    def svp_dimension(cls, r, D, is_homogeneous=False):
        """
        Return required svp dimension for a given lattice shape and distance.

        :param r: squared Gram-Schmidt norms

        """
        from math import lgamma, log, pi

        def ball_log_vol(n):
            return (n / 2.0) * log(pi) - lgamma(n / 2.0 + 1)

        # If B is a basis with GSO profiles r, this returns an estimate for the shortest vector in the lattice
        # [ B | * ]
        # [ 0 |tau]
        # if the tau is None, the instance is homogeneous, and we omit the final row/column.
        def svp_gaussian_heuristic_log_input(r, tau):
            if tau is None:
                n = len(list(r))
                log_vol = sum(r)
            else:
                n = len(list(r)) + 1
                log_vol = sum(r) + 2 * log(tau)
            log_gh = 1.0 / n * (log_vol - 2 * ball_log_vol(n))
            return log_gh

        d = len(r)
        r = [log(x) for x in r]

        if d > 4096:
            # chosen since RC.ADPS16(1754, 1754).log(2.) = 512.168000000000
            min_i = d - 1754
        else:
            min_i = 0

        if is_homogeneous:
            tau = None
            for i in range(min_i, d):
                if svp_gaussian_heuristic_log_input(r[i:], tau) < log(D.stddev**2 * (d - i)):
                    return ZZ(d - (i - 1))
            return ZZ(2)

        else:
            # we look for the largest i such that (pi_i(e), tau) is shortest in the embedding lattice
            # [pi_i(B) | * ]
            # [   0    |tau]
            tau = D.stddev
            for i in range(min_i, d):
                if svp_gaussian_heuristic_log_input(r[i:], tau) < log(D.stddev**2 * (d - i) + tau ** 2):
                    return ZZ(d - (i - 1) + 1)
            return ZZ(2)

    @classmethod
    def svp_dimension_gsa(cls, d, log_total_vol, log_delta, D, is_homogeneous=False):
        """
        Return required svp dimension for a given lattice shape and distance.

        :param r: squared Gram-Schmidt norms

        """
        from math import lgamma, log, pi

        def log_projected_vol(i):
            return (d - i) / d * log_total_vol - i * (d - i) * log_delta

        def ball_log_vol(n):
            return (n / 2.0) * log(pi) - lgamma(n / 2.0 + 1)

        # If B is a BKZ reduced basis, this returns an estimate for the shortest vector in the lattice
        # [ B | * ]
        # [ 0 |tau]
        # under the GSA assumption, where total_vol is the volume of B, and delta is the root Hermite factor.
        # if the tau is None, the instance is homogeneous, and we omit the final row/column.
        def svp_gaussian_heuristic_gsa(i, tau):
            if tau is None:
                n = d - i
                log_vol = 2 * log_projected_vol(i)
            else:
                n = d - i + 1
                log_vol = 2 * log_projected_vol(i) + 2 * log(tau)
            log_gh = 1.0 / n * (log_vol - 2 * ball_log_vol(n))
            return log_gh

        if d > 4096:
            # chosen since RC.ADPS16(1754, 1754).log(2.) = 512.168000000000
            min_i = d - 1754
        else:
            min_i = 0

        if is_homogeneous:
            tau = None
            for i in range(min_i, d):
                if svp_gaussian_heuristic_gsa(i, tau) < log(D.stddev**2 * (d - i)):
                    return ZZ(d - (i - 1))
            return ZZ(2)
        else:
            # we look for the largest i such that (pi_i(e), tau) is shortest in the embedding lattice
            # [pi_i(B) | * ]
            # [   0    |tau]
            tau = D.stddev
            for i in range(min_i, d):
                if svp_gaussian_heuristic_gsa(i, tau) < log(D.stddev**2 * (d - i) + tau ** 2):
                    return ZZ(d - (i - 1) + 1)
            return ZZ(2)

    @staticmethod
    @cached_function
    def cost(
        beta: int,
        params: LWEParameters,
        zeta: int = 0,
        babai=False,
        mitm=False,
        m: int = oo,
        d: int = None,
        red_shape_model=red_shape_model_default,
        red_cost_model=red_cost_model_default,
        mitm_heuristic="square root",
        log_level=5,
    ):
        """
        Cost of the hybrid attack.

        :param beta: Block size.
        :param params: LWE parameters.
        :param zeta: Guessing dimension ζ ≥ 0.
        :param babai: Insist on Babai's algorithm for finding close vectors.
        :param mitm: Simulate MITM approach (√ of search space).
        :param m: We accept the number of samples to consider from the calling function.
        :param d: We optionally accept the dimension to pick.

        .. note :: This is the lowest level function that runs no optimization, it merely reports
           costs.

        """
        simulator = simulator_normalize(red_shape_model)
        if d is None:
            delta = deltaf(beta)
            d = min(ceil(sqrt(params.n * log(params.q) / log(delta))), m)
        d -= zeta

        if d < beta:
            # cannot BKZ-β on a basis of dimension < β
            return Cost(rop=oo)

        xi = PrimalUSVP._xi_factor(params.Xs, params.Xe)

        # 1. Simulate BKZ-β
        # We simulate BKZ-β on the dxd basis B_BKZ:
        # [q I_m |  A_{n - zeta}  ]
        # [  0   | xi I_{n - zeta}]
        # if we need to set it, r holds the simulated squared GSO norms after BKZ-β
        r = None
        bkz_cost = costf(red_cost_model, beta, d)

        # 2. Required SVP dimension η + 1
        # We select η such that (pi_{d - η + 1}(e | s_{n - zeta}), tau) is the shortest vector in
        # [pi(B_BKZ) | t ]
        # [    0     |tau]
        if babai:
            eta = 2
            svp_cost = PrimalHybrid.babai_cost(d)
        else:
            # we scaled the lattice so that χ_e is what we want
            if red_shape_model == "gsa":
                log_vol = RR((d - (params.n - zeta)) * log(params.q) + (params.n - zeta) * log(xi))
                log_delta = RR(log(deltaf(beta)))
                svp_dim = PrimalHybrid.svp_dimension_gsa(d, log_vol, log_delta, params.Xe, params._homogeneous)
            else:
                r = simulator(d, params.n - zeta, params.q, beta, xi=xi, tau=False, dual=True)
                svp_dim = PrimalHybrid.svp_dimension(r, params.Xe, is_homogeneous=params._homogeneous)
            eta = svp_dim if params._homogeneous else svp_dim - 1
            if eta > d:
                # Lattice reduction was not strong enough to "reveal" the LWE solution.
                # A larger `beta` should perhaps be attempted.
                return Cost(rop=oo)
            # we make one svp call on a lattice of rank eta + 1
            svp_cost = costf(red_cost_model, svp_dim, svp_dim)
            # when η ≪ β, lifting may be a bigger cost
            svp_cost["rop"] += PrimalHybrid.babai_cost(d - eta)["rop"]

        # 3. Search
        # We need to do one BDD call at least
        search_space, probability, hw = 1, 1.0, 0

        # MITM or no MITM
        # we have three options:
        # mitm = False: search space = n * plain_search_space
        # mitm = True, heuristic = "estimator": return n * sqrt(plain_search_space)
        # mitm = True, heuristic = "square root": return sqrt(n * plain_search_space)
        def ssf(plain_search_space, n):
            if not mitm:
                return n * plain_search_space
            if mitm and mitm_heuristic == "estimator":
                return RR(n * sqrt(plain_search_space))
            if mitm and mitm_heuristic == "square root":
                return RR(sqrt(n * plain_search_space))
            else:
                raise ValueError(f"unrecognised mitm heuristic: {mitm=}, {mitm_heuristic=}, expected heuristic one of `estimator`, `square root`.")
                
        # e.g. (-1, 1) -> two non-zero per entry
        base = params.Xs.bounds[1] - params.Xs.bounds[0]

        # EDIT: 
        if zeta:
            # we maintain the size and hitting probability of two search spaces:
            # - plain_search_space and plain_probability: the size and probability without rotation
            # - search_space and probability: the size and probability with rotation
            # = we have that search_space = n * plain_search_space and probability = 1 - (1 - plain_probability) ** n
            # the number of non-zero entries
            h = params.Xs.hamming_weight
            # we check each rotation of zeta 0 coordinates
            plain_search_space = 1
            search_space = params.n * plain_search_space
            # this is the probability of 0 on zeta coordinates
            plain_probability = RR(binomial(params.n - zeta, h) / binomial(params.n, h))
            probability = 1 - (1 - plain_probability) ** params.n
            hw = 1
            while hw < min(h, zeta):
                new_plain_search_space = binomial(zeta, hw) * base**hw
                new_search_space = params.n * new_plain_search_space
                if svp_cost.repeat(ssf(plain_search_space + new_plain_search_space, params.n))["rop"] >= bkz_cost["rop"]:
                    break
                search_space += new_search_space
                plain_search_space += new_plain_search_space
                
                plain_probability += RR(binomial(params.n - zeta, h - hw) * binomial(zeta, hw) / binomial(params.n, h))
                probability = RR(1 - (1 - plain_probability) ** params.n)
                hw += 1
            svp_cost = svp_cost.repeat(ssf(plain_search_space, params.n))

        if mitm and zeta > 0:
            if babai:
                if r is None:
                    r = simulator(d, params.n - zeta, params.q, beta, xi=xi, tau=False, dual=True)
                probability *= mitm_babai_probability(r, params.Xe.stddev)
            else:
                # TODO: the probability in this case needs to be analysed
                probability *= 1

        if eta <= 20 and d >= 0:  # NOTE: η: somewhat arbitrary bound, d: we may guess it all
            if r is None:
                r = simulator(d, params.n - zeta, params.q, beta, xi=xi, tau=False, dual=True)
            probability *= RR(prob_babai(r, sqrt(d) * params.Xe.stddev))

        ret = Cost()
        ret["rop"] = bkz_cost["rop"] + svp_cost["rop"]
        ret["red"] = bkz_cost["rop"]
        ret["svp"] = svp_cost["rop"]
        ret["beta"] = beta
        ret["eta"] = eta
        ret["zeta"] = zeta
        ret["|S|"] = search_space
        ret["d"] = d
        ret["prob"] = probability
        ret["max_hw"] = hw

        ret.register_impermanent(
            {"|S|": False, "max_hw": False},
            rop=True,
            red=True,
            svp=True,
            eta=False,
            zeta=False,
            prob=False,
        )

        # 4. Repeat whole experiment ~1/prob times
        if probability and not RR(probability).is_NaN():
            ret = ret.repeat(
                prob_amplify(0.99, probability),
            )
        else:
            return Cost(rop=oo)

        return ret

    @classmethod
    def cost_zeta(
        cls,
        zeta: int,
        params: LWEParameters,
        red_shape_model=red_shape_model_default,
        red_cost_model=red_cost_model_default,
        m: int = oo,
        babai: bool = True,
        mitm: bool = True,
        mitm_heuristic="square root",
        optimize_d=True,
        log_level=5,
        **kwds,
    ):
        """
        This function optimizes costs for a fixed guessing dimension ζ.
        """

        # step 0. establish baseline
        baseline_cost = primal_usvp(
            params,
            red_shape_model=simulator_normalize(red_shape_model),
            red_cost_model=red_cost_model,
            optimize_d=False,
            log_level=log_level + 1,
            **kwds,
        )
        Logging.log("bdd", log_level, f"H0: {repr(baseline_cost)}")

        f = partial(
            cls.cost,
            params=params,
            zeta=zeta,
            babai=babai,
            mitm=mitm,
            mitm_heuristic=mitm_heuristic,
            red_shape_model=red_shape_model,
            red_cost_model=red_cost_model,
            m=m,
            **kwds,
        )

        # step 1. optimize β
        with local_minimum(
            40, baseline_cost["beta"] + 1, precision=2, log_level=log_level + 1
        ) as it:
            for beta in it:
                it.update(f(beta))
            for beta in it.neighborhood:
                it.update(f(beta))
            cost = it.y

        Logging.log("bdd", log_level, f"H1: {cost!r}")

        # step 2. optimize d
        if cost and cost.get("tag", "XXX") != "usvp" and optimize_d:
            with local_minimum(
                params.n, cost["d"] + cost["zeta"] + 1, log_level=log_level + 1
            ) as it:
                for d in it:
                    it.update(f(beta=cost["beta"], d=d))
                cost = it.y
            Logging.log("bdd", log_level, f"H2: {cost!r}")

        if cost is None:
            return Cost(rop=oo)
        return cost

    def __call__(
        self,
        params: LWEParameters,
        babai: bool = True,
        zeta: int = None,
        mitm: bool = True,
        mitm_heuristic = "square root",
        red_shape_model=red_shape_model_default,
        red_cost_model=red_cost_model_default,
        log_level=1,
        ternary_search_=False,
        **kwds,
    ):
        """
        Estimate the cost of the hybrid attack and its variants.

        :param params: LWE parameters.
        :param zeta: Guessing dimension ζ ≥ 0.
        :param babai: Insist on Babai's algorithm for finding close vectors.
        :param mitm: Simulate MITM approach (√ of search space).
        :return: A cost dictionary

        The returned cost dictionary has the following entries:

        - ``rop``: Total number of word operations (≈ CPU cycles).
        - ``red``: Number of word operations in lattice reduction.
        - ``δ``: Root-Hermite factor targeted by lattice reduction.
        - ``β``: BKZ block size.
        - ``η``: Dimension of the final BDD call.
        - ``ζ``: Number of guessed coordinates.
        - ``|S|``: Guessing search space.
        - ``prob``: Probability of success in guessing.
        - ``repeat``: How often to repeat the attack.
        - ``d``: Lattice dimension.

        - When ζ = 0 this function essentially estimates the BDD strategy as given in [RSA:LiuNgu13]_.
        - When ζ ≠ 0 and ``babai=True`` this function estimates the hybrid attack as given in
          [C:HowgraveGraham07]_
        - When ζ ≠ 0 and ``babai=False`` this function estimates the hybrid attack as given in
          [SAC:AlbCurWun19]_

        EXAMPLES::

            >>> from estimator import *
            >>> params = schemes.Kyber512.updated(Xs=ND.SparseTernary(16))
            >>> LWE.primal_hybrid(params, mitm=False, babai=False)
            rop: ≈2^89.3, red: ≈2^88.8, svp: ≈2^87.7, β: 106, η: 18, ζ: 321, |S|: ≈2^39.7, d: 360, prob: ≈2^-26.9, ↻...

            >>> LWE.primal_hybrid(params, mitm=False, babai=True)
            rop: ≈2^88.4, red: ≈2^87.8, svp: ≈2^86.9, β: 98, η: 2, ζ: 321, |S|: ≈2^39.7, d: 347, prob: ≈2^-28.1, ↻...

            >>> LWE.primal_hybrid(params, mitm=True, babai=False)
            rop: ≈2^73.4, red: ≈2^72.5, svp: ≈2^72.3, β: 109, η: 15, ζ: 320, |S|: ≈2^82.8, d: 366, prob: 0.001, ↻...

            >>> LWE.primal_hybrid(params, mitm=True, babai=True)
            rop: ≈2^85.5, red: ≈2^84.5, svp: ≈2^84.5, β: 105, η: 2, ζ: 364, |S|: ≈2^85.0, d: 316, prob: ≈2^-23.2, ↻...

        TESTS:

        We test a trivial instance::

            >>> params = LWE.Parameters(2**10, 2**100, ND.DiscreteGaussian(3.19), ND.DiscreteGaussian(3.19))
            >>> LWE.primal_bdd(params)
            rop: ≈2^43.6, red: ≈2^43.6, svp: ≈2^21.9, β: 40, η: 2, d: 1514, tag: bdd

        We also test a LWE instance with a large error (coming from issue #106)::

            >>> LWE.primal_bdd(LWE.Parameters(n=256, q=12289, Xs=ND.UniformMod(2), Xe=ND.UniformMod(1024)))
            rop: ≈2^115.4, red: ≈2^41.3, svp: ≈2^115.4, β: 40, η: 336, d: 336, tag: bdd

            >>> LWE.primal_bdd(LWE.Parameters(n=700, q=2**64, Xs=ND.UniformMod(2), Xe=ND.UniformMod(2**59)))
            rop: ≈2^259.8, red: ≈2^42.8, svp: ≈2^259.8, β: 40, η: 854, d: 854, tag: bdd


        """

        if zeta == 0:
            tag = "bdd"
        else:
            tag = "hybrid"

        params = LWEParameters.normalize(params)

        # allow for a larger embedding lattice dimension: Bai and Galbraith
        m = params.m + params.n if params.Xs <= params.Xe else params.m

        f = partial(
            self.cost_zeta,
            params=params,
            red_shape_model=red_shape_model,
            red_cost_model=red_cost_model,
            babai=babai,
            mitm=mitm,
            mitm_heuristic=mitm_heuristic,
            m=m,
            log_level=log_level + 1,
        )

        if zeta is None:
            # Find the smallest value for zeta such that the square root of the search space for
            # zeta is larger than the number of operations to solve uSVP on the whole LWE instance
            # (without guessing).
            usvp_cost = primal_usvp(params, red_cost_model=red_cost_model)["rop"]
            zeta_max = params.n
            while (
                zeta_max < params.n and sqrt(params.Xs.resize(zeta_max).support_size()) < usvp_cost
            ):
                zeta_max += 1
            if ternary_search_:
                with ternary_search(0, min(zeta_max, params.n), log_level=log_level) as it:
                    for zeta in it:
                        it.update(f(zeta=zeta, optimize_d=False, **kwds))
                # TODO: this should not be required
                cost = min(it.y, f(0, optimize_d=False, **kwds))
            else:
                with local_minimum(0, min(zeta_max, params.n), log_level=log_level) as it:
                    for zeta in it:
                        it.update(f(zeta=zeta, optimize_d=False, **kwds))
                # TODO: this should not be required
                cost = min(it.y, f(0, optimize_d=False, **kwds))
        else:
            cost = f(zeta=zeta)

        cost["tag"] = tag
        cost["problem"] = params

        if tag == "bdd":
            for k in ("|S|", "prob", "repetitions", "zeta"):
                try:
                    del cost[k]
                except KeyError:
                    pass

        return cost.sanity_check()

    __name__ = "rot_primal_hybrid"


rot_primal_hybrid = RotPrimalHybrid()
