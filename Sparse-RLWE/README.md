This code is used to generate all the data in the chapter `A Gap Between the Concrete Hardness of Sparse Secret LWE and
  RLWE`. We include code and data in this folder. Below is a guide to the python script used to generate the numbers, as well as the location of the raw data.

Figure	| 	script			|	 data 
-------------------------------------------------------------
Fig 1	|	rotations.py		|	data/cdf/32_2_12.txt
Fig 2 	|	hwt_heuristic.py	|	data/heuristic/heuristic_data.txt
Fig 3	|	simulate_gap.py		|	data/simulations/simulations_data.txt
Tab 1	|	lwe_rlwe_gap.py		|	data/security_estimates/gap/*
Tab 2	|	sparse_parameters.py	|	data/security_estimates/security_estimates.txt

We use matplotlib during development, and have left these plotting functions in the code.

To estimate the runtime of our attack, we use the class `RotPrimalHybrid`, which imitates the estimator class `PrimalHybrid`. This code can be found in lwe_rot_primal.py.

As discussed in our chapter, the estimator implementation of the primal hybrid has a security overestimation issue [1], sometimes giving a security overestimation of around 20 bits. For our results, we apply the WIP hot fix to this issue [2] for both the LWE estimates and our RLWE estimates. We include `estimator_diff.txt` which documents the modification to the estimator.

[1] https://github.com/malb/lattice-estimator/issues/171
[2] https://github.com/malb/lattice-estimator/pull/170

