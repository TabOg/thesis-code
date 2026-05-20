This code is used to generate all the data in the chapter `A Gap Between the Concrete Hardness of Sparse Secret LWE and RLWE`. We include code and data in this folder. Below is a guide to the python script used to generate the numbers, as well as the location of the raw data.

Figure	| 	script			            |	 data 
-----------------------------------------------------------------------
Fig 4.2	|	simulate_probabilities.py	|	probabilities_data.txt
Tab 4.2	|	lwe_rlwe_gap.py            	|	gap.txt
Fig 4.3	|	ckks_sparse_parameters.py	|	ckks_sparse_estimates.txt

We use matplotlib during development, and have left these plotting functions in the code.

To estimate the runtime of our attack, we use the class `RotPrimalHybrid`, which imitates the estimator class `PrimalHybrid`. This code can be found in lwe_rot_primal.py.

