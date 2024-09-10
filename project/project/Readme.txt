# Readme Text
###################################################################################################

# How to compile c code and cuda code:
gcc -o gabc GaBR_Final.c -lm
nvcc -o gab$(Version) GaB$(Version)_Final.cu
###################################################################################################
# How to run c code and cuda code:
./gabc
./gab$(Version)

###################################################################################################
# C code
The original provided C code has been modified, so that the generated codewords and receivedwords in each step are stored in their specific folder for fair comparison.
The code has been run with the below simple H matrix with rowDegree of 4 and columnDegree of 2, which is used in the literature for easier understanding:
1 1 0 0 1 0 0 1
0 1 1 1 0 0 1 0
1 0 0 0 0 1 1 1
0 0 1 1 1 1 0 0
The DataPassGBIter0, DataPassGB, CheckPassGB, APP_GB, ComputeSyndrome are called for the first two iterations only and the results are stored in "Matrix.txt" file 
to be checked with the CUDA results.

###################################################################################################
# CUDA code
First, we have the preprocessings in host side. Each process execution time is calculated and reported by CUDA events.
The H matrix has been loaded from the provided MatrixFileName in the run command and printed for justification.
Then, the columnIndex and rowPointer vectors are obtained as introduced in lectures for sparse matrix representation and verifed by restoring the H matrix from them.
The columnIndex is stored in a file to be fed to the C code.
Next, the interleaver vector which is the index number of columnIndex in column-wise order to be used for variable nodes computation in order.
Now, the device required matrices are allocated and transfered from the calculated values of the host.
Then, the initVNU, VNU, CNU, CheckDecide, ComputeSyndrome are launched for the first two iterations only same as the C code to check our kernels functionality.
Since ComputeSyndrome in C code is performing XOR on decide vector until it finds the first non-zero element and breaks the loop, we find out that this operation
can be parallelized by computing the OR of all Decide vector elements. This can be done both by atomicOR (ver1) or a reduction version (ver2) with pre-processing of 
first step when assigning the shared memory.
Next, the result device vectors are transfered to allocated host vectors and the printing and storing in files is performed for verification.
In final step, the host and device allocated memories are freed before terminating the program. 

###################################################################################################
# Funtionality verification
To check the functionality of our CUDA implemention, the results of each kernel, which are the VtoC, CtoV, Decide, and SYndrome vectors are stored in "Time.txt"
file with execution time of each kernel in every process. Then, the results are compared one by one, running for a smaller iterative process, e.g., for 2 codewords
each running for 2 iterations. When, the results are the same, then the programs are run with actual number of codewords and iteration. The error correction parameters
(NbEr, NbTested, IterAver, NbUndec, ...) should be the same.

###################################################################################################
# Expected output
# Run simulation for alpha from 0.06 to 0.01 with the step of 0.01
# Each result (error correction characteristics and execution times) are stored in their specific file with the following format (Baseline result).
# Output file name for C code: $(inputfilename)c_Res.txt
# Output file name for CUDA code: $(inputfilename)$(Version)_Res.txt

-----------------------------------------------------Gallager B-----------------------------------------------------
alpha	  NbEr(BER)			NbFer(FER)		 Nbtested	IterAver(Itermax)	NbUndec(Dmin)
0.06000	27044(0.17989038)	100(0.86206895)	    116		89.09(42)		    0(100000)
0.05000	12516(0.04688062)	100(0.48543689)	    206		55.05(36)		    0(100000)
0.04000	 1235(0.00136328)	100(0.14306152)	    699		20.86(60)		    0(100000)
0.03000	  422(0.00009665)	100(0.02968240)	   3369		7.14(63)		    0(100000)
0.02000	  382(0.00002005)	100(0.00680226)	  14701		3.33(11)		    0(100000)

Execution time for Alpha0: 8736.909342 (ms)
Execution time for Alpha1: 9651.903372 (ms)
Execution time for Alpha2: 13094.675717 (ms)
Execution time for Alpha3: 39502.060067 (ms)
Execution time for Alpha4: 159889.165461 (ms)