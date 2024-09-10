#!/bin/bash
# Specify the input file
#fileMat=hmatR.txt
module load cuda11/11.0
gcc -o gabc GaBR_Final.c -lm
nvcc -G -g -o gab1 GaB1_Final.cu
nvcc -G -g -o gab2 GaB2_Final.cu
nvcc -G -g -o gab3 GaB3_Final.cu
./gabc
./gab1
./gab2
./gab3
## Loop over each block dimension
#for blockDim in 32 64 128 256 512 1024 2048
#do
#    # Run gab3 with the current block dimension
#    ./gab3 $blockDim
#    ./gab3 $fileMat $blockDim
#done
# Loop over each number of codeword
#for blockDim in 128
#do
#	for NbMonteCarlo in 1 10 100 1000 10000
#	do
#		# Run gab3 with the current block dimension
#		./gab3 $blockDim $NbMonteCarlo
#		./gab3 $fileMat $blockDim $NbMonteCarlo
#	done
#done
# Loop over each number of codeword
#for blockDim in 64 128
#do
#	for NbMonteCarlo in 10
#	do
#		#for StreamNum in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 
#		#do
#			# Run gab3 with the current block dimension
#		#	./gab3 $fileMat $blockDim $NbMonteCarlo $StreamNum
#			./gab3 $fileMat $blockDim $NbMonteCarlo
#		#	./gab3 $blockDim $NbMonteCarlo $StreamNum
#			./gab3 $blockDim $NbMonteCarlo
#		#done
#	done
#done