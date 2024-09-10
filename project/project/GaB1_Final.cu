/* ###########################################################################################################################
## Organization         : The University of Arizona
## File name            : GaB.cu
## Language             : CUDA
## Short description    : Gallager-B Hard decision Bit-Flipping algorithm
## ######################################################################################################################## */
//// A header for each kernel describing the inputs and outputs and include a summary of the function. 
//// Comments for key stages within the kernel code.

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <string.h>

#define CUDA_CHECK(ans)                                                   \
{ gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) 
{
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

////  Define thread block dimension as a global constant for easier modification
#define MatrixFileName	"hmatR.txt"
#define Version			"1"
#define	BlockDimension	64
#define min(x,y)    	((x)<(y)?(x):(y))
#define max(x,y)    	((x)<(y)?(y):(x))

//#define VarNum			8
//#define	CheckNum		4
//#define columnDegree	2
//#define rowDegree		4
#define VarNum			1296
#define	CheckNum		648
#define columnDegree	4
#define rowDegree		8

// Implement variable-to-check message update
__global__ void initVNU(int *VtoC, int *receivedWord, int *interleaver) {
    // Determine global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = columnDegree*idx;
    // Check if thread index is within the range of VarNum
    if (idx < VarNum) {
        for (int i = 0; i < columnDegree; i++) { 
            VtoC[interleaver[offset + i]] = receivedWord[idx];
        //    printf("Thread%d: VtoC[interleaver[%d]] = VtoC[%d] = %d\n", idx, (columnDegree*idx + i), interleaver[columnDegree*idx + i], VtoC[interleaver[columnDegree*idx + i]]);
        //    printf("Thread%d: receivedWord[%d] = %d\n", idx, idx, ReceivedWord[idx]);
		}
    }
}

// Implement variable-to-check message update
__global__ void VNU(int *VtoC, int *CtoV, int *receivedWord, int *interleaver) {
    // Determine global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Check if thread index is within the range of VarNum
    if (idx < VarNum) {
        int i;
        int offset = columnDegree*idx;
        int Global = (1 - 2 * receivedWord[idx]);
		// Perform variable-to-check message update calculation
        for (i = 0; i < columnDegree; i++) 
            Global += (-2) * CtoV[interleaver[offset + i]] + 1;
        for (i = 0; i < columnDegree; i++) {
            int buf = Global - ((-2) * CtoV[interleaver[offset + i]] + 1);
			// Store the updated message in the VtoC array
            if (buf < 0)  
                VtoC[interleaver[offset + i]] = 1;
            else if (buf > 0) 
                VtoC[interleaver[offset + i]] = 0;
            else  
                VtoC[interleaver[offset + i]] = receivedWord[idx];
        }
    }
}

// Implement check-to-variable message update
__global__ void CNU(int *CtoV, int *VtoC) {
	// Determine global thread index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Check if thread index is within the range of CheckNum
    if (idx < CheckNum) {  
		int i; 
        int signe = 0;
        int offset = rowDegree*idx;
        // Calculate signe
        for (i = 0; i < rowDegree; i++) {
            signe ^= VtoC[offset + i];
        }
        // Update CtoV
        for (i = 0; i < rowDegree; i++) {
            CtoV[offset + i] = signe ^ VtoC[offset + i];
        }
    }
}

// Implement decision making based on check-to-variable messages
__global__ void Checkdecide(int *decide, int *CtoV, int *receivedWord, int *interleaver) {
	// Determine global thread index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Check if thread index is within the range of VarNum
	//	if (idx == 0)
	//		decide[0] = 0;
    if (idx < VarNum) {
        int offset = idx * columnDegree;
	//	int idxShifted = idx + 1;
        int Global = (1 - 2 * receivedWord[idx]);
        
        for (int i = 0; i < columnDegree; i++)
            Global += (-2) * CtoV[interleaver[offset + i]] + 1;
        
        if  (Global > 0) 
            decide[idx] = 0;
        else if (Global < 0) 
            decide[idx] = 1;
        else  
            decide[idx] = receivedWord[idx];
    }
}

// Change order of vector into subsequest index of columnIndex
__global__ void ChangeOrder(int* vectorOut, int* vectorIn, int* index, int num, int degree) {
	// Determine global thread index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if ((idx < num)) {
		for (int i = 0; i < degree; i++)
			vectorOut[idx] = vectorIn[index[idx*degree+i]];	
    }
}

// Implement syndrome computation using the decide matrix
__global__ void ComputeSyndrome1(int *decide, int *columnIndex, int* syndrome) {
	// Determine global thread index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < CheckNum) {
		syndrome[idx] = decide[columnIndex[idx*rowDegree]]^ 
						decide[columnIndex[idx*rowDegree+1]]^
						decide[columnIndex[idx*rowDegree+2]]^
						decide[columnIndex[idx*rowDegree+3]]^
						decide[columnIndex[idx*rowDegree+4]]^
						decide[columnIndex[idx*rowDegree+5]]^
						decide[columnIndex[idx*rowDegree+6]]^
						decide[columnIndex[idx*rowDegree+7]];
    }
}

// Implement final syndrome ORing calculated syndrome matrix
__global__ void ComputeSyndrome2_1(int* syndromeOut, int* syndromeIn) {
	// Determine global thread index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int tid  = threadIdx.x;
	
	// in each block, we have log(block dimension) steps to calculate the sum
	for (int stride = 1; stride < blockDim.x; stride *= 2) 
	{
		// just some specific threads are involved in calculation	
		if (idx % (2*stride) == 0)		
			syndromeIn[idx] |= syndromeIn[idx + stride];
	}

	// write back the last partial sum to dout added in the next phase kernel run
	if (tid == 0)
		syndromeOut[blockIdx.x] = syndromeIn[idx];
}

// Implement syndrome computation using the decide matrix
__global__ void ComputeSyndrome2(int* syndromeOut, int* syndromeIn) {
	// Determine global thread index
	int idx = blockIdx.x * 2*blockDim.x + threadIdx.x;
	int tid  = threadIdx.x;
	// shared memory for syndrome calculation of each thread
	extern __shared__ int shared_syndrome[];
	// pre-calculation to half # of threads
	shared_syndrome[tid] = syndromeIn[idx] | syndromeIn[idx+blockDim.x];
	//	shared_syndrome[tid] = decide[columnIndex[idx]];
	__syncthreads();
	
	for (int stride = blockDim.x/2; stride > 0; stride >>= 1) 
	{
		if (tid < stride)		
			shared_syndrome[tid] |= shared_syndrome[tid + stride];
		__syncthreads();
	}
	
	if (tid < 32) {
		shared_syndrome[tid] |= shared_syndrome[tid + 32];
		shared_syndrome[tid] |= shared_syndrome[tid + 16];
		shared_syndrome[tid] |= shared_syndrome[tid + 8];
		shared_syndrome[tid] |= shared_syndrome[tid + 4];
		shared_syndrome[tid] |= shared_syndrome[tid + 2];
		shared_syndrome[tid] |= shared_syndrome[tid + 1];
	}

	if (tid == 0)
		syndromeOut[blockIdx.x] = shared_syndrome[tid];
}

////  Pass block dimention as a main function argument to be able to generate 
////  Different dataset test cases
int main(int argc, char **argv) {
	//============================================================================
		// Declare Simulation input variables for GaB BF
	//============================================================================
	int iterNum[] = {116, 206, 699, 3369, 14701};
	float NbMonteCarlo = 1000000000000;	// Maximum # of codewords sent
	int NbIter = 100;					// Maximum # of iterations
//	float NbMonteCarlo = 2;			// Maximum # of codewords sent
//	int NbIter = 2;						// Maximum # of iterations
	int NBframes = 100;					// Simulation stops when NBframes in error
//	int NBframes = 200;					// Simulation stops when NBframes in error
	int NiterMoy;						// Parameters to calculate BER
	int NiterMax;
	int Dmin;
	int NbTotalErrors;
	int NbBitError;
	int NbUnDetectedErrors;
	int NbError;
	int nbtestedframes;
	int nb;
	int iter;
	int alphaNum;
	float alpha = 0.01;					// Channel probability of error
	float alpha_max = 0.0600;		    // Channel Crossover Probability Max and Min
//	float alpha_max = 0.0200;		    // Channel Crossover Probability Max and Min
	float alpha_min = 0.0200;
	float alpha_step = 0.0100;
	char FileSimu[200];
	char folder_path[200]; 
	char file_path[200];

	//============================================================================
		// declare timing variables for execution time report
	//============================================================================
	cudaEvent_t startEvent, stopEvent;
	cudaEvent_t totalStartEvent, totalStopEvent;
	float elapsedTime;
	float totalElapsedTime[5];
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	cudaEventCreate(&totalStartEvent);
	cudaEventCreate(&totalStopEvent);

	//============================================================================
		// load H matrix from the file
	//============================================================================
	// start keeping track of load time
	cudaEventRecord(startEvent, 0);
	// allocate memory for the H matrix
	bool *hostH = (bool*)malloc(CheckNum*VarNum*sizeof(bool));
    char FileName[200] = MatrixFileName;
    char FileNameRes[200];
    //strcpy(FileName, argv[1]); 	// Matrix file
	strcpy(FileName, MatrixFileName);
	printf("/////////////////////////////////////////////////////////////////\n");
	printf("Importing H matrix [%dx%d] from file '%s'\n", CheckNum, VarNum, FileName);
	FILE* hmat = fopen(FileName, "r");
	for (int i = 0; i < CheckNum; i++)
		for (int j = 0; j < VarNum; j++)
        	fscanf(hmat, "%d", &hostH[i*VarNum+j]);
    fclose(hmat);
	// // print the loaded H matrix
	// for (int i = 0; i < CheckNum; i++){
	// 	for (int j = 0; j < VarNum; j++) {
    //     	printf("%d ", hostH[i*VarNum+j]);
	// 	}
	// 	printf("\n");
	// }
	// stop keeping track of execution time
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	FILE* times = fopen("Time.txt", "w+");
	fprintf(times, "Total loading time: %f (ms)\n", elapsedTime);
	fprintf(times, "/////////////////////////////////////////////////////////////////\n");

	//============================================================================
		// convert H matrix to compressed sparse row representation
	//============================================================================
	// allocate columnIndex vector
	int *hostColumnIndex = (int*)malloc(CheckNum*rowDegree*sizeof(int));
	// allocate rowPointer Vector
	int *hostRowPointer = (int*)malloc(CheckNum*sizeof(int));
	int k = 0;
	for (int i = 0; i < CheckNum; i++) {
		for (int j = 0; j < VarNum; j++) {
			if (hostH[i*VarNum+j] == 1) {
				hostColumnIndex[k] = j;
				hostRowPointer[i] = k-rowDegree+1;
				k++;
			}
		}
	}
	// // print the obtained columnIndex vector
	// printf("columnIndex: ");
	// for (int i = 0; i < CheckNum*rowDegree; i++)
	// 	printf("%d ", hostColumnIndex[i]);
	// printf("\n");
	// // print the obtained rowPointer vector
	// printf("rowPointer: ");
	// for (int i = 0; i < CheckNum; i++)
	//	printf("i(%d): %d\n", i, hostRowPointer[i]);
	// printf("\n");

	//============================================================================
		// Restore matrix H from rowPointer and columnIndex
	//============================================================================
	bool *restoredH = (bool*)malloc(CheckNum*VarNum*sizeof(bool));
	// initialize matrix
	for (int i = 0; i < CheckNum; i++)
		for (int j = 0; j < VarNum; j++)
        	restoredH[i*VarNum+j] = 0;
	// assign each specific index to the non-zero value
	for (int i = 0; i < CheckNum; i++) {
        // Starting index of non-zero elements in row i
		int rowStart = hostRowPointer[i]; 
        for (int j = 0; j < rowDegree; j++) {
			// Column index of non-zero element
            int col = hostColumnIndex[rowStart + j]; 
			// Mark the non-zero element in matrix H
            restoredH[i*VarNum+col] = 1; 
        }
    }
	//	// print the restored H matrix
	//	printf("restored H matrix:\n");
	//	for (int i = 0; i < CheckNum; i++){
	//		for (int j = 0; j < VarNum; j++) {
	//        	printf("%d ", restoredH[i*VarNum+j]);
	//		}
	//		printf("\n");
	//	}

	//============================================================================
		// NtoB as index # of columnIndex in column-wise order
	//============================================================================
	int BranchNum = rowDegree*CheckNum;
	printf("# of graph edges: %d\n", BranchNum);
    int **NtoB = (int**)malloc(VarNum*sizeof(int*)); 
    for (int i = 0; i < VarNum; i++) 
        NtoB[i] =  (int*)malloc(columnDegree*sizeof(int));
    // initialize matrix
    for (int i = 0; i < CheckNum; i++)
        for (int j = 0; j < rowDegree; j++)
            NtoB[i][j] = 0;
    int numBranch = 0;
    int *index = (int*)malloc(VarNum*sizeof(int));
	// initialize index vector
	for (int i = 0; i < VarNum; i++)
		index[i] = 0;
	// compute NtoB
    for (int i = 0; i < CheckNum; i++) {
		int rowStart = hostRowPointer[i];
        for (int j = 0; j < rowDegree; j++) 
        { 
            int numColumn = hostColumnIndex[rowStart+j];
            if (numColumn < VarNum)
                NtoB[numColumn][index[numColumn]++] = numBranch++; 
        }
	}
	free(index);
	//    printf("NtoB = index # of Mat in column-wise order:\n");
	//    for (int i = 0; i < VarNum; i++) {
	//        for (int j = 0; j < columnDegree; j++)
	//            printf("%d\t", NtoB[i][j]);
	//        printf("\n");
	//    }
     
	//============================================================================
		// Interleaver : NtoB matrix in vector representation
	//============================================================================
    numBranch = 0;
	int *hostInterleaver = (int*)malloc(BranchNum*sizeof(int));
    for (int i = 0; i < VarNum; i++) 
        for (int j = 0; j < columnDegree; j++) 
            hostInterleaver[numBranch++]  = NtoB[i][j];
    // print the interleaver
    // printf("interleaver: ");
    // for (int i = 0; i < numBranch; i++)
    //     printf("%d ", hostInterleaver[i]);
    // printf("\n");

	printf("/////////////////////////////////////////////////////////////////\n");
	//============================================================================
		// specify grid and block dimensions for kernels 
	//============================================================================
	// parallelising Variable Nodes
	dim3 VarGrid(ceil(VarNum/((double) BlockDimension)));
	dim3 VarBlock(BlockDimension);
	// parallelising Check Nodes
	dim3 CheckGrid(ceil(CheckNum/((double) BlockDimension)));
	dim3 CheckBlock(BlockDimension);
	// computeSyndrome kernel
	int GridNum = ceil(CheckNum/((double) BlockDimension));
	int BlockNum = BlockDimension;

	printf("<<%d, %d>>\n", GridNum, BlockNum);

	//============================================================================
		// Host & device memory allocation
	//============================================================================
	// start keeping track of execution time
	cudaEventRecord(startEvent, 0);
	// allocate memory on host
	int *hostVtoC = (int*)malloc(BranchNum*sizeof(int));
	int *hostCtoV = (int*)malloc(BranchNum*sizeof(int));
	int *hostReceivedWord = (int*)malloc(VarNum*sizeof(int));
	int *hostCodeWord = (int*)malloc(VarNum*sizeof(int));
	//	int *hostDecide = (int*)malloc((VarNum+1)*sizeof(int));
	int *hostDecide = (int*)malloc(VarNum*sizeof(int));
	int *hostSyndrome = (int*)malloc(CheckNum*sizeof(int));
	int *hostSyndromeIntermediate = (int*)malloc(GridNum*sizeof(int));
	int *hostIsCodeword = (int*)malloc(sizeof(int));
	// declare device vectors
	int *deviceVtoC;
	int *deviceCtoV;
	int *deviceDecide;
	int *deviceDecideOrdered;
	int *deviceSyndrome;
	int *deviceSyndromeIntermediate;
	int *deviceNotCodeword;
	int *deviceColumnIndex;
	int *deviceReceivedWord;
	int *deviceInterleaver;
	// allocate memory on device
	CUDA_CHECK(cudaMalloc((void**) &deviceVtoC, BranchNum*sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**) &deviceCtoV, BranchNum*sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**) &deviceDecide, VarNum*sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**) &deviceDecideOrdered, CheckNum*sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**) &deviceSyndrome, CheckNum*sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**) &deviceSyndromeIntermediate, GridNum*sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**) &deviceColumnIndex, CheckNum*rowDegree*sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**) &deviceReceivedWord, VarNum*sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**) &deviceInterleaver, BranchNum*sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**) &deviceNotCodeword, sizeof(int)));
	// copy data to device
	CUDA_CHECK(cudaMemcpy(deviceInterleaver, hostInterleaver, BranchNum*sizeof(int), cudaMemcpyHostToDevice));
	// add one to columnIndex to shift decide and add one zero to index0
	//for (int i = 0; i < CheckNum; i++)
	//	hostColumnIndex[i]++;
	CUDA_CHECK(cudaMemcpy(deviceColumnIndex, hostColumnIndex, CheckNum*rowDegree*sizeof(int), cudaMemcpyHostToDevice));
	// stop keeping track of execution time
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	fprintf(times, "Total data transfer time from host to device except the receivedword: %f (ms)\n", elapsedTime);
	fprintf(times, "/////////////////////////////////////////////////////////////////\n");
	
	//strcpy(FileNameRes,FileName);
	int p = 0;
	while(FileName[p] != '.') {
		FileNameRes[p] = FileName[p];
		p++;
	}
	strcat(FileNameRes,Version);
	strcat(FileNameRes,"_Res.txt");
  	FILE* f = fopen(FileNameRes,"w+");
  	fprintf(f,"-----------------------------------------------------Gallager B-----------------------------------------------------\n");
  	fprintf(f,"alpha\t  NbEr(BER)\t\t\tNbFer(FER)\t\t Nbtested\tIterAver(Itermax)\tNbUndec(Dmin)\n");

  	printf("-----------------------------------------------------Gallager B-----------------------------------------------------\n");
  	printf("alpha\t  NbEr(BER)\t\t\tNbFer(FER)\t\t Nbtested\tIterAver(Itermax)\tNbUndec(Dmin)\n");
	//============================================================================
		// loop through different values of alpha 
	//============================================================================	
	for(alphaNum = 0, alpha = alpha_max; alpha >= alpha_min; alpha -= alpha_step, alphaNum++) {
		NiterMoy = 0;
		NiterMax = 0;
		Dmin = 1e5;
		NbTotalErrors = 0;
		NbBitError = 0;
		NbUnDetectedErrors = 0;
		NbError = 0;

		// start keeping track of load time
		cudaEventRecord(totalStartEvent, 0);
		//============================================================================
			// loop through different values of codeword 
		//============================================================================
		for (nb = 0, nbtestedframes = 0; nb < NbMonteCarlo; nb++) {
			//============================================================================
				// scan Codeword
			//============================================================================
			FILE* f1;
			// specify file name
			sprintf(FileSimu, "CodeWord_%d", nb);
			strcat(FileSimu, ".txt");
			// Specify the path to each file in the corresponding folder
			sprintf(folder_path, "/home/u4/haniehta/Documents/ece569/project/Codeword/Alpha%d/", alphaNum);
			// Concatenate the folder path and file name to form the complete file path
			snprintf(file_path, sizeof(file_path), "%s%s", folder_path, FileSimu);
			// scan Codeword of each 
			if (nb < iterNum[alphaNum]) 
				f1 = fopen(file_path,"r");
			else 
				f1 = fopen("/home/u4/haniehta/Documents/ece569/project/Receivedword/allzero.txt","r");
			if (f1 == NULL) { 
				perror("Error opening file"); 
				printf("Failed to open %s\n", file_path); 
				return 1;
			}
			else 
				for (int n = 0; n < VarNum; n++) 
					fscanf(f1, "%d", &hostCodeWord[n]); 
			fclose(f1);

			//============================================================================
				// scan Receivedword
			//============================================================================
			// specify file name
			sprintf(FileSimu, "ReceivedWord_%d", nb);
			strcat(FileSimu, ".txt");
			// Specify the path to each file in the corresponding folder
			sprintf(folder_path, "/home/u4/haniehta/Documents/ece569/project/Receivedword/Alpha%d/", alphaNum);
			// Concatenate the folder path and file name to form the complete file path
			snprintf(file_path, sizeof(file_path), "%s%s", folder_path, FileSimu);
			// scan Receivedword of each 
			if (nb < iterNum[alphaNum])
				f1 = fopen(file_path,"r");
			else 
				f1 = fopen("/home/u4/haniehta/Documents/ece569/project/Receivedword/allzero.txt","r");
			if (f1 == NULL) { 
				perror("Error opening file"); 
				printf("Failed to open %s\n", file_path); 
				return 1;
			}
			else 
				for (int n = 0; n < VarNum; n++) 
					fscanf(f1, "%d", &hostReceivedWord[n]); 
			fclose(f1);

			//============================================================================
				// initialize vectors in each codeword
			//============================================================================
			for (int k = 0; k < BranchNum; k++) 
				hostCtoV[k] = 0;
			for (int k = 0; k < VarNum; k++) 
				hostDecide[k] = hostReceivedWord[k];
		
			//============================================================================
				// copy data to device
			//============================================================================
			CUDA_CHECK(cudaMemcpy(deviceReceivedWord, hostReceivedWord, VarNum*sizeof(int), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(deviceCtoV, hostCtoV, BranchNum*sizeof(int), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(deviceDecide, hostDecide, VarNum*sizeof(int), cudaMemcpyHostToDevice));
			
			//============================================================================
				// load kernels
			//============================================================================
			for (iter = 0; iter < NbIter; iter++) {
			//	printf("Alpha (%.2f), Codeword(%d), Iter(%d)\n", alpha, nb, iter);
				fprintf(times, "Alpha (%.2f), Codeword(%d), Iter(%d)\n", alpha, nb, iter);
				fprintf(times, "/////////////////////////////////////////////////////////////////\n");
				if (iter == 0) {
					//---------------------------------------------------------------------
						// load initVNU kernel
					//---------------------------------------------------------------------
					// start keeping track of execution time
					cudaEventRecord(startEvent, 0);
					initVNU<<<VarGrid,VarBlock>>>(deviceVtoC, deviceReceivedWord, deviceInterleaver);
					cudaDeviceSynchronize();
					// stop keeping track of execution time
					cudaEventRecord(stopEvent, 0);
					cudaEventSynchronize(stopEvent);
					cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
					fprintf(times, "Total execution time to launch initVNU kernel: %f (ms)\n", elapsedTime);
					fprintf(times, "/////////////////////////////////////////////////////////////////\n");
				}
				else {
					//---------------------------------------------------------------------
						// load VNU kernel
					//---------------------------------------------------------------------
					// start keeping track of execution time
					cudaEventRecord(startEvent, 0);
					VNU<<<VarGrid,VarBlock>>>(deviceVtoC, deviceCtoV, deviceReceivedWord, deviceInterleaver);
					cudaDeviceSynchronize();
					// stop keeping track of execution time
					cudaEventRecord(stopEvent, 0);
					cudaEventSynchronize(stopEvent);
					cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
					fprintf(times, "Total execution time to launch VNU kernel: %f (ms)\n", elapsedTime);
					fprintf(times, "/////////////////////////////////////////////////////////////////\n");
				}
			//	// print the received word 
			//	printf("Receivedword from %s\n", file_path);
			//	for (int i = 0; i < VarNum; i++) 
			//		printf("%d", hostReceivedWord[i]);
			//	printf("\n");

				//---------------------------------------------------------------------
					// load CNU kernel
				//---------------------------------------------------------------------
				// start keeping track of execution time
				cudaEventRecord(startEvent, 0);
				CNU<<<CheckGrid,CheckBlock>>>(deviceCtoV, deviceVtoC);
				cudaDeviceSynchronize();
				// stop keeping track of execution time
				cudaEventRecord(stopEvent, 0);
				cudaEventSynchronize(stopEvent);
				cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
				fprintf(times, "Total execution time to launch CNU kernel: %f (ms)\n", elapsedTime);
				fprintf(times, "/////////////////////////////////////////////////////////////////\n");

				//---------------------------------------------------------------------
					// load CheckDecide kernel
				//---------------------------------------------------------------------
				// start keeping track of execution time
				cudaEventRecord(startEvent, 0);
				Checkdecide<<<VarGrid,VarBlock>>>(deviceDecide, deviceCtoV, deviceReceivedWord, deviceInterleaver);
				cudaDeviceSynchronize();
				// stop keeping track of execution time
				cudaEventRecord(stopEvent, 0);
				cudaEventSynchronize(stopEvent);
				cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
				fprintf(times, "Total execution time to launch CheckDecide kernel: %f (ms)\n", elapsedTime);
				fprintf(times, "/////////////////////////////////////////////////////////////////\n");

				//---------------------------------------------------------------------
					// load ComputeSyndrome kernel
				//---------------------------------------------------------------------
				// start keeping track of execution time
				cudaEventRecord(startEvent, 0);
			//	ChangeOrder<<<GridNum,BlockNum>>>(deviceDecideOrdered, deviceDecide, deviceColumnIndex, CheckNum*rowDegree, rowDegree);
				ComputeSyndrome1<<<GridNum,BlockNum>>>(deviceDecide, deviceColumnIndex, deviceSyndrome);
				cudaDeviceSynchronize();
				CUDA_CHECK(cudaMemcpy(hostSyndrome, deviceSyndrome, CheckNum*sizeof(int), cudaMemcpyDeviceToHost));
			//	ComputeSyndrome2_1<<<GridNum,BlockNum>>>(deviceSyndromeIntermediate, deviceSyndrome);
			//	ComputeSyndrome2_1<<<1,GridNum>>>(deviceNotCodeword, deviceSyndromeIntermediate);
				ComputeSyndrome2<<<GridNum/2,BlockNum,BlockNum*sizeof(int)>>>(deviceSyndromeIntermediate, deviceSyndrome);
				ComputeSyndrome2<<<1,GridNum,GridNum*sizeof(int)>>>(deviceNotCodeword, deviceSyndromeIntermediate);
	
				cudaDeviceSynchronize();
				// stop keeping track of execution time
				cudaEventRecord(stopEvent, 0);
				cudaEventSynchronize(stopEvent);
				cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
				fprintf(times, "Total execution time to launch ComputeSyndrome kernel: %f (ms)\n", elapsedTime);
				fprintf(times, "/////////////////////////////////////////////////////////////////\n");

				//============================================================================
					// data transfer between device to host
				//============================================================================
				// start keeping track of execution time
				cudaEventRecord(startEvent, 0);
				// copy data to host
				CUDA_CHECK(cudaMemcpy(hostVtoC, deviceVtoC, BranchNum*sizeof(int), cudaMemcpyDeviceToHost));
				CUDA_CHECK(cudaMemcpy(hostDecide, deviceDecide, VarNum*sizeof(int), cudaMemcpyDeviceToHost));
				CUDA_CHECK(cudaMemcpy(hostIsCodeword, deviceNotCodeword, sizeof(int), cudaMemcpyDeviceToHost));

				hostIsCodeword[0] = 1 - hostIsCodeword[0];
				// print calculated vectors
				//fprintf(times, "CodeWord: from %s\n", file_path);
				//for (int i = 0; i < VarNum; i++) 
				//fprintf(times, "%d", hostCodeWord[i]);
				//fprintf(times, "\n/////////////////////////////////////////////////////////////////\n");
				fprintf(times, "ReceivedWord: from %s\n", file_path);
				for (int i = 0; i < VarNum; i++) 
				fprintf(times, "%d", hostReceivedWord[i]);
				fprintf(times, "\n/////////////////////////////////////////////////////////////////\n");
				fprintf(times, "VtoC:\n");
				for (int i = 0; i < BranchNum; i++) 
				fprintf(times, "%d", hostVtoC[i]);
				fprintf(times, "\n/////////////////////////////////////////////////////////////////\n");
				fprintf(times, "Decide:\n");
				for (int i = 0; i < VarNum; i++) 
				fprintf(times, "%d", hostDecide[i]);
				fprintf(times, "\n/////////////////////////////////////////////////////////////////\n");
				fprintf(times, "Syndrome:\n");
				for (int j = 0; j < CheckNum; j++) {
				//  for (int i = 0; i < rowDegree; i++) {
				//	fprintf(times, "%d ", hostDecide[hostColumnIndex[j*rowDegree+i]]);
				//  }
				//  fprintf(times, "%d ", hostColumnIndex[j*rowDegree]);
				//  fprintf(times, " = ");
					fprintf(times, "%d", hostSyndrome[j]);
				}
				fprintf(times, "\n/////////////////////////////////////////////////////////////////\n");
				//fprintf(times, "SyndromeIntermediate:\n");
				//for (int i = 0; i < GridNum; i++) 
				//fprintf(times, "%d", hostSyndromeIntermediate[i]);
				//fprintf(times, "\n/////////////////////////////////////////////////////////////////\n");
				fprintf(times, "IsCodeWord:\n");
				for (int i = 0; i < 1; i++) 
				fprintf(times, "%d", hostIsCodeword[i]);
				fprintf(times, "\n/////////////////////////////////////////////////////////////////\n");
				// stop keeping track of execution time
				cudaEventRecord(stopEvent, 0);
				cudaEventSynchronize(stopEvent);
				cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
				fprintf(times, "Total data transfer time from device to host except some: %f (ms)\n", elapsedTime);
				// terminate the loop if the error has been corrected
				if (hostIsCodeword[0]) 
					break;
			}

			//============================================================================
				// Compute Statistics
			//============================================================================
			nbtestedframes++;
			NbError = 0;
			for (int k = 0; k < VarNum; k++)  
				if (hostDecide[k] != hostCodeWord[k]) 
					NbError++;
			NbBitError = NbBitError + NbError;
			// Case Divergence
			if (!hostIsCodeword[0])
			{
				NiterMoy = NiterMoy+NbIter;
				NbTotalErrors++;
			}
			// Case Convergence to Right Codeword
			if ((hostIsCodeword[0]) && (NbError == 0)) { 
				NiterMax = max(NiterMax, iter+1); 
				NiterMoy = NiterMoy+(iter+1); 
			}
			// Case Convergence to Wrong Codeword
			if ((hostIsCodeword[0]) && (NbError != 0))
			{
				NiterMax = max(NiterMax, iter+1); 
				NiterMoy = NiterMoy + (iter+1);
				NbTotalErrors++; 
				NbUnDetectedErrors++;
				Dmin = min(Dmin, NbError);
			}
			// Stopping Criterion
			if (NbTotalErrors == NBframes) 
				break;
		}

		printf("%.5f\t  ",alpha);
		printf("%5d(%1.16f)\t",NbBitError,(float)NbBitError/VarNum/nbtestedframes);
		printf("%3d(%1.16f)\t",NbTotalErrors,(float)NbTotalErrors/nbtestedframes);
		printf("%7d\t\t",nbtestedframes);
		printf("%1.2f(%d)\t\t",(float)NiterMoy/nbtestedframes,NiterMax);
		printf("%d(%d)\n",NbUnDetectedErrors,Dmin);

		fprintf(f,"%.5f\t",alpha);
		fprintf(f,"%5d(%1.8f)\t",NbBitError,(float)NbBitError/VarNum/nbtestedframes);
		fprintf(f,"%3d(%1.8f)\t",NbTotalErrors,(float)NbTotalErrors/nbtestedframes);
		fprintf(f,"%7d\t\t",nbtestedframes);
		fprintf(f,"%1.2f(%d)\t\t",(float)NiterMoy/nbtestedframes,NiterMax);
		fprintf(f,"%d(%d)\n",NbUnDetectedErrors,Dmin);

		// start keeping track of load time
		cudaEventRecord(totalStopEvent, 0);
		cudaEventSynchronize(totalStopEvent);
		cudaEventElapsedTime(&totalElapsedTime[alphaNum], totalStartEvent, totalStopEvent);
	}

	// Print the elapsed time
	fprintf(f,"\n");
	for (p=0;p<5;p++)
		fprintf(f,"Execution time for Alpha%d: %f (ms)\n", p, totalElapsedTime[p]);
	fclose(f);
	//fclose(times);

	//============================================================================
		// destroy timing events
	//============================================================================
	//cudaEventDestroy(startEvent);
	//cudaEventDestroy(stopEvent);

	//============================================================================
		// free host memory
	//============================================================================
	free(hostH);
	free(restoredH);
	free(hostColumnIndex);
	//free(hostRowPointer);
	free(hostInterleaver);
	free(hostVtoC);
	free(hostCtoV);
	free(hostReceivedWord);
	free(hostCodeWord);
	free(hostDecide);
	free(hostSyndrome);
	free(hostSyndromeIntermediate);
	free(hostIsCodeword);
	// Free memory for each row of NtoB
	//for (int i = 0; i < VarNum; i++)
	//	free(NtoB[i]);
	// Free memory for the array of pointers
	//free(NtoB);

	//============================================================================
		// free device memory
	//============================================================================
	//CUDA_CHECK(cudaFree(deviceVtoC));
	CUDA_CHECK(cudaFree(deviceCtoV));
	CUDA_CHECK(cudaFree(deviceReceivedWord));
	CUDA_CHECK(cudaFree(deviceInterleaver));
	CUDA_CHECK(cudaFree(deviceDecide));
	CUDA_CHECK(cudaFree(deviceNotCodeword));
	CUDA_CHECK(cudaFree(deviceColumnIndex));
	CUDA_CHECK(cudaFree(deviceSyndrome));
	CUDA_CHECK(cudaFree(deviceSyndromeIntermediate));
	CUDA_CHECK(cudaGetLastError());
	return 0;
}
