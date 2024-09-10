#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define VarNum			1296
#define rowDegree		8
#define	CheckNum		648
#define columnDegree	4

__global__ void ComputeSyndrome1(int *decide, int *columnIndex, int* syndrome) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx < CheckNum) {
		syndrome[idx] = idx;
		printf("T%d: (%d)\n", idx, syndrome[idx]);
    }
}

__global__ void initVNU(int *VtoC, int *receivedWord, int* index) {
    // Determine global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = columnDegree*idx;
    // Check if thread index is within the range of VarNum
	if (idx < VarNum) {
        for (int i = 0; i < columnDegree; i++) { 
            int writeIndex = index[offset + i];
			VtoC[writeIndex] = receivedWord[idx];
		}
    }
}

__global__ void initVNUChanged(int *VtoC, int *receivedWord) {
    // Determine global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Check if thread index is within the range of VarNum
    if (idx < VarNum) {
        //for (int i = 0; i < columnDegree; i++) { 
		//	VtoC[idx+VarNum*i] = receivedWord[idx];
			VtoC[idx+VarNum*0] = receivedWord[idx];
			VtoC[idx+VarNum*1] = receivedWord[idx];
			VtoC[idx+VarNum*2] = receivedWord[idx];
			VtoC[idx+VarNum*3] = receivedWord[idx];
			VtoC[idx+VarNum*4] = receivedWord[idx];
			VtoC[idx+VarNum*5] = receivedWord[idx];
			VtoC[idx+VarNum*6] = receivedWord[idx];
			VtoC[idx+VarNum*7] = receivedWord[idx];
		//	VtoC[idx] = receivedWord[idx];
		//	VtoC[idx*(columnDegree)+i] = receivedWord[idx];
		//	if (idx == 1293)
		//		printf("Thread%d writes %d to index (%d)\n", idx, VtoC[idx+(VarNum/columnDegree)*i], idx+(VarNum/columnDegree)*i);
		//}
    }
}

__global__ void VNUChanged(int *VtoC, int *CtoV, int *receivedWord) {
    // Determine global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Check if thread index is within the range of VarNum
    if (idx < VarNum) {
        int i;
        int Global = (1 - 2 * receivedWord[idx]);
		// Perform variable-to-check message update calculation
        for (i = 0; i < columnDegree; i++) {
            Global += (-2) * CtoV[idx+VarNum*i] + 1;
        }
        for (i = 0; i < columnDegree; i++) {
            int buf = Global - ((-2) * CtoV[idx+VarNum*i] + 1);
			// Store the updated message in the VtoC array
            if (buf < 0)  
                VtoC[idx+VarNum*i] = 1;
            else if (buf > 0) 
                VtoC[idx+VarNum*i] = 0;
            else  
                VtoC[idx+VarNum*i] = receivedWord[idx];
        }
    }
}

__global__ void VNU(int *VtoC, int *CtoV, int *receivedWord, int* interleaver) {
    // Determine global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Check if thread index is within the range of VarNum
    if (idx < VarNum) {
        int i;
		int offset = columnDegree*idx;
        int Global = (1 - 2 * receivedWord[idx]);
        int writeIndex[columnDegree];
		// Perform variable-to-check message update calculation
        for (i = 0; i < columnDegree; i++) {
            writeIndex[i] = interleaver[offset + i];
            Global += (-2) * CtoV[writeIndex[i]] + 1;
        }
        for (i = 0; i < columnDegree; i++) {
            int buf = Global - ((-2) * CtoV[writeIndex[i]] + 1);
			// Store the updated message in the VtoC array
            if (buf < 0)  
                VtoC[writeIndex[i]] = 1;
            else if (buf > 0) 
                VtoC[writeIndex[i]] = 0;
            else  
                VtoC[writeIndex[i]] = receivedWord[idx];
        }
    }
}

__global__ void ChangeOrderC(int* vectorOut, int* vectorIn, int* index, int num, int degree) {
	// Determine global thread index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < num) {
		for (int i = 0; i < degree; i++)
			vectorOut[idx+num*i] = vectorIn[index[idx*degree+i]];	
    }
}

__global__ void RestoreOrder(int* vectorOut, int* vectorIn, int* index, int num, int degree) {
	// Determine global thread index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < num) {
		for (int i = 0; i < degree; i++)
			vectorOut[index[idx*degree+i]] = vectorIn[idx+num*i];
    }
}

int main(int argc, char **argv) {
	cudaEvent_t startEvent, stopEvent;
	float elapsedTime;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	int *hostColumnIndex = (int*)malloc(CheckNum*rowDegree*sizeof(int));
	FILE* f = fopen("columnIndexR.txt", "r");
	for (int i = 0; i < CheckNum; i++) {
		for (int j = 0; j < rowDegree; j++) {
			fscanf(f, "%d", &hostColumnIndex[i*rowDegree+j]);
		}
	}
	fclose(f);

	int **NtoB = (int**)malloc(VarNum*sizeof(int*)); 
    for (int i = 0; i < VarNum; i++) 
        NtoB[i] =  (int*)malloc(columnDegree*sizeof(int));
    for (int i = 0; i < CheckNum; i++)
        for (int j = 0; j < rowDegree; j++)
            NtoB[i][j] = 0;
    int numBranch = 0;
    int *index = (int*)malloc(VarNum*sizeof(int));
	for (int i = 0; i < VarNum; i++)
		index[i] = 0;
    for (int i = 0; i < CheckNum; i++) {
	//	int rowStart = hostRowPointer[i];
		int rowStart = i*rowDegree;
        for (int j = 0; j < rowDegree; j++) 
        { 
            int numColumn = hostColumnIndex[rowStart+j];
            if (numColumn < VarNum)
                NtoB[numColumn][index[numColumn]++] = numBranch++; 
        }
	}
	free(index);
    numBranch = 0;
	int *hostInterleaver = (int*)malloc(CheckNum*rowDegree*sizeof(int));
    for (int i = 0; i < VarNum; i++) 
        for (int j = 0; j < columnDegree; j++) 
            hostInterleaver[numBranch++]  = NtoB[i][j];

	int *hostdataIn = (int*)malloc(VarNum*sizeof(int));
	int *hostdataOut = (int*)malloc(VarNum*sizeof(int));
	int *hostVtoCOut = (int*)malloc(CheckNum*rowDegree*sizeof(int));
	int *hostVtoCIn = (int*)malloc(CheckNum*rowDegree*sizeof(int));
	int *hostVtoC = (int*)malloc(CheckNum*rowDegree*sizeof(int));
	for (int j = 0; j < VarNum; j++) 
            hostdataIn[j]  = j;

	int *deviceInterleaver;
	int *deviceVtoC;
	int *deviceVtoCIn;
	int *devicedataIn;
	int *deviceVtoCOut;
	int *devicedataOut;
	CUDA_CHECK(cudaMalloc((void**) &deviceInterleaver, numBranch*sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**) &deviceVtoC, numBranch*sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**) &deviceVtoCIn, numBranch*sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**) &deviceVtoCOut, numBranch*sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**) &devicedataIn, VarNum*sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**) &devicedataOut, VarNum*sizeof(int)));

	CUDA_CHECK(cudaMemcpy(deviceInterleaver, hostInterleaver, CheckNum*rowDegree*sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(devicedataIn, hostdataIn, VarNum*sizeof(int), cudaMemcpyHostToDevice));

	printf("run the code\n");
	int GridNum = ceil(VarNum/((double) 64));
	printf("<<%d, %d>>\n", GridNum, 64);

	cudaEventRecord(startEvent, 0);
	//initVNUChanged<<<GridNum,64>>>(deviceVtoCIn, devicedataIn);
	VNUChanged<<<GridNum,64>>>(deviceVtoCIn, devicedataIn, devicedataIn);
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	printf("Total loading time: %f (ms)\n", elapsedTime);

	cudaEventRecord(startEvent, 0);
	RestoreOrder<<<GridNum,64>>>(deviceVtoCOut, deviceVtoCIn, deviceInterleaver, VarNum, columnDegree);
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	printf("Total loading time: %f (ms)\n", elapsedTime);

	cudaEventRecord(startEvent, 0);
	//initVNU<<<GridNum,64>>>(deviceVtoC, devicedataIn, deviceInterleaver);
	VNU<<<GridNum,64>>>(deviceVtoC, devicedataIn, devicedataIn, deviceInterleaver);
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
	printf("Total loading time: %f (ms)\n", elapsedTime);
	CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
	
	CUDA_CHECK(cudaMemcpy(hostVtoCIn, deviceVtoCIn, CheckNum*rowDegree*sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(hostVtoCOut, deviceVtoCOut, CheckNum*rowDegree*sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(hostVtoC, deviceVtoC, CheckNum*rowDegree*sizeof(int), cudaMemcpyDeviceToHost));
	
	FILE* Rec = fopen("OriginalReceivedWord.txt", "w+");
	fprintf(Rec, "ReceivedWord:\n");
    for (int i = 0; i < VarNum; i++) {
		if (i%4 == 3)
        	fprintf(Rec, "%d\n", hostdataIn[i]);
		else
			fprintf(Rec, "%d ", hostdataIn[i]);
	}
	FILE* Org = fopen("OriginalVtoC.txt", "w+");
	fprintf(Org, "Original VtoC:\n");
    for (int i = 0; i < VarNum; i++) {
		for (int j = 0; j < columnDegree; j++)
        	fprintf(Org, "%d ", hostVtoC[i*columnDegree+j]);
		fprintf(Org, "\n");
	}
    fprintf(Org, "\n");
	FILE* first = fopen("FirstVtoC.txt", "w+");
	fprintf(first, "VtoC with change ordering:\n");
    for (int i = 0; i < VarNum*columnDegree; i++) {
		fprintf(first, "%d\n", hostVtoCIn[i]);
	}
    fprintf(first, "\n");
	FILE* Rst = fopen("RestoredVtoC.txt", "w+");
    fprintf(Rst, "VtoC after restoring:\n");
    for (int i = 0; i < VarNum; i++) {
		for (int j = 0; j < columnDegree; j++)
        	fprintf(Rst, "%d ", hostVtoCOut[i*columnDegree+j]);
		fprintf(Rst, "\n");
	}
    fprintf(Rst, "\n");
    fprintf(Rst, "No match at (row, column)\n");
	int notMatch = 0;
	for (int i = 0; i < VarNum; i++) {
		for (int j = 0; j < columnDegree; j++)
			if (hostVtoCOut[i*columnDegree+j] != hostVtoC[i*columnDegree+j]) {
        		fprintf(Rst, "(%d, %d), ", i, j);
				notMatch++;
			}
		fprintf(Rst, "\n");
	}
	fprintf(Rst, "Total %d missmatches\n", notMatch);

	free(hostdataIn);
	free(hostdataOut);
	//free(hostVtoCOut);
	//free(hostVtoCIn);
	//free(hostVtoC);
	CUDA_CHECK(cudaFree(deviceInterleaver));
	CUDA_CHECK(cudaFree(deviceVtoC));
	CUDA_CHECK(cudaFree(deviceVtoCIn));
	CUDA_CHECK(cudaFree(devicedataIn));
	CUDA_CHECK(cudaFree(deviceVtoCOut));
	CUDA_CHECK(cudaFree(devicedataOut));
	/*int *hostDecide = (int*)malloc(VarNum*sizeof(int));
	for (int j = 0; j < VarNum; j++) {
		hostDecide[j] = 0;
	}
	int *hostSyndrome = (int*)malloc(CheckNum*sizeof(int));
	
	int *deviceSyndrome;
	int *deviceDecide;
    int *deviceColumnIndex;
	if (cudaMalloc((void**) &deviceDecide, VarNum*sizeof(int)) != cudaSuccess)
		printf("malloc error for deviceDecide\n");
	if (cudaMalloc((void**) &deviceSyndrome, CheckNum*sizeof(int)) != cudaSuccess)
		printf("malloc error for deviceSyndrome\n");
	if (cudaMalloc((void**) &deviceColumnIndex, CheckNum*rowDegree*sizeof(int)) != cudaSuccess)
		printf("malloc error for deviceColumnIndex\n");
	
	if (cudaMemcpy(deviceColumnIndex, hostColumnIndex, CheckNum*rowDegree*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
		printf("data transfer error from host to device on columnIndex\n");
	if (cudaMemcpy(deviceDecide, hostDecide, VarNum*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
		printf("data transfer error from host to device on decide\n");
	
	
	ComputeSyndrome1<<<1,10>>>(deviceDecide, deviceColumnIndex, deviceSyndrome);
	cudaDeviceSynchronize();
	
	if (cudaMemcpy(hostSyndrome, deviceSyndrome, CheckNum*sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
		printf("data transfer error from device to host on Syndrome\n");

	printf("\n/////////////////////////////////////////////////////////////////\n");
	printf("Syndrome:\n");
	for (int j = 0; j < CheckNum; j++) {
		printf("%d ", hostSyndrome[j]);
	}
	printf("\n/////////////////////////////////////////////////////////////////\n");

	
	free(hostColumnIndex);
	free(hostDecide);
	free(hostSyndrome);

	cudaFree(deviceDecide);
	cudaFree(deviceColumnIndex);
	cudaFree(deviceSyndrome);
	*/
	return 0;
}
