
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <time.h>


#define SIZE 10000
#define THREADS 1024

#define debug false


__global__ void C_Compute(int* A, int* B, int* C) {
	int ind = threadIdx.x;

	int x = ind / SIZE;
	int y = ind % SIZE;


	int ret = 0;
	for (int i = 0; i < SIZE; i++) {

		int a_val = A[x * SIZE + i];
		int b_val = B[i * SIZE + y];

		if (b_val == a_val) {

		}
		else if (b_val < a_val) {
			ret += 1;
		}
		else {
			ret += -1;
		}
	}

	C[ind] = ret;
}

__global__ void X_Compute(int* C, int* X) {
	int i = threadIdx.x;

	int ret = 0;
	for (int j = 0; j < SIZE; j++) {

		if (C[j * SIZE + i] == 0) {
			continue;
		}

		if (C[j * SIZE + i] > 0) {
			ret++;
		}
		else {
			ret--;
		}
	}
	X[i] = ret;
}

__global__ void W_Compute(int* C, int* W) {
	int i = threadIdx.x;

	int ret = 0;
	for (int j = 0; j < SIZE; j++) {

		if (C[i * SIZE + j] == 0) {
			continue;
		}

		if (C[i * SIZE + j] > 0) {
			ret++;
		}
		else {
			ret--;
		}
	}
	W[i] = ret;
}

int R_Compute(int* W, int* X, int size) {
	int res = 0;
	for (int i = 0; i < size; i++) {
		if ((i % 2) == 0) {
			res += W[i] - X[i];
		}
		else {
			res += W[i] + X[i];
		}
	}
	return res;
}

void FillArray(int* arr, int size) {
	for (int i = 0; i < size * size; i++) {
		arr[i] = rand() % 10;
	}
}

void PrintArraySingle(int* arr, int size) {
	printf("\n");
	for (int i = 0; i < size; i++) {
		printf("%3d", arr[i]);
	}
	printf("\n");
}


void PrintArray(int* arr, int size) {
	for (int i = 0; i < size * size; i++) {
		if (i % size == 0) {
			printf("\n");
		}
		printf("%3d", arr[i]);
	}
	printf("\n");
}

// Does CUDA work
cudaError_t doWork(int* A, int* B, int* C, int* X, int* W)
{

	cudaError_t cudaStatus = cudaSuccess;

	// CUDA arrays
	int* dev_a;
	int* dev_b;
	int* dev_c;
	int* dev_x;
	int* dev_w;

	// If you have more than one gpu you're a nerd
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	/* ALLOCATE DATA */
	// allocate gpu buffer
	cudaStatus = cudaMalloc(&dev_c, SIZE * SIZE * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc A failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&dev_a, SIZE * SIZE * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc B failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&dev_b, SIZE * SIZE * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc C failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&dev_x, SIZE * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc X failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&dev_w,  SIZE * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc W failed!");
		goto Error;
	}

	/* DATA COPY */
	// Copy CPU data to gpu buffer
	cudaStatus = cudaMemcpy(dev_a, A, SIZE * SIZE * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, B, SIZE * SIZE * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	/* DO WORK */
	// Launch Kernel, with blocksize 1 and threads SIZE*SIZE
	C_Compute <<<1, THREADS >>> (dev_a, dev_b, dev_c);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "C_Compute launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	cudaStatus = cudaDeviceSynchronize();
	// any errors encountered during the launch.
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d C_Compute!\n", cudaStatus);
		goto Error;
	}

	W_Compute << <1, THREADS >> > (dev_c, dev_w);

	// launch errors
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "W_Compute launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// sync errors
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d W_Compute!\n", cudaStatus);
		goto Error;
	}

	X_Compute << <1, THREADS >> > (dev_c, dev_x);

	// launch errors
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "X_Compute launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// sync errors
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d X_Compute!\n", cudaStatus);
		goto Error;
	}

	/* COPY OUT DATA */
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(C, dev_c, SIZE * SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy C failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(W, dev_w, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy W failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(X, dev_x, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy X failed!");
		goto Error;
	}

// craziest thing i've ever seen
// i like it
Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}

int main()
{
	srand(0);

	clock_t start, end;
	double time_taken;

	printf("Starting Malloc\n");
	start = clock();

	int* A = (int*)malloc(SIZE * SIZE * sizeof(int));
	int* B = (int*)malloc(SIZE * SIZE * sizeof(int));
	int* C = (int*)malloc(SIZE * SIZE * sizeof(int));

	int* X = (int*)malloc(SIZE * sizeof(int));

	int* W = (int*)malloc(SIZE * sizeof(int));

	end = clock();
	time_taken = ((double)end-start) / CLOCKS_PER_SEC;
	printf("Ending Malloc %f seconds\n\n", time_taken);

	printf("Starting A and B Fill\n");
	start = clock();

	FillArray(A, SIZE);

	FillArray(B, SIZE);

	end = clock();
	time_taken = ((double)end - start) / CLOCKS_PER_SEC;
	printf("Ending A and B Fill %f seconds\n\n", time_taken);

	printf("Starting C, W, X, work with %d threads and matrix size %d, %d.\n", THREADS, SIZE, SIZE);
	start = clock();

	//// Do WORK FAST VERY FAST
    cudaError_t cudaStatus = doWork(A, B, C, X, W);
    if (cudaStatus != cudaSuccess) {
        printf("Cuda failed!");
        return 1;
    }

	end = clock();
	time_taken = ((double)end - start) / CLOCKS_PER_SEC;
	printf("Starting C, W, X, work %f seconds\n\n", time_taken);

	int result = R_Compute(W, X, SIZE);


	if (debug) {
		printf("A:");
		PrintArray(A, SIZE);

		printf("B:");
		PrintArray(B, SIZE);

		printf("C:");
		PrintArray(C, SIZE);

		printf("W:");
		PrintArraySingle(W, SIZE);

		printf("X:");
		PrintArraySingle(X, SIZE);
	}

	printf("RESULT: %d", result);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        printf("cudaDeviceReset failed! idk what this does");
        return 1;
    }

    return 0;
}

