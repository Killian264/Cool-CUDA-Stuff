
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <time.h>


#define SIZE 4

#define debug false
#define print_tables true


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

bool ChooseDevice(cudaDeviceProp* chosen_out, int* chosen_number_out) {
	cudaDeviceProp chosen_property;
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	int chosen_device_number = 0;

	if (debug) {
		printf("\nUSER DEVICES:\n");
	}

	// set initial cuda device
	cudaError_t cudaStatus = cudaGetDeviceProperties(&chosen_property, 0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return false;
	}


	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);

		if (debug) {
			printf(" Device Number: %d\n", i);
			printf("   Device name: %s\n", prop.name);
			printf("   Core Count #: %d\n", prop.multiProcessorCount);
		}

	}

	printf("\n Choosing device #%d, | %s | %d Cores\n\n", chosen_device_number, chosen_property.name, chosen_property.multiProcessorCount);

	*chosen_number_out = chosen_device_number;
	*chosen_out = chosen_property;

	return true;
}

bool HandleRunErrors() {
	// Check for any errors launching the kernel
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("C_Compute launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return false;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	cudaStatus = cudaDeviceSynchronize();
	// any errors encountered during the launch.
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d C_Compute!\n", cudaStatus);
		return false;
	}
	return true;
}

// Does CUDA work
cudaError_t doWork(int* A, int* B, int* C, int* X, int* W)
{

	cudaError_t cudaStatus = cudaSuccess;

	// CUDA arrays
	int* dev_a = nullptr;
	int* dev_b = nullptr;
	int* dev_c = nullptr;
	int* dev_x = nullptr;
	int* dev_w = nullptr;

	int single_size = SIZE * sizeof(int);
	int matrix_size = SIZE * single_size;

	int num_arrays = 5;
	int** in[] = { &dev_a, &dev_b, &dev_c, &dev_w, &dev_x };
	int** out[] = { &A, &B, &C, &W, &X };
	int size[] = { matrix_size, matrix_size, matrix_size, single_size, single_size };
	char letters[] = { 'A' , 'B', 'C', 'W', 'X' };

	/* CHOOSE DEVICE */
	cudaDeviceProp chosen_property;
	int chosen_device_number;
	bool device_found = ChooseDevice(&chosen_property, &chosen_device_number);

	if (!device_found) {
		goto Error;
	}

	cudaStatus = cudaSetDevice(chosen_device_number);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!");
		goto Error;
	}

	/* ALLOCATE DATA */
	for (int i = 0; i < num_arrays; i++) {
		cudaStatus = cudaMalloc(in[i], size[i]);
		if (cudaStatus != cudaSuccess) {
			printf("cudaMalloc %c failed!", letters[i]);
			goto Error;
		}
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

	bool ran;

	/* DO WORK */
	// Launch Kernel, with blocksize 1 and threads SIZE*SIZE
	C_Compute <<<1, chosen_property.multiProcessorCount >>> (dev_a, dev_b, dev_c);

	ran = HandleRunErrors();

	if (!ran) {
		goto Error;
	}

	W_Compute << <1, chosen_property.multiProcessorCount >> > (dev_c, dev_w);

	ran = HandleRunErrors();

	if (!ran) {
		goto Error;
	}

	X_Compute << <1, chosen_property.multiProcessorCount >> > (dev_c, dev_x);

	ran = HandleRunErrors();

	if (!ran) {
		goto Error;
	}

	/* COPY DATA BACK TO CPU */
	for (int i = 2; i < 5; i++) {
		cudaStatus = cudaMemcpy(*(out[i]), *(in[i]), size[i], cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			printf("cudaMemcpy %c failed!", letters[i]);
			goto Error;
		}
	}

// craziest thing i've ever seen
// i like it
Error:
	cudaFree(dev_a);
	cudaFree(dev_b);
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
	clock_t start_first;

	/* MALLOC */
	printf("Starting Malloc\n");
	start = clock();
	start_first = start;

	int* A = (int*)malloc(SIZE * SIZE * sizeof(int));
	int* B = (int*)malloc(SIZE * SIZE * sizeof(int));
	int* C = (int*)malloc(SIZE * SIZE * sizeof(int));

	int* X = (int*)malloc(SIZE * sizeof(int));

	int* W = (int*)malloc(SIZE * sizeof(int));

	end = clock();
	time_taken = ((double)end-start) / CLOCKS_PER_SEC;
	printf("Ending Malloc %f seconds\n\n", time_taken);


	/* FILL */
	printf("Starting A and B Fill\n");
	start = clock();

	FillArray(A, SIZE);

	FillArray(B, SIZE);

	end = clock();
	time_taken = ((double)end - start) / CLOCKS_PER_SEC;
	printf("Ending A and B Fill %f seconds\n\n", time_taken);


	/* CUDA WORK */
	printf("Starting C, W, X, work with matrix size %d, %d.\n", SIZE, SIZE);
	start = clock();

    cudaError_t cudaStatus = doWork(A, B, C, X, W);
    if (cudaStatus != cudaSuccess) {
        printf("Cuda failed!");
        return 1;
    }

	end = clock();
	time_taken = ((double)end - start) / CLOCKS_PER_SEC;
	printf("Ending C, W, X, work %f seconds\n\n", time_taken);

	/* RESULTS */
	int result = R_Compute(W, X, SIZE);


	if (print_tables) {
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
	end = clock();
	time_taken = ((double)end - start_first) / CLOCKS_PER_SEC;
	printf("Ending Program Total Time Taken: %f seconds\n\n", time_taken);

	printf("RESULT: %d", result);

	// Free Data
	free(A);
	free(B);
	free(C);
	free(X);
	free(W);


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        printf("cudaDeviceReset failed! idk what this does");
        return 1;
    }

    return 0;
}

