#include <cuda_runtime.h>
#include <iostream>
#include <ctime>
#include <cmath> 

// executing on GPU
__global__
void matmulDevice(float* A, float* B, float* C, int N)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < N && col < N) {
		int sum = 0;
		for (int i = 0; i < N; i++)
			sum += A[row * N + i] * B[i * N + col];
		C[row * N + col] = sum;
	}
}

// executing on CPU
void matmulHost(float* A, float* B, float* C, int N)
{

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			int sum = 0;
			for (int k = 0; k < N; k++)
				sum += A[i * N + k] * B[k * N + j];
			C[i * N + j] = sum;
		}
	}
}

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

using namespace std;

int main(void)
{
	
		int N = 64;
		int block_size = 16;

		// number of iterations
		int iterNumber = 100;

		unsigned int count = N*N;
		unsigned int mem_size = sizeof(float) * count;
	
	
		float* A = (float*)malloc(mem_size);
		float* B = (float*)malloc(mem_size);
		float* resaultCPU = (float*)malloc(mem_size);
		float* resaultGPU = (float*)malloc(mem_size);
	
		float* device_A, * device_B, * device_C;
	
		for (int i = 0; i < count; i++) {
			A[i] = sin(i);
			B[i] = cos(i);
		}
	
		unsigned int start_time = clock();

		for (int j = 0; j < iterNumber; j++) {
			matmulHost(A, B, resaultCPU, N);
		}

		unsigned int elapsedTime = clock() - start_time;
		float msecPerMatrixMulCpu = elapsedTime / iterNumber;

		cout << "CPU time: " << msecPerMatrixMulCpu << endl;
	
		checkCudaErrors(cudaMalloc((void**)& device_A, mem_size));
		checkCudaErrors(cudaMalloc((void**)& device_B, mem_size));
		checkCudaErrors(cudaMalloc((void**)& device_C, mem_size));
	
		// copy data to device
		checkCudaErrors(cudaMemcpy(device_A, A, mem_size,
			cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(device_B, B, mem_size,
			cudaMemcpyHostToDevice));
	
		dim3 threadsPerBlock(block_size, block_size);
		dim3 blocksPerGrid(N / block_size, N / block_size);
		
		cudaEvent_t start;
		cudaEvent_t stop;
		checkCudaErrors(cudaEventCreate(&start));
		checkCudaErrors(cudaEventCreate(&stop));

		// start of recording event
		checkCudaErrors(cudaEventRecord(start, 0));

		for (int j = 0; j < iterNumber; j++) {
			matmulDevice << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, N);
		}

		// end of recording event
		checkCudaErrors(cudaEventRecord(stop, 0));

		// waiting for end of event
		checkCudaErrors(cudaEventSynchronize(stop));

		float msecTotal = 0.0f;
		checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

		float msecPerMatrixMul = msecTotal / iterNumber;
			   
		cout << "GPU time: " << msecPerMatrixMul << endl;

		cudaDeviceSynchronize();
	
		// copy data from device
		checkCudaErrors(cudaMemcpy(resaultGPU, device_C, mem_size, cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();
	
		// free device memory
		cudaFree(device_A);
	    cudaFree(device_B);
	    cudaFree(device_C);
	
    return 0;
}

