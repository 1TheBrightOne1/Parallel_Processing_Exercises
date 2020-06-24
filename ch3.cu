#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <iostream>

__global__ void addVec(int* a, int* b, int* c, int size)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    c[index] = a[index] + b[index];
}

//3.1 a
__global__
void MatrixAdditionElement(int* left, int* right, int* result, size_t width)
{
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    result[index] = left[index] + right[index];
}

//3.1 b 
__global__
void MatrixAdditionRow(int* left, int* right, int* result, size_t width)
{
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < width)
    {
        index *= width;

        for (size_t i = 0; i < width; ++i)
        {
            result[index + i] = left[index + i] + right[index + i];
        }
    }
}

//3.1 c
__global__
void MatrixAdditionCol(int* left, int* right, int* result, size_t width)
{
    size_t elements = width * width;
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < width)
    {
        for (size_t i = index; i < elements; i += blockDim.x)
        {
            result[i] = left[i] + right[i];
        }
    }
}

void Driver3_1(size_t width)
{
    int* A, *B, *C1, *C2, *C3;
    size_t elements = width * width;

    cudaMallocManaged(&A, elements * sizeof(int));
    cudaMallocManaged(&B, elements * sizeof(int));
    cudaMallocManaged(&C1, elements * sizeof(int));
    cudaMallocManaged(&C2, elements * sizeof(int));
    cudaMallocManaged(&C3, elements * sizeof(int));

    for (size_t i = 0; i < elements; ++i)
    {
        A[i] = i;
        B[i] = elements - i;
    }

    MatrixAdditionElement<<<elements / 256 + 1, 256>>>(A, B, C1, width);
    MatrixAdditionRow<<<width / 256 + 1, 256>>>(A, B, C2, width);
    MatrixAdditionCol<<<width / 256 + 1, 256>>>(A, B, C3, width);

    cudaDeviceSynchronize();

    for (size_t i = 0; i < elements; ++i)
    {
        if (C1[i] != C2[i] || C2[i] != C3[i])
        {
            std::cout << "Mismatch at " << i << '\n';
            break;
        }
    }

    std::cout << "Finished\n";

    cudaFree(A);
    cudaFree(B);
    cudaFree(C1);
    cudaFree(C2);
    cudaFree(C3);
}

int main()
{
    std::cout << "Starting 3_1\nEnter width: ";
    size_t width;
    std::cin >> width;
    Driver3_1(width);

    return 0;
}