#include <stdbool.h>

void convolve_1d(double *input, double *output, double *kernel, int input_size, int kernel_size, bool circular) {
    int half_kernel = kernel_size / 2;

    for (int i = 0; i < input_size; ++i) {
        output[i] = 0.0;

        // Circular convolution handling
        for (int j = 0; j < kernel_size; ++j) {
            int index = i - half_kernel + j;

            if (circular) {
                index = (index + input_size) % input_size;
            } else {
                if (index < 0) {
                    index = 0;
                } else if (index >= input_size) { // Note: Changed to handle index == input_size 
                    index = input_size - 1;
                } 
            }

            output[i] += input[index] * kernel[j];
        }
    }
}

void convolve(double *A, double *kernels[3], bool isCircular[3], int *dims, int *kernel_sizes) {
    for (int dim = 0; dim < 3; ++dim) {
        convolve_1d(A + dim * dims[0] * dims[1],  // Pointer arithmetic for 3D array access
                    A + dim * dims[0] * dims[1],  // In-place modification 
                    kernels[dim], 
                    dims[0],  // Assumes dimensions are passed to C
                    kernel_sizes[dim], // Variable for kernel size per dimension
                    isCircular[dim]);
    }
}