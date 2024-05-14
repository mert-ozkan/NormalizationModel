#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

double complex* convolve3D(double* A, double*** kernels, int* is_circular, int* shape, int num_kernels);

double complex* convolve3D(double* A, double*** kernels, int* is_circular, int* shape, int num_kernels) {
    double complex* result = malloc(sizeof(double complex) * shape[0] * shape[1] * shape[2]);
    double complex* temp = malloc(sizeof(double complex) * shape[0] * shape[1] * shape[2]);

    for (int i = 0; i < shape[0] * shape[1] * shape[2]; i++) {
        result[i] = A[i];
        temp[i] = A[i];
    }

    for (int i = 0; i < num_kernels; i++) {
        double** kernel = kernels[i];
        int kernel_len = shape[2];

        if (!is_circular[i]) {
            for (int d = 0; d < shape[0]; d++) {
                for (int h = 0; h < shape[1]; h++) {
                    for (int w = 0; w < shape[2]; w++) {
                        double complex sum = 0;
                        int start = fmax(0, w - kernel_len / 2);
                        int end = fmin(shape[2], w - kernel_len / 2 + kernel_len);
                        for (int l = start; l < end; l++) {
                            sum += temp[(d * shape[1] + h) * shape[2] + l] * kernel[h][l - (w - kernel_len / 2)];
                        }
                        result[(d * shape[1] + h) * shape[2] + w] = sum;
                    }
                }
            }
        } else {
            for (int d = 0; d < shape[0]; d++) {
                for (int h = 0; h < shape[1]; h++) {
                    for (int w = 0; w < shape[2]; w++) {
                        double complex sum = 0;
                        for (int l = 0; l < shape[2]; l++) {
                            sum += cexp(2 * M_PI * I * w * l / shape[2]) * temp[(d * shape[1] + h) * shape[2] + l] * kernel[h][l];
                        }
                        result[(d * shape[1] + h) * shape[2] + w] = creal(sum);
                    }
                }
            }
        }

        for (int j = 0; j < shape[0] * shape[1] * shape[2]; j++) {
            temp[j] = result[j];
        }
    }

    free(temp);
    return result;
}
