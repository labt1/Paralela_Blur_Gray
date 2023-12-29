#include <iostream>
#include <cstdint>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./utils/stb_image.h"
#include "./utils/stb_image_write.h"

// Kernel CUDA
__global__ void blurKernel(const uint8_t* input, uint8_t* output, int width, int height, int channels, int radio) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int pixelIndex = (row * width + col) * channels;

        // Filtro de desenfoque
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;
            int count = 0;

            for (int i = -radio; i <= radio; ++i) {
                for (int j = -radio; j <= radio; ++j) {
                    int neighborRow = row + i;
                    int neighborCol = col + j;

                    if (neighborRow >= 0 && neighborRow < height && neighborCol >= 0 && neighborCol < width) {
                        int neighborIndex = (neighborRow * width + neighborCol) * channels + c;
                        sum += input[neighborIndex];
                        count++;
                    }
                }
            }

            output[pixelIndex + c] = static_cast<uint8_t>(sum / count);
        }
    }
}

int main() {
    const char* inputImagePath = "../cat.jpg";
    const char* outputImagePath = "../output_rgb.jpg";

    int width, height, channels;
    uint8_t* inputImage = stbi_load(inputImagePath, &width, &height, &channels, 0);

    if (!inputImage) {
        std::cerr << "Error al cargar la imagen de entrada." << std::endl;
        return 1;
    }

    std::cout<<"Canales: "<<channels<<std::endl;

    size_t imageSize = width * height * channels * sizeof(uint8_t);

    // Reserva de memoria en el dispositivo CUDA
    uint8_t* d_inputImage, *d_outputImage;
    cudaMalloc((void**)&d_inputImage, imageSize);
    cudaMalloc((void**)&d_outputImage, imageSize);

    // Copia de la imagen de entrada a la memoria del dispositivo
    cudaMemcpy(d_inputImage, inputImage, imageSize, cudaMemcpyHostToDevice);

    // Configuración de la cuadrícula y los bloques de hilos
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Llamada al kernel de CUDA para aplicar el filtro de desenfoque
    blurKernel<<<gridDim, blockDim>>>(d_inputImage, d_outputImage, width, height, channels, 50);

    // Copia del resultado de vuelta a la memoria del host
    uint8_t* outputImage = new uint8_t[imageSize];
    cudaMemcpy(outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);

    // Guardar la imagen de salida
    stbi_write_jpg(outputImagePath, width, height, channels, outputImage, 100);

    // Liberar memoria
    stbi_image_free(inputImage);
    delete[] outputImage;
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    return 0;
}
