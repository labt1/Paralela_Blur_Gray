#include <iostream>
#include <cstdint>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./utils/stb_image.h"
#include "./utils/stb_image_write.h"

// Kernel de CUDA
__global__ void rgbToGrayKernel(const uint8_t* input, uint8_t* output, int width, int height, int channels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int pixelIndex = (row * width + col) * channels;

        // Metodo de luminosidad para convertir de RGB a escala de grises: Y = 0.299*R + 0.587*G + 0.114*B
        output[row * width + col] = static_cast<uint8_t>(0.299f * input[pixelIndex] +
                                                         0.587f * input[pixelIndex + 1] +
                                                         0.114f * input[pixelIndex + 2]);
    }
}

// Función para convertir una imagen RGB a escala de grises utilizando CUDA
void rgbToGrayCUDA(const uint8_t* input, uint8_t* output, int width, int height, int channels) {
    size_t imageSize = width * height * channels * sizeof(uint8_t);
    uint8_t* d_input, *d_output;

    // Reserva de memoria en el dispositivo CUDA
    cudaMalloc((void**)&d_input, imageSize);
    cudaMalloc((void**)&d_output, width * height * sizeof(uint8_t));

    // Copia de la imagen de entrada a la memoria del dispositivo
    cudaMemcpy(d_input, input, imageSize, cudaMemcpyHostToDevice);

    // Configuración de la cuadrícula y los bloques de hilos
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Llamada al kernel de CUDA para la conversión a escala de grises
    rgbToGrayKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height, channels);

    // Copia del resultado de vuelta a la memoria del host
    cudaMemcpy(output, d_output, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Liberar memoria en el dispositivo CUDA
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    const char* inputImagePath = "../cat.jpg";
    const char* outputImagePath = "../gray_scale.jpg";

    int width, height, channels;
    uint8_t* inputImage = stbi_load(inputImagePath, &width, &height, &channels, 0);

    if (!inputImage) {
        std::cerr << "Error al cargar la imagen de entrada." << std::endl;
        return 1;
    }

    // Verifica que la imagen sea de tipo RGB
    if (channels != 3) {
        std::cerr << "La imagen no es de tipo RGB." << std::endl;
        stbi_image_free(inputImage);
        return 1;
    }

    // Calcula el tamaño de la imagen en escala de grises
    size_t grayImageSize = width * height * sizeof(uint8_t);

    // Reserva memoria para la imagen en escala de grises
    uint8_t* grayImage = new uint8_t[grayImageSize];

    // Convierte la imagen RGB a escala de grises utilizando CUDA
    rgbToGrayCUDA(inputImage, grayImage, width, height, channels);

    // Guarda la imagen en escala de grises
    stbi_write_jpg(outputImagePath, width, height, 1, grayImage, 100); //Guarda en un canal

    // Libera la memoria
    stbi_image_free(inputImage);
    delete[] grayImage;

    return 0;
}
