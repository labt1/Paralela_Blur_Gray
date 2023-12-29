cmake_minimum_required( VERSION 3.10.0 )

project( CUDA_BLUR VERSION 1.0.0 )

# 1. 
set(CMAKE_INSTALL_PREFIX ${EXECUTABLE_OUTPUT_PATH}/install/) # 
# set(CMAKE_BUILD_TYPE "Release") 
set(CMAKE_BUILD_TYPE "Debug") #

#2
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_89,code=compute_89")

#3
find_package( CUDA REQUIRED )
find_package( OpenCV REQUIRED )

#4

#5
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/utils/
)

#6
link_directories(
    # ${CUDA_LIBRARIES}
    # ${CUDA_cublas_LIBRARY}
    # ${CUDA_cudart_static_LIBRARY}
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

#7

#8
cuda_add_executable( ${PROJECT_NAME} blur.cu )

target_link_libraries(${PROJECT_NAME}
    cuda
    ${OpenCV_LIBS}
    # nvonnxparser    
)