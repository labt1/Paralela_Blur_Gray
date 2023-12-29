cmake_minimum_required( VERSION 3.10.0 )

project( RGB_TO_GRAY VERSION 1.0.0 )

set(CMAKE_BUILD_TYPE "Debug")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-gencode arch=compute_89,code=compute_89")

find_package( CUDA REQUIRED )
find_package( OpenCV REQUIRED )

#set ( CMAKE_CXX_FLAGS "-no-pie" )

include_directories( 
    ${CUDA_INCLUDE_DIRS}
)

link_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
)

cuda_add_executable( ${PROJECT_NAME} rgb_gris.cu )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES})