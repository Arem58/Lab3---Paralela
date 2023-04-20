cmake_minimum_required(VERSION 3.18)
project(lab03)
enable_language(Fortran)

set(CMAKE_CXX_STANDARD 20)

find_package(MPI REQUIRED)
add_definitions(-DOMPI_SKIP_MPICXX)

link_libraries(MPI::MPI_CXX)

add_executable(mpi_vector_operations mpi_vector_operations.c)
#add_executable(worker2 worker2.c)

#add_custom_target(Ejercicio1 DEPENDS master2 worker2)
#
#add_custom_command(
#        TARGET Ejercicio1
#        POST_BUILD
#        COMMAND ${CMAKE_COMMAND}
#        ARGS -E copy_if_different ${CMAKE_SOURCE_DIR}/appFile ${CMAKE_BINARY_DIR}/appFile
#)
#
#add_executable(Ejercicio2 mpi_trapIO.c)
