# Lab3---Paralela
## Cómo ejecutar los archivos de código?
Para poder ejecutar el código provisto dentro de este repositorio es necesario que se tenga instalado y configurado MPI en su máquina local. 

## Para ejecutar y compilar el primer programa secuencial vector_add_num_aleatorios.c

* Compile:  mpicc -g -Wall -o vector_add vector_add_num_aleatorios.c
* Run:      mpiexec -n 1 ./vector_add

## Para poder ejecutar y compilar el programa mpi_vector_add_num_aleatorios.c 

* Compile:  mpicc -g -Wall -o mpi_vector_add vector_add_num_aleatorios.c
* Run:      mpiexec -n <comm_sz> ./mpi_vector_add

## mpi_vector_operations
Para el caso de las operaciones de producto punto y la multiplicación de los vectores por un escalar, se provee un cmake que ya cuenta 
con la configuración necesaria para su compilación. Sin embargo, en caso de que esto no llegara a funcionar, el archivo se puede compilar y ejecutar
de la siguiente manera: 
```Bash
mpicc mpi_vector_operations.c -o mpi_vector_operations && mpirun -np <no_process> mpi_vector_operations
```
