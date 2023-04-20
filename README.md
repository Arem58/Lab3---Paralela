# Lab3---Paralela
## Cómo ejecutar los archivos de código?



## mpi_vector_operations
Para el caso de las operaciones de producto punto y la multiplicación de los vectores por un escalar, se provee un cmake que ya cuenta 
con la configuración necesaria para su compilación. Sin embargo, en caso de que esto no llegara a funcionar, el archivo se puede compilar y ejecutar
de la siguiente manera: 
```Bash
mpicc mpi_vector_operations.c -o mpi_vector_operations && mpirun -np <no_process> mpi_vector_operations
```
