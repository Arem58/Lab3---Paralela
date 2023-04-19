/* File:     mpi_vector_add.c
 *
 * Purpose:  Implement parallel vector addition using a block
 *           distribution of the vectors.  This version also
 *           illustrates the use of MPI_Scatter and MPI_Gather.
 *
 * Compile:  mpicc -g -Wall -o mpi_vector_add mpi_vector_add.c
 * Run:      mpiexec -n <comm_sz> ./vector_add
 *
 * Input:    The order of the vectors, n, and the vectors x and y
 * Output:   The sum vector z = x+y
 *
 * Notes:
 * 1.  The order of the vectors, n, should be evenly divisible
 *     by comm_sz
 * 2.  DEBUG compile flag.
 * 3.  This program does fairly extensive error checking.  When
 *     an error is detected, a message is printed and the processes
 *     quit.  Errors detected are incorrect values of the vector
 *     order (negative or not evenly divisible by comm_sz), and
 *     malloc failures.
 *
 * IPP:  Section 3.4.6 (pp. 109 and ff.)
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void Check_for_error(int local_ok, char fname[], char message[], MPI_Comm comm);

void Read_n(int *n, int *elements_per_process_p, int rank, int comm_sz, MPI_Comm comm);

void Allocate_vectors(double **local_x_pp, double **local_y_pp, double **local_z_pp, int elements_per_process,
                      MPI_Comm comm);

void Read_vector(double local_a[], int elements_per_process, int n, char vec_name[], int rank, MPI_Comm comm);

void Print_vector(double local_b[], int elements_per_process, int n, char title[], int rank, MPI_Comm comm);

void Parallel_vector_sum(double local_x[], double local_y[], double local_z[], int elements_per_process);


/*-------------------------------------------------------------------*/
int main(void) {

    int n, elements_per_process, comm_sz, rank;
    double *local_x, *local_y, *local_z, start, end;
    MPI_Comm comm;

    MPI_Init(NULL, NULL);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &comm_sz);
    MPI_Comm_rank(comm, &rank);

    Read_n(&n, &elements_per_process, rank, comm_sz, comm);


    start = MPI_Wtime();

    Allocate_vectors(&local_x, &local_y, &local_z, elements_per_process, comm);
    Read_vector(local_x, elements_per_process, n, "x", rank, comm);
    Read_vector(local_y, elements_per_process, n, "y", rank, comm);
    Parallel_vector_sum(local_x, local_y, local_z, elements_per_process);

    end = MPI_Wtime();

    if (rank == 0)
        printf("\nTook %f ms to run\n", (end - start) * 1000);

    free(local_x);
    free(local_y);
    free(local_z);

    MPI_Finalize();

    return 0;
}  /* main */

/*-------------------------------------------------------------------
 * Function:  Check_for_error
 * Purpose:   Check whether any process has found an error.  If so, print message and terminate all processes. Otherwise,
 *            continue execution.
 * In args:   local_ok:  1 if calling process has found an error, 0 otherwise
 *            fname:     name of function calling Check_for_error
 *            message:   message to print if there's an error
 *            comm:      communicator containing processes calling
 * Check_for_error:  should be MPI_COMM_WORLD.
 * Note:
 *    The communicator containing the processes calling Check_for_error
 *    should be MPI_COMM_WORLD.
 */
void Check_for_error(int local_ok, char fname[], char message[], MPI_Comm comm) {
    int ok;

    MPI_Allreduce(&local_ok, &ok, 1, MPI_INT, MPI_MIN, comm);
    if (ok == 0) {
        int rank;
        MPI_Comm_rank(comm, &rank);
        if (rank == 0) {
            fprintf(stderr, "Proc %d > In %s, %s\n", (rank, fname,message));
            fflush(stderr);
        }
        MPI_Finalize();
        exit(-1);
    }
}  /* Check_for_error */

/*-------------------------------------------------------------------
 * Function:  Read_n
 * Purpose:   Get the order of the vectors from stdin on proc 0 and
 *            broadcast to other processes.
 * In args:   (rank:    process (rank in communicator
 *            comm_sz:    number of processes in communicator
 *            comm:       communicator containing all the processes
 *                        calling Read_n
 * Out args:  n:        global value of n
 *            elements_per_process_p:  local value of n = n/comm_sz
 *
 * Errors:    n should be positive and evenly divisible by comm_sz
 */
void Read_n(int *n, int *elements_per_process, int rank, int comm_sz, MPI_Comm comm) {
    int local_ok = 1;
    char *fname = "Read_n";

    if (rank == 0) {
        printf("What's the order of the vectors?\n");
        scanf("%d", n);
    }

// broadcast
    MPI_Bcast(n, 1, MPI_INT, 0, comm);

    if (*n <= 0 || *n % comm_sz != 0) local_ok = 0;
    Check_for_error(local_ok, fname, "n should be > 0 and evenly divisible by comm_sz", comm);
    *elements_per_process = *n / comm_sz;

}  /* Read_n */

/*-------------------------------------------------------------------
 * Function:  Allocate_vectors
 * Purpose:   Allocate storage for x, y, and z
 * In args:   elements_per_process:  the size of the local vectors
 *            comm:     the communicator containing the calling processes
 * Out args:  local_x_pp, local_y_pp, local_z_pp:  pointers to memory
 *               blocks to be allocated for local vectors
 *
 * Errors:    One or more of the calls to malloc fails
 */
void Allocate_vectors(double **local_x_pp, double **local_y_pp, double **local_z_pp, int elements_per_process, MPI_Comm comm) {
    int local_ok = 1;
    char *fname = "Allocate_vectors";

    *local_x_pp = malloc(elements_per_process * sizeof(double));
    *local_y_pp = malloc(elements_per_process * sizeof(double));
    *local_z_pp = malloc(elements_per_process * sizeof(double));

    if (*local_x_pp == NULL || *local_y_pp == NULL || *local_z_pp == NULL) local_ok = 0;
    Check_for_error(local_ok, fname, "Can't allocate local vector(s)", comm);
}  /* Allocate_vectors */

/*-------------------------------------------------------------------
 * Function:   Read_vector
 * Purpose:    Read a vector from stdin on process 0 and distribute
 *             among the processes using a block distribution.
 * In args:    elements_per_process:  size of local vectors
 *             n:        size of global vector
 *             vec_name: name of vector being read (e.g., "x")
 *             rank:  calling process' rank in comm
 *             comm:     communicator containing calling processes
 * Out arg:    local_a:  local vector read
 *
 * Errors:     if the malloc on process 0 for temporary storage
 *             fails the program terminates
 *
 * Note:
 *    This function assumes a block distribution and the order
 *   of the vector evenly divisible by comm_sz.
 */
void Read_vector(double local_a[], int elements_per_process, int n, char vec_name[], int rank, MPI_Comm comm) {
    double *a = NULL;
    int i;
    int local_ok = 1;
    char *fname = "Read_vector";

    if (rank == 0) {
        a = malloc(n * sizeof(double));

        if (a == NULL) local_ok = 0;
        Check_for_error(local_ok, fname, "Can't allocate temporary vector", comm);

        //fill vec with index
        for (i = 0; i < n; i++) {
            a[i] = rand() % 1000;;
        }

        MPI_Scatter(a, elements_per_process, MPI_DOUBLE, local_a, elements_per_process, MPI_DOUBLE, 0, comm);
        free(a);

    } else {
        Check_for_error(local_ok, fname, "Can't allocate temporary vector", comm);
        MPI_Scatter(a, elements_per_process, MPI_DOUBLE, local_a, elements_per_process, MPI_DOUBLE, 0, comm);
    }
}  /* Read_vector */

/*-------------------------------------------------------------------
 * Function:  Print_vector
 * Purpose:   Print a vector that has a block distribution to stdout
 * In args:   local_b:  local storage for vector to be printed
 *            elements_per_process:  order of local vectors
 *            n:        order of global vector (elements_per_process*comm_sz)
 *            title:    title to precede print out
 *            comm:     communicator containing processes calling
 *                      Print_vector
 *
 * Error:     if process 0 can't allocate temporary storage for
 *            the full vector, the program terminates.
 *
 * Note:
 *    Assumes order of vector is evenly divisible by the number of
 *    processes
 */
void Print_vector(double local_b[], int elements_per_process, int n, char title[], int rank, MPI_Comm comm) {
    double *b = NULL;
    int i;
    int local_ok = 1;
    char *fname = "Print_vector";

    if (rank == 0) {
        b = malloc(n * sizeof(double));

        if (b == NULL) local_ok = 0;
        Check_for_error(local_ok, fname, "Can't allocate temporary vector", comm);
        MPI_Gather(local_b, elements_per_process, MPI_DOUBLE, b, elements_per_process, MPI_DOUBLE, 0, comm);

        printf("%s\n", title);

//        for (i = 0; i < n; i++)
//            printf("%f ", b[i]);
//        printf("\n");

        free(b);
    } else {
        Check_for_error(local_ok, fname, "Can't allocate temporary vector", comm);
        MPI_Gather(local_b, elements_per_process, MPI_DOUBLE, b, elements_per_process, MPI_DOUBLE, 0, comm);
    }
}  /* Print_vector */

/*-------------------------------------------------------------------
 * Function:  Parallel_vector_sum
 * Purpose:   Add a vector that's been distributed among the processes
 * In args:   local_x:  local storage of one of the vectors being added
 *            local_y:  local storage for the second vector being added
 *            elements_per_process:  the number of components in local_x, local_y,
 *                      and local_z
 * Out arg:   local_z:  local storage for the sum of the two vectors
 */
void Parallel_vector_sum(double local_x[], double local_y[], double local_z[], int elements_per_process) {
    int x;

    for (x = 0; x < elements_per_process; x++)
        local_z[x] = local_x[x] + local_y[x];
}  /* Parallel_vector_sum */
