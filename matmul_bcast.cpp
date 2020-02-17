#include<stdio.h> 
#include<mpi.h> 
#include<stdlib.h>
//N=2^4 x28, 2^6 x28 and 2^8 x28
#define N 448 
#define MASTER_TO_SLAVE_TAG 1 
#define SLAVE_TO_MASTER_TAG 4 
class solution{
    void makeAB(); 
    void printArray(); 
    int rank; 
    int size; 
    int i, j, k; 
    double mat_a[N][N]; //initialze matrix a
    double mat_b[N][N]; //initialze matrix b
    double mat_result[N][N]; //initialze result matrix
    double start_time; //initialzie start time of matrix multiplication
    double end_time; //initialzie end time of matrix multiplication
    int low_bound; //initialzie lower bound to send matrix a row wise
    int upper_bound; //initialzie upper bound to send matrix a row wise
    int portion; //initialzie portion to keep count of no. of rows sent to each core
    MPI_Status status; 
    MPI_Request request; 
    int main(int argc, char *argv[]) 
    { 
        MPI_Init(&argc, &argv); 
        MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
        MPI_Comm_size(MPI_COMM_WORLD, &size); 
        
        if (rank == 0) { //if master
            makeAB(); //function to create matrices a and b
            start_time = MPI_Wtime(); //set start time
            for (i = 1; i < size; i++) { 
                portion = (N / (size - 1)); //set postion size
                low_bound = (i - 1) * portion; //set lower bound
                if (((i + 1) == size) && ((N % (size - 1)) != 0)) { //for last core
                    upper_bound = N; 
                } else { 
                    upper_bound = low_bound + portion; //set upper bound for all cores except last
                } 
            MPI_Isend(&low_bound, 1, MPI_INT, i, MASTER_TO_SLAVE_TAG, MPI_COMM_WORLD, &request); //send lower bound to slaves
            MPI_Isend(&upper_bound, 1, MPI_INT, i, MASTER_TO_SLAVE_TAG + 1, MPI_COMM_WORLD, &request); //send upper bound to slaves
            MPI_Isend(&mat_a[low_bound][0], (upper_bound - low_bound) * N, MPI_DOUBLE, i, MASTER_TO_SLAVE_TAG + 2, MPI_COMM_WORLD, &request); //send matrix a row wise to slaves
            } 
        } 
        
        MPI_Bcast(&mat_b, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD); //broadcast matrix b to all slaves
        
        if (rank > 0) { //if slave
            MPI_Recv(&low_bound, 1, MPI_INT, 0, MASTER_TO_SLAVE_TAG, MPI_COMM_WORLD, &status); //recieve lower bound
            MPI_Recv(&upper_bound, 1, MPI_INT, 0, MASTER_TO_SLAVE_TAG + 1, MPI_COMM_WORLD, &status); //recieve upper bound
            MPI_Recv(&mat_a[low_bound][0], (upper_bound - low_bound) * N, MPI_DOUBLE, 0, MASTER_TO_SLAVE_TAG + 2, MPI_COMM_WORLD, &status); //recieve rows of matrix a
            for (i = low_bound; i < upper_bound; i++) {//iterate through a given set of rows of [A] 
                for (j = 0; j < N; j++) {//iterate through columns of [B] 
                    for (k = 0; k < N; k++) {//iterate through rows of [B] 
                        mat_result[i][j] += (mat_a[i][k] * mat_b[k][j]); //matrix multiplication
                    } 
                } 
            } 
        MPI_Isend(&low_bound, 1, MPI_INT, 0, SLAVE_TO_MASTER_TAG, MPI_COMM_WORLD, &request); //send lower bound
        MPI_Isend(&upper_bound, 1, MPI_INT, 0, SLAVE_TO_MASTER_TAG + 1, MPI_COMM_WORLD, &request); //send upper bound
        MPI_Isend(&mat_result[low_bound][0], (upper_bound - low_bound) * N, MPI_DOUBLE, 0, SLAVE_TO_MASTER_TAG + 2, MPI_COMM_WORLD, &request); //send calculated result
        } 
        if (rank == 0) { //if master
            for (i = 1; i < size; i++) {// untill all slaves have handed back the processed data 
                MPI_Recv(&low_bound, 1, MPI_INT, i, SLAVE_TO_MASTER_TAG, MPI_COMM_WORLD, &status); //recieve lower bound
                MPI_Recv(&upper_bound, 1, MPI_INT, i, SLAVE_TO_MASTER_TAG + 1, MPI_COMM_WORLD, &status);//recieve upper bound 
                MPI_Recv(&mat_result[low_bound][0], (upper_bound - low_bound) * N, MPI_DOUBLE, i, SLAVE_TO_MASTER_TAG + 2, MPI_COMM_WORLD, &status); //recieve calculated result
            } 
            end_time = MPI_Wtime(); //set end time as process is over
            printf("\nRunning Time = %f\n\n", end_time - start_time); //calculate & print running time
        } 
        MPI_Finalize(); //finalize MPI operations 
        return 0; 
    } 
    void makeAB() { 
        //initialzie matrix a
        for (i = 0; i < N; i++) { 
            for (j = 0; j < N; j++) { 
                mat_a[i][j] = i + j; 
            } 
        } 
        //initialzie matrix b
        for (i = 0; i < N; i++) { 
            for (j = 0; j < N; j++) { 
                mat_b[i][j] = i*j; 
            } 
        } 
    }
};

