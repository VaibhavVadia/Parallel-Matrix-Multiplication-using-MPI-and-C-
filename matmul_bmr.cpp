#include <mpi.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <math.h> 
#include <string.h> 
//N=24x28, 26x28 and 28x28
#define N 448 
int main(int argc,char *argv[]){ 
    int i,j,k;  
    double a[N][N],b[N][N],c[N][N]; 
    int numP,rank; 
    double timeCost; 
    MPI_Init(&argc,&argv); 
    MPI_Comm_size(MPI_COMM_WORLD, &numP); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    if(rank == 0) 
        timeCost = MPI_Wtime(); 
    int procGrid = sqrt(numP); 
    int blockLength = N/procGrid; 
    int localA[blockLength][blockLength],localB[blockLength][blockLength],localResult[blockLength][blockLength]; 
    for(i=0;i<blockLength;i++){ 
        for(j=0;j<blockLength;j++){ 
            localResult[i][j] = 0; 
        } 
    }   
    if(rank == 0){ 
        int count = 0; 
        for (i=0; i<N; i++) { 
            for (j=0; j<N; j++){ 
                float v= (1+0.9)*((float)rand() /RAND_MAX) -0.9; 
                a[i][j] =v ; 
                b[i][j] = v; 
                count++; 
            } 
        } 
    } 
    int sizes[2] = {N, N}; 
    int subsizes[2] = {blockLength, blockLength}; 
    int starts[2] = {0,0}; 
    MPI_Datatype type, subArrayType;    
    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &type); 
    MPI_Type_create_resized(type, 0, blockLength*sizeof(int), &subArrayType); 
    MPI_Type_commit(&subArrayType);     
    if(rank == 0){ 
        for(i = 0;i<blockLength;i++){ 
            memcpy(&localA[i][0],&a[i][0],sizeof(int)*blockLength); 
            memcpy(&localB[i][0],&b[i][0],sizeof(int)*blockLength); 
        } 
        for(i=0;i<numP;i++){ 
            int pAi = floor(blockLength*i/N); 
            int pAj = i - pAi*procGrid; 
            int pBi = floor(blockLength*i/N); 
            int pBj = i - pBi*procGrid; 
            int startAx = 0, startAy = 0; 
            int startBx = 0, startBy = 0; 
            startAx = ((pAj-pAi)<0)?(procGrid+pAj-pAi)*blockLength:(pAj-pAi)*blockLength; 
            startAy = pAi*blockLength; 
            MPI_Isend(&a[startAy][startAx],1,subArrayType,i,0,MPI_COMM_WORLD,&sendReqs[0]); 
            startBy = ((pBi-pBj)<0)?(procGrid + pBi-pBj)*blockLength:(pBi-pBj)*blockLength; 
            startBx = pBj*blockLength; 
            MPI_Request sendReqs[2]; 
            MPI_Status sendSta[2]; 
            int numCompleted,finishedReqs[2]; 
            MPI_Isend(&a[startAy][startAx],1,subArrayType,i,0,MPI_COMM_WORLD,&sendReqs[0]); 
            MPI_Isend(&b[startBy][startBx],1,subArrayType,i,1,MPI_COMM_WORLD,&sendReqs[1]); 
        } 
    } 
    int calCount = 0; 
    while(calCount < procGrid){ 
        MPI_Request recvReqs[2]; 
        MPI_Status recvSta[2]; 
        if(calCount == 0){ 
            MPI_Irecv(&localA[0][0],blockLength*blockLength,MPI_INT,0,0,MPI_COMM_WORLD,&recvReqs[0]); 
            MPI_Irecv(&localB[0][0],blockLength*blockLength,MPI_INT,0,1,MPI_COMM_WORLD,&recvReqs[1]); 
        }else{ 
            int sourceA = ((rank+1)%procGrid == 0)?rank+1-procGrid:rank+1; 
            int sourceB = (floor(rank/procGrid) != (procGrid-1))?rank+procGrid:rank-procGrid*(procGrid-1); 
            MPI_Irecv(&localA[0][0],blockLength*blockLength,MPI_INT,sourceA,0,MPI_COMM_WORLD,&recvReqs[0]); MPI_Irecv(&localB[0][0],blockLength*blockLength,MPI_INT,sourceB,1,MPI_COMM_WORLD,&recvReqs[1]); 
        } 
        MPI_Wait(&recvReqs[0],&recvSta[0]); 
        MPI_Wait(&recvReqs[1],&recvSta[1]); 
        int localTempA[blockLength][blockLength],localTempB[blockLength][blockLength]; 
        memcpy(&localTempA[0][0],&localA[0][0],sizeof(int)*blockLength*blockLength); 
        memcpy(&localTempB[0][0],&localB[0][0],sizeof(int)*blockLength*blockLength); 
        MPI_Request sendReqs[2]; 
        MPI_Status sendSta[2]; 
        int nextAid = ((rank%procGrid) != 0)?rank-1:rank-1+N/blockLength; 
        MPI_Isend(&localA[0][0],blockLength*blockLength,MPI_INT,nextAid,0,MPI_COMM_WORLD,&sendReqs[0]); 
        int nextBid = (floor(rank/procGrid) != 0)?rank-procGrid:rank+procGrid*(procGrid-1); 
        MPI_Isend(&localB[0][0],blockLength*blockLength,MPI_INT,nextBid,1,MPI_COMM_WORLD,&sendReqs[1]); 
        for(i=0;i<blockLength;i++){ 
            for(j=0;j<blockLength;j++){ 
                int sum = 0; 
                for(k=0;k<blockLength;k++){ 
                    sum += localTempA[i][k]*localTempB[k][j]; 
                } 
                localResult[i][j] += sum; 
            } 
        } 
        calCount ++; 
    } 
    int *startPtrC = (rank == 0)?&(c[0][0]):NULL; 
    /* scatter initial block to all processors */ 
    int sendCounts[numP]; 
    int displs[numP]; 
    if (rank == 0) { 
        for (i=0; i<numP; i++) 
            sendCounts[i] = 1; 
        int disp = 0; 
        for (i=0; i<procGrid; i++) { 
            for (j=0; j<procGrid; j++) { 
                displs[i*procGrid+j] = disp; 
                disp += 1; 
            } 
            disp += (blockLength-1)*procGrid; 
        } 
    } 
    MPI_Gatherv(&(localResult[0][0]), blockLength*blockLength,MPI_INT,startPtrC,sendCounts,displs,subArrayType,0,MPI_COMM_WORLD); 
    if(rank == 0) printf("Cannon time cost for size %d is %lf\n",N,MPI_Wtime()-timeCost); 
    MPI_Type_free(&subArrayType); 
    MPI_Finalize(); 
}
