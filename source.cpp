
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <tchar.h>
#include "PerformanceTimer.h"
#include "PThread\pthread.h"
#include "mpi.h"

#define NUM_THREADS 8

pthread_mutex_t g_Mutex;

int **g_MatrixA;
int **g_MatrixB;

int g_rowA = 0, g_colA = 0;
int g_rowB = 0, g_colB = 0;

FILE *fileA1;
FILE *fileA2;
FILE *fileB1;
FILE *fileB2;
FILE *fileC;

void *threadFunc(void *pArg){
	int tID = *((int*)pArg);
	switch (tID){
	case 0:{
		for (int i = 0; i < g_rowA / 2; i++){
			fread(g_MatrixA[i], sizeof(int), g_colA, fileA1);
		}
		break;
	}
	case 1:{
		fseek(fileA2, ((g_rowA / 2)*g_colA * 4) + 8, SEEK_SET);
		for (int i = g_rowA / 2; i < g_rowA; i++){
			fread(g_MatrixA[i], sizeof(int), g_colA, fileA2);
		}
		break;
	}
	case 2:{
		for (int i = 0; i < (g_rowB / 2); i++){
			fread(g_MatrixB[i], sizeof(int), g_colB, fileB1);
		}
		break;
	}
	case 3:{
		fseek(fileB2, ((g_rowB / 2)*g_colB * 4) + 8, SEEK_SET);
		for (int i = g_rowB / 2; i < g_rowB; i++){
			fread(g_MatrixB[i], sizeof(int), g_colB, fileB2);
		}
		break;
	}
	}
	return NULL;
}

int **allocate_2d_array(int rows, int cols){
	int **result = (int **)malloc(rows * sizeof(int *));
	result[0] = (int *)malloc(rows * cols * sizeof(int));

	for (int i = 1; i < rows; i++)
		result[i] = result[i - 1] + cols;
	
	return result;
}

void deallocate_2d_array(int **array, int rows){
	for (int i = 1; i < rows; i++)
		array[i] = NULL;
	free(array[0]);
	free(array);
}

int main(int argc, char* argv[])
{
	// Initialize performance timer
	InitializeTimer();
	int myRank, totalProcs;
	int numPerProcs, numPerProcsRemain;

	int i = 0, j = 0, k = 0;
	pthread_t pThreads[NUM_THREADS];
	int tIDs[NUM_THREADS];
	int **matrixA; //Matrix A
	int **matrixB; //matrix B
	int **matrixC; //final Matrix C
	int **cpart; //parts of the sum of Matrix C for each
	int *temp_1dA1;
	int *temp_1dA2;
	int *temp_1dB1;
	int *temp_1dB2;

	int rowC = 0, colC = 0;
	int offsetBCol = 0;
	int offsetBRow = 0;
	char fileName[256] = { 0 };
	unsigned long long startTime;
	unsigned long long endTime;
	unsigned long long processTime;

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &totalProcs);

	// 1. Read Matrix A
	if (myRank == 0){
		startTime = GetCustomCurrentTime();

		int rowA = 0, colA = 0;
		printf("Matrix Multiplication using Multi-Dimension Arrays - Start\n\n");
		printf("Reading Matrix A - Start\n");
		fileA1 = fopen("ma.bin", "rb"); // open the file for binary reading (notice the "rb").
		fileA2 = fopen("ma.bin", "rb");
		fread(&g_rowA, sizeof(int), 1, fileA1); // Use fread to read a single element
		fread(&g_colA, sizeof(int), 1, fileA1); // Use fread to read a single element
		g_MatrixA = allocate_2d_array(g_rowA, g_colA);

		fileB1 = fopen("mb.bin", "rb");
		fileB2 = fopen("mb.bin", "rb");
		fread(&g_rowB, sizeof(int), 1, fileB1); // Use fread to read a single element
		fread(&g_colB, sizeof(int), 1, fileB1); // Use fread to read a single element
		g_MatrixB = allocate_2d_array(g_rowB, g_colB);

		pthread_mutex_init(&g_Mutex, NULL);
		for (i = 0; i < NUM_THREADS; i++){
			tIDs[i] = i;
			pthread_create(&pThreads[i], NULL, threadFunc, &tIDs[i]);
		}
		for (i = 0; i < NUM_THREADS; i++)
			pthread_join(pThreads[i], NULL);
		pthread_mutex_destroy(&g_Mutex);
		fclose(fileA1);
		fclose(fileA2);
		fclose(fileB1);
		fclose(fileB2);

		endTime = GetCustomCurrentTime();
		processTime = endTime - startTime;
		printf("Process time (ms) for Reading File into Matrix is: %lld\n", processTime / 1000);
	}
	MPI_Bcast(&g_rowA, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&g_colA, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&g_rowB, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&g_colB, 1, MPI_INT, 0, MPI_COMM_WORLD);

	//Variables for each process that handles num of processes in both matrix A & B
	int sizeRowPerProcsA = g_rowA / 2;
	int sizeRowPerProcsARemain = g_rowA % 2;
	int sizeColPerProcsA = g_colA / 2;
	int sizeRowPerProcsB = g_rowB / 2;
	int sizeColPerProcsB = g_colB / 2;
	int sizeColPerProcsBRemain = g_colB % 2;
	int sizeRowPerPartC = g_rowA / 2;
	int sizeRowPerPartCRemain = g_rowA % 2;
	int sizeColPerPartC = g_colB / 2;
	int sizeColPerPartCRemain = g_colB % 2;

	rowC = g_rowA;
	colC = g_colB;
	
	//2. Storing 2D matrix into 1D and send to each node
	switch (myRank){
	case 0: {
		cpart = allocate_2d_array(sizeRowPerPartC, sizeColPerPartC);
		temp_1dA1 = (int *)malloc(sizeRowPerProcsA * g_colA * sizeof(int));
		if(sizeRowPerProcsARemain != 0){
			temp_1dA2 = (int *)malloc((sizeRowPerProcsA + sizeRowPerProcsARemain) * g_colA * sizeof(int));
		} else{
			temp_1dA2 = (int *)malloc(sizeRowPerProcsA * g_colA * sizeof(int));
		}
		temp_1dB1 = (int *)malloc(g_rowB * sizeColPerProcsB * sizeof(int));
		if(sizeColPerProcsBRemain != 0){
			temp_1dB2 = (int *)malloc(g_rowB * (sizeColPerProcsB + sizeColPerProcsBRemain) * sizeof(int));
		} else {
			temp_1dB2 = (int *)malloc(g_rowB * sizeColPerProcsB * sizeof(int));
		}
		matrixC = allocate_2d_array(rowC, colC);
		int l = 0;
		for (int i = 0; i < g_rowB; i++){
			for (int j = 0; j < sizeColPerProcsB; j++){
				temp_1dB1[l] = g_MatrixB[i][j + offsetBCol];
				l++;
			}
		}

		l = 0;
		offsetBCol += sizeColPerProcsB;
		for (int i = 0; i < g_rowB; i++){
			for (int j = 0; j < sizeColPerProcsB + sizeColPerProcsBRemain; j++){
				temp_1dB2[l] = g_MatrixB[i][j + offsetBCol];
				l++;
			}
		}

		l = 0;
		offsetBRow = 0;
		for (int i = 0; i < sizeRowPerProcsA; i++){
			for (int j = 0; j < g_colA; j++){
				temp_1dA1[l] = g_MatrixA[i + offsetBRow][j];
				l++;
			}
		}

		l = 0;
		offsetBRow += sizeRowPerProcsA;
		for (int i = 0; i < sizeRowPerProcsA + sizeRowPerProcsARemain; i++){
			for (int j = 0; j < g_colA; j++){
				temp_1dA2[l] = g_MatrixA[i + offsetBRow][j];
				l++;
			}
		}
		/*
		Proc 1: temp_1dA1 x temp_1dB1;
		Proc 2: temp_1dA1 x temp_1dB2;
		Proc 3: temp_1dA2 x temp_1dB1;
		Proc 4: temp_1dA2 x temp_1dB2;
		*/
		for (int i = 1; i < totalProcs; i++){
			if (i == 1){
				MPI_Send(&temp_1dA1[0], sizeRowPerProcsA * g_colA, MPI_INT, i, 0, MPI_COMM_WORLD);
				MPI_Send(&temp_1dB2[0], g_rowB * (sizeColPerProcsBRemain+ sizeColPerProcsB), MPI_INT, i, 0, MPI_COMM_WORLD);
			}
			else if (i == 2){
				MPI_Send(&temp_1dA2[0], (sizeRowPerProcsARemain+sizeRowPerProcsA) * g_colA, MPI_INT, i, 0, MPI_COMM_WORLD);
				MPI_Send(&temp_1dB1[0], g_rowB * sizeColPerProcsB, MPI_INT, i, 0, MPI_COMM_WORLD);
			}
			else if (i == 3){
				MPI_Send(&temp_1dA2[0], (sizeRowPerProcsARemain + sizeRowPerProcsA) * g_colA, MPI_INT, i, 0, MPI_COMM_WORLD);
				MPI_Send(&temp_1dB2[0], g_rowB * (sizeColPerProcsBRemain+ sizeColPerProcsB), MPI_INT, i, 0, MPI_COMM_WORLD);
			}
		}

		//Matrix Multiplication for root node
		for (int i = 0; i < sizeRowPerPartC; i++){
			for (int j = 0; j < sizeColPerPartC; j++){
				cpart[i][j] = 0.0e0;
			}
		}
		for (int i = 0; i < sizeRowPerPartC; i++){
			for (int k = 0; k < g_colA; k++){
				for (int j = 0; j < sizeColPerPartC; j++){
					cpart[i][j] += g_MatrixA[i][k] * g_MatrixB[k][j];
				}
			}
		}
		
		break;
	}
	default: {
		//3. Calculations for each partition of Matrix C
		if (myRank == 1){
			cpart = allocate_2d_array(sizeRowPerPartC, sizeColPerPartC + sizeColPerProcsBRemain);
			matrixA = allocate_2d_array(sizeRowPerProcsA, g_colA);
			matrixB = allocate_2d_array(g_rowB, sizeColPerProcsB + sizeColPerProcsBRemain);
			temp_1dA1 = (int *)malloc(sizeRowPerProcsA * g_colA * sizeof(int));
			temp_1dB2 = (int *)malloc(g_rowB * (sizeColPerProcsBRemain +sizeColPerProcsB) * sizeof(int));
			MPI_Recv(&temp_1dA1[0], sizeRowPerProcsA * g_colA, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
			MPI_Recv(&temp_1dB2[0], g_rowB * (sizeColPerProcsBRemain+sizeColPerProcsB), MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
			int l = 0;
			for (int i = 0; i < g_rowB; i++){
				for (int j = 0; j < sizeColPerProcsB + sizeColPerProcsBRemain; j++){
					matrixB[i][j] = temp_1dB2[l];
					l++;
				}
			}
			l = 0;
			for (int i = 0; i < sizeRowPerProcsA; i++){
				for (int j = 0; j < g_colA; j++){
					matrixA[i][j] = temp_1dA1[l];
					l++;
				}
			}
			for (int i = 0; i < sizeRowPerPartC; i++){
				for (int j = 0; j < sizeColPerPartC + sizeColPerProcsBRemain; j++){
					cpart[i][j] = 0.0e0;
				}
			}

			for (int i = 0; i < sizeRowPerPartC; i++){
				for (int k = 0; k < g_colA; k++){
					for (int j = 0; j < sizeColPerPartC + sizeColPerProcsBRemain; j++){
						cpart[i][j] += matrixA[i][k] * matrixB[k][j];
					}
				}
			}
			free(temp_1dA1);
			free(temp_1dB2);
			MPI_Send(&cpart[0][0], sizeRowPerPartC * (sizeColPerPartC + sizeColPerProcsBRemain), MPI_INT, 0, 0, MPI_COMM_WORLD);
		}
		else if (myRank == 2){
			cpart = allocate_2d_array(sizeRowPerPartC + sizeRowPerProcsARemain, sizeColPerPartC);
			matrixA = allocate_2d_array(sizeRowPerProcsA + sizeRowPerProcsARemain, g_colA);
			matrixB = allocate_2d_array(g_rowB, sizeColPerProcsB);
			temp_1dA2 = (int *)malloc((sizeRowPerProcsARemain+sizeRowPerProcsA) * g_colA * sizeof(int));
			temp_1dB1 = (int *)malloc(g_rowB * sizeColPerProcsB * sizeof(int));
			MPI_Recv(&temp_1dA2[0], (sizeRowPerProcsARemain+sizeRowPerProcsA) * g_colA, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
			MPI_Recv(&temp_1dB1[0], g_rowB * sizeColPerProcsB, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
			int l = 0;
			for (int i = 0; i < g_rowB; i++){
				for (int j = 0; j < sizeColPerProcsB; j++){
					matrixB[i][j] = temp_1dB1[l];
					l++;
				}
			}
			l = 0;
			for (int i = 0; i < sizeRowPerProcsA + sizeRowPerProcsARemain; i++){
				for (int j = 0; j < g_colA; j++){
					matrixA[i][j] = temp_1dA2[l];
					l++;
				}
			}
			for (int i = 0; i < sizeRowPerPartC + sizeRowPerProcsARemain; i++){
				for (int j = 0; j < sizeColPerPartC; j++){
					cpart[i][j] = 0.0e0;
				}
			}

			for (int i = 0; i < sizeRowPerPartC + sizeRowPerProcsARemain; i++){
				for (int k = 0; k < g_colA; k++){
					for (int j = 0; j < sizeColPerPartC; j++){
						cpart[i][j] += matrixA[i][k] * matrixB[k][j];
					}
				}
			}
			free(temp_1dA2);
			free(temp_1dB1);
			MPI_Send(&cpart[0][0], (sizeRowPerProcsARemain+sizeRowPerPartC) * sizeColPerPartC, MPI_INT, 0, 0, MPI_COMM_WORLD);
		}
		else if (myRank == 3){
			cpart = allocate_2d_array(sizeRowPerPartC + sizeRowPerProcsARemain, sizeColPerPartC + sizeColPerProcsBRemain);
			matrixA = allocate_2d_array(sizeRowPerProcsA + sizeRowPerProcsARemain, g_colA);
			matrixB = allocate_2d_array(g_rowB, sizeColPerProcsB + sizeColPerProcsBRemain);
			temp_1dA2 = (int *)malloc((sizeRowPerProcsARemain+sizeRowPerProcsA) * g_colA * sizeof(int));
			temp_1dB2 = (int *)malloc(g_rowB * (sizeColPerProcsBRemain+sizeColPerProcsB) * sizeof(int));
			MPI_Recv(&temp_1dA2[0], (sizeRowPerProcsARemain+sizeRowPerProcsA) * g_colA, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
			MPI_Recv(&temp_1dB2[0], g_rowB * (sizeColPerProcsBRemain+sizeColPerProcsB), MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
			int l = 0;
			for (int i = 0; i < g_rowB; i++){
				for (int j = 0; j < sizeColPerProcsB + sizeColPerProcsBRemain; j++){
					matrixB[i][j] = temp_1dB2[l];
					l++;
				}
			}
			l = 0;
			for (int i = 0; i < sizeRowPerProcsA + sizeRowPerProcsARemain; i++){
				for (int j = 0; j < g_colA; j++){
					matrixA[i][j] = temp_1dA2[l];
					l++;
				}
			}
			for (int i = 0; i < sizeRowPerPartC + sizeRowPerProcsARemain; i++){
				for (int j = 0; j < sizeColPerPartC + sizeColPerProcsBRemain; j++){
					cpart[i][j] = 0.0e0;
				}
			}
			for (int i = 0; i < sizeRowPerPartC + sizeRowPerProcsARemain; i++){
				for (int k = 0; k < g_colA; k++){
					for (int j = 0; j < sizeColPerPartC + sizeColPerProcsBRemain; j++){
						cpart[i][j] += matrixA[i][k] * matrixB[k][j];
					}
				}
			}

			free(temp_1dA2);
			free(temp_1dB2);
			MPI_Send(&cpart[0][0], (sizeRowPerPartC + sizeRowPerProcsARemain) * (sizeColPerPartC+sizeColPerProcsBRemain), MPI_INT, 0, 0, MPI_COMM_WORLD);
			}
		}
	break;
	}
	
	//4.Finalize the Matrix C and it to node 0 (Root)
	if (myRank == 0){
		int offsetRow = 0;
		int offsetCol = 0;

		int**cpart1 = allocate_2d_array(sizeRowPerPartC, sizeColPerPartC + sizeColPerProcsBRemain);
		int**cpart2 = allocate_2d_array(sizeRowPerPartC + sizeRowPerProcsARemain, sizeColPerPartC);
		int**cpart3 = allocate_2d_array(sizeRowPerPartC + sizeRowPerProcsARemain, sizeColPerPartC + sizeColPerProcsBRemain);
			for (i = 0; i < sizeRowPerPartC; i++){
				for (j = 0; j < sizeColPerPartC; j++){
					matrixC[i][j] = cpart[i][j];
				}
			}
		MPI_Recv(&cpart1[0][0], sizeRowPerPartC * (sizeColPerPartC + sizeColPerProcsBRemain), MPI_INT, 1, 0, MPI_COMM_WORLD, &status);
		offsetCol += sizeColPerPartC;
		for (i = 0; i < sizeRowPerPartC; i++){
			for (j = 0; j < sizeColPerPartC + sizeColPerProcsBRemain; j++){
				matrixC[i][j + offsetCol] = cpart1[i][j];
			}
		}
		MPI_Recv(&cpart2[0][0], (sizeRowPerProcsARemain+sizeRowPerPartC) * sizeColPerPartC, MPI_INT, 2, 0, MPI_COMM_WORLD, &status);
		offsetCol = 0;
		offsetRow += sizeRowPerPartC;
		for (i = 0; i < sizeRowPerPartC + sizeRowPerProcsARemain; i++){
			for (j = 0; j < sizeColPerPartC; j++){
				matrixC[i + offsetRow][j] = cpart2[i][j];
			}
		}
		MPI_Recv(&cpart3[0][0], (sizeRowPerPartC + sizeRowPerProcsARemain) * (sizeColPerPartC+sizeColPerProcsBRemain), MPI_INT, 3, 0, MPI_COMM_WORLD, &status);
		offsetCol += sizeColPerPartC;
		for (i = 0; i < sizeRowPerPartC+sizeRowPerProcsARemain; i++){
			for (j = 0; j < sizeColPerPartC+sizeColPerProcsBRemain; j++){
				matrixC[i + offsetRow][j + offsetCol] = cpart3[i][j];
			}
		}
	/*endTime = GetCustomCurrentTime();
	processTime = endTime - startTime;
	printf("Process time (ms) for Matrix Multiplication: %lld\n", processTime / 1000);
	}*/
	}
	//5. Save the output to "mc.bin"
	if (myRank == 0){
		startTime = GetCustomCurrentTime();
		printf("\nFinal Result in Matrix C is: \n");
			for (int i = 0; i < rowC; i++){
				for (int j = 0; j < colC; j++){
					printf("%d\t", matrixC[i][j]);
				}
			printf("\n");
		}
		fileC = fopen("mc.bin", "wb");
		fwrite(&rowC, sizeof(int), 1, fileC);
		fwrite(&colC, sizeof(int), 1, fileC);
		for (int i = 0; i < rowC; i++){
			fwrite(matrixC[i], sizeof(int), colC, fileC);
		}
		fclose(fileC);
		endTime = GetCustomCurrentTime();
		processTime = endTime - startTime;
		printf("Process time (ms) for Writing output to file is: %lld\n", processTime / 1000);
		printf("Results done printed!");
	}
	
	// 5. Cleaning up!
	if (myRank == 0){
		deallocate_2d_array(matrixC, rowC);
		deallocate_2d_array(cpart, sizeRowPerPartC);
		free(temp_1dA1);
		free(temp_1dA2);
		free(temp_1dB1);
		free(temp_1dB2);
		printf("Matrix Multiplication using Multi-Dimension Arrays - Done\n");
	}
	else {
		deallocate_2d_array(matrixA, sizeRowPerProcsA);
		deallocate_2d_array(matrixB, sizeRowPerProcsB);
		deallocate_2d_array(cpart, sizeRowPerPartC);
	}

	MPI_Finalize();
	ShutdownTimer();
	return 0;
}
