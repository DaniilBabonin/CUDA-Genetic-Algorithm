#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
using namespace std;
#include <stdio.h>
#include <algorithm>
#include <ctime>
#include <cstdlib> 


const int sizePoint = 700;
const int sizeIndividum = 1000;
const int MutationProbability = 10;
const float MutationDispersion = 5.0f;
const int Polynom = 3;
const float randMaxCount = 20.0f;
const int maxGeneration = 30;

__global__ void errorsKernel(float* points, float* individs, float* errors, int Polynom, int sizePoint)					// ¬ычисл€ем ошибки на GPU
{

	int id = threadIdx.x;
	float ans = 0;
	int x = 1;
	for (int i = 0; i < sizePoint; i++)
	{
		for (int j = 0; j < Polynom; j++)
		{
			for (int k = 0; k < j; k++)
			{
				x *= i;
			}
			x *= individs[id * Polynom + j];
			ans += x;
			x = 1;
		}

		ans = points[i] - ans;
		errors[id] += sqrt(ans * ans);
		ans = 0;
	}
}


void testErrorsKernel(float* points, float* individs, float* errors, int Polynom, int sizePoint, int sizeIndividum)		// ¬ычисл€ем ошибки на CPU
{
	for (int id = 0; id < sizeIndividum; id++)
	{
		float ans = 0.0f;
		errors[id] = 0.0f;
		int x = 0;
		for (int i = 0; i < sizePoint; i++)
		{
			for (int j = 0; j < Polynom; j++)
			{
				x = pow(i, j);
				x *= individs[id * Polynom + j];
				ans += x;
				x = 0;
			}

			ans = points[i] - ans;
			errors[id] += sqrt(ans * ans);
			ans = 0;
		}
	}
}

float RandomFloat(float a, float b) {
	float random = ((float)rand()) / (float)RAND_MAX;
	float diff = b - a;
	float r = random * diff;
	return a + r;
}

void cpu() {
	float* pointsH = new float[sizePoint]; 
	for (int i = 0; i < sizePoint; i++)							// создаем случайный набор точек
	{
		pointsH[i] = RandomFloat(0, randMaxCount);
	}

	float* individumsH = new float[sizeIndividum * Polynom];
	for (int i = 0; i < sizeIndividum * Polynom; i++)			// создаем первое поколение
	{
		individumsH[i] = RandomFloat(0, randMaxCount);			
	}

	float* errorsH = new float[sizeIndividum];
	for (int i = 0; i < sizeIndividum; i++)
	{
		errorsH[i] = 1000;
	}

	unsigned int start_time_cpu = clock(); 

	for (int generation = 0; generation < maxGeneration; generation++)
	{
		testErrorsKernel(pointsH, individumsH, errorsH, Polynom, sizePoint, sizeIndividum);

		float* errorsCrossOver = new float[sizeIndividum];

		for (size_t i = 0; i != sizeIndividum; ++i)
			errorsCrossOver[i] = errorsH[i];
		sort(errorsCrossOver, errorsCrossOver + sizeIndividum);

		int merodianCrossOvering = sizeIndividum / 2;
		float merodianErrorCrossOvering = errorsCrossOver[merodianCrossOvering];
		float* theBestInd = new float[Polynom];

		for (size_t i = 0; i < sizeIndividum; i++)			// смена поколени€
		{
			if (merodianErrorCrossOvering < errorsH[i]) {
				for (size_t j = 0; j < Polynom; j++)
				{
					individumsH[i * Polynom + j] = 0;
				}
			}
			if (errorsH[i] == errorsCrossOver[0]) {
				for (int j = 0; j < Polynom; j++)
				{
					theBestInd[j] = individumsH[i * Polynom + j];
				}
			}
		}

		printf("Error = %f\n", errorsCrossOver[0]);

		for (int i = 0; i < sizeIndividum * Polynom; i++)
		{
			if (individumsH[i] == 0) {
				individumsH[i] = theBestInd[rand() % Polynom];
			}

			if (MutationProbability > (rand() % 100 + 1)) {
				individumsH[i] += RandomFloat(-MutationDispersion, MutationDispersion);
			}
		}
	}
	unsigned int end_time_cpu = clock(); 
	unsigned int search_time_cpu = end_time_cpu - start_time_cpu;
	printf("Time CPU = %i\n", search_time_cpu);
}

void gpu() {
	float* pointsH = new float[sizePoint];
	for (int i = 0; i < sizePoint; i++)				// создаем случайный набор точек
	{  
		pointsH[i] = RandomFloat(0, randMaxCount);
	}

	float* individumsH = new float[sizeIndividum * Polynom];
	for (int i = 0; i < sizeIndividum * Polynom; i++)	// создаем первое поколение
	{
		individumsH[i] = RandomFloat(0, randMaxCount);
	}

	float* errorsH = new float[sizeIndividum];
	for (int i = 0; i < sizeIndividum; i++)
	{
		errorsH[i] = 1000;
	}

	unsigned int start_time_gpu = clock();
	float* pointsD = NULL;
	float* individumsD = NULL;
	float* errorsD = NULL;

	for (int generation = 0; generation < maxGeneration; generation++)
	{

		int sizeIndividumBytes = sizeIndividum * Polynom * sizeof(float);
		int sizePointBytes = sizePoint * sizeof(float);

		cudaMalloc((void**)&pointsD, sizePointBytes);
		cudaMalloc((void**)&individumsD, sizeIndividumBytes * Polynom);
		cudaMalloc((void**)&errorsD, sizeIndividum * sizeof(float));

		cudaMemcpy(pointsD, pointsH, sizePointBytes, cudaMemcpyHostToDevice);
		cudaMemcpy(individumsD, individumsH, sizeIndividumBytes, cudaMemcpyHostToDevice);
		cudaMemcpy(errorsD, errorsH, sizeIndividumBytes, cudaMemcpyHostToDevice);

		errorsKernel << <1, sizeIndividum >> > (pointsD, individumsD, errorsD, Polynom, sizePoint);

		cudaMemcpy(errorsH, errorsD, sizeIndividum * sizeof(float), cudaMemcpyDeviceToHost);

		//----------------------
		float* errorsCrossOver = new float[sizeIndividum];

		for (size_t i = 0; i != sizeIndividum; ++i)
			errorsCrossOver[i] = errorsH[i];
		sort(errorsCrossOver, errorsCrossOver + sizeIndividum);
		printf("Error = %f\n", errorsCrossOver[0]);
		int merodianCrossOvering = sizeIndividum / 2;
		float merodianErrorCrossOvering = errorsCrossOver[merodianCrossOvering];
		float* theBestInd = new float[Polynom];

		for (size_t i = 0; i < sizeIndividum; i++)				// —мена поколени€
		{
			if (merodianErrorCrossOvering < errorsH[i]) {
				for (size_t j = 0; j < Polynom; j++)
				{
					individumsH[i * Polynom + j] = 0;
				}
			}
			if (errorsH[i] == errorsCrossOver[0]) {
				for (int j = 0; j < Polynom; j++)
				{
					theBestInd[j] = individumsH[i * Polynom + j];
				}
			}
		}

		for (int i = 0; i < sizeIndividum * Polynom; i++)
		{
			if (individumsH[i] == 0) {
				individumsH[i] = theBestInd[rand() % Polynom];
			}

			if (MutationProbability > (rand() % 100 + 1)) {
				individumsH[i] += RandomFloat(-MutationDispersion, MutationDispersion);
			}
		}
	}
	unsigned int end_time_gpu = clock();
	unsigned int search_time_gpu = end_time_gpu - start_time_gpu;

	printf("Time GPU = %i\n", search_time_gpu);
	
	cudaFree(pointsD);
	cudaFree(individumsD);
	cudaFree(errorsD);

	delete pointsH;
	delete individumsH;
	delete errorsH;
}

int main()
{
	cpu();
	gpu();
	system("pause");
	return 0;
}
