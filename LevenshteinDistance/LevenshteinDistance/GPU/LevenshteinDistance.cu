#include "LevenshteinDistance.cuh"
#include "../Timers/TimerManager.h"
#include "DeviceRawData.cuh"

int GPU::LevenshteinDistance::CalculateLevenshteinDistance(const std::string& sourceWord,
	const std::string& targetWord,
	std::string& transformation,
	bool showTables)
{
	auto& timerManager = Timers::TimerManager::GetInstance();

	auto data = DeviceRawData(
		static_cast<unsigned>(sourceWord.size()), 
		static_cast<unsigned>(targetWord.size())
	);

	auto threadsInBlock = std::min(data.TargetWordLength + 1, THREADS_IN_ONE_BLOCK);
	auto blocksInGrid = (data.TargetWordLength + threadsInBlock) / threadsInBlock;
	auto hostNextColumn = 0;

	std::cout << "Transferring data from host to device..." << std::endl << std::endl;

	timerManager.Host2DeviceDataTransferTimer.Start();
	data.FromHost(sourceWord, targetWord, &hostNextColumn);
	timerManager.Host2DeviceDataTransferTimer.Stop();

	std::cout << "Starting computation..." << std::endl << std::endl;
	std::cout << " -> Populating X..." << std::endl;

	timerManager.PopulateDeviceXTimer.Start();

	PopulateDeviceX << <1u, AlphabetLength, (AlphabetLength + 1) * sizeof(char) >> > (
		data.DeviceX,
		data.DeviceAlphabet,
		AlphabetLength,
		data.DeviceTargetWord,
		data.TargetWordLength
	);

	CUDACHECK(cudaPeekAtLastError());
	CUDACHECK(cudaDeviceSynchronize());

	timerManager.PopulateDeviceXTimer.Stop();

	std::cout << std::setw(35) << std::left << "    Elapsed time: "
		<< timerManager.PopulateDeviceXTimer.ElapsedMiliseconds() << " ms" << std::endl;

	std::cout << " -> Populating distances..." << std::endl;

	timerManager.PopulateDeviceDistancesTimer.Start();

	for (int i = 0; i < blocksInGrid; i++)
	{
		PopulateDeviceDistances << <1u, threadsInBlock, (threadsInBlock + data.SourceWordLength) * sizeof(char) >> > (
			data.DeviceDistances,
			data.DeviceTransformations,
			data.DeviceX,
			data.DeviceSourceWord,
			data.SourceWordLength,
			data.DeviceTargetWord,
			data.TargetWordLength,
			WARP_SIZE,
			data.DeviceNextColumn
		);

		CUDACHECK(cudaPeekAtLastError());
		CUDACHECK(cudaDeviceSynchronize());
	}

	timerManager.PopulateDeviceDistancesTimer.Stop();

	std::cout << std::setw(35) << std::left << "    Elapsed time: "
		<< timerManager.PopulateDeviceDistancesTimer.ElapsedMiliseconds() << " ms" << std::endl;

	int* hostDistances = new int[(data.SourceWordLength + 1) * (data.TargetWordLength + 1)];
	char* hostTransformations = new char[(data.SourceWordLength + 1) * (data.TargetWordLength + 1)];

	std::cout << std::endl << "Transferring data from device to host..." << std::endl << std::endl;

	timerManager.Device2HostDataTransferTimer.Start();
	data.ToHost(&hostDistances, &hostTransformations);
	timerManager.Device2HostDataTransferTimer.Stop();

	int distance = hostDistances[data.SourceWordLength * (data.TargetWordLength + 1) + data.TargetWordLength];

	std::cout << " -> Retrieving transformation..." << std::endl;

	timerManager.RetrieveTransformationTimer.Start();
	transformation = RetrieveTransformation(hostTransformations, data.SourceWordLength, data.TargetWordLength);
	timerManager.RetrieveTransformationTimer.Stop();

	std::cout << std::setw(35) << std::left << "    Elapsed time: "
		<< timerManager.RetrieveTransformationTimer.ElapsedMiliseconds() << " ms" << std::endl;

	if (showTables) {
		std::cout << std::endl <<"Distances:" << std::endl << std::endl;
		PrintMatrix(hostDistances, sourceWord, targetWord);

		std::cout << std::endl << "Transformations:" << std::endl << std::endl;
		PrintMatrix(hostTransformations, sourceWord, targetWord);

		std::cout << std::endl;
	}

	delete[] hostTransformations;
	delete[] hostDistances;

	return distance;
}

__device__
int GPU::ResolveTransformation(int s, int d, int i, char* transformation)
{
	int result = s;
	*transformation = SUBSTITUTE;

	if (d < result)
	{
		result = d;
		*transformation = DELETE;
	}

	if (i < result)
	{
		result = i;
		*transformation = INSERT;
	}

	return result;
}

std::string GPU::LevenshteinDistance::RetrieveTransformation(char* transformations, int m, int n)
{
	std::string transformation{};

	int i{ m }, j{ n };

	char current{};

	while (i != 0 || j != 0)
	{
		current = transformations[i * (n + 1) + j];
		transformation.push_back(current);

		if (current == DELETE) {
			i--;
		}
		else if (current == INSERT) {
			j--;
		}
		else {
			i--;
			j--;
		}
	}

	std::reverse(
		transformation.begin(),
		transformation.end()
	);

	return transformation;
}

__global__
void GPU::PopulateDeviceX(
	int* deviceX,
	char* deviceAlphabet,
	int alphabetLength,
	char* deviceTargetWord,
	int targetWordLength)
{
	int tid = threadIdx.x;

	extern __shared__ char sharedMemory[];

	sharedMemory[tid] = deviceAlphabet[tid];

	__syncthreads();

	int offset = tid * (targetWordLength + 1);

	deviceX[offset] = 0;

	for (int j = 1; j <= targetWordLength; ++j)
	{
		if (deviceTargetWord[j - 1] == sharedMemory[tid]) {
			deviceX[offset + j] = j;
		}
		else {
			deviceX[offset + j] = deviceX[offset + j - 1];
		}
	}
}

__global__
void GPU::PopulateDeviceDistances(
	int* deviceDistances,
	char* deviceTransformations,
	int* deviceX,
	char* deviceSourceWord,
	int sourceWordLenght,
	char* deviceTargetWord,
	int targetWordLength,
	int warpCount,
	int* deviceNextColumn)
{

	int tid = deviceNextColumn[0] + threadIdx.x;

	extern __shared__ char sharedMemory[];

	char* sharedSourceWord = sharedMemory + blockDim.x * sizeof(char);

	if (tid != 0 && tid <= targetWordLength) {
		sharedMemory[threadIdx.x] = deviceTargetWord[tid - 1];
	}

	int sharedSourceLength = (sourceWordLenght + blockDim.x) / blockDim.x;
	int sharedsourceStart = threadIdx.x * sharedSourceLength;

	for (int i = 0; i < sharedSourceLength && sharedsourceStart + i < sourceWordLenght; ++i) {
		sharedSourceWord[sharedsourceStart + i] = deviceSourceWord[sharedsourceStart + i];
	}

	__syncthreads();

	if (tid > targetWordLength) {
		return;
	}

	int aVar{}, bVar{}, cVar{}, dvar{ tid };
	char t{};

	deviceDistances[tid] = dvar;
	deviceTransformations[tid] = INSERT;

	for (int i = 1; i <= sourceWordLenght; ++i)
	{
		__syncthreads();

		int shuffledUp = __shfl_up(dvar, 1);

		if (tid != 0 && tid % warpCount == 0) {
			aVar = deviceDistances[(i - 1) * (targetWordLength + 1) + tid - 1];
		}
		else
		{
			if (tid != 0) {
				aVar = shuffledUp;
			}
		}

		char letter = sharedSourceWord[i - 1];
		int letterOffset = letter - 'a';

		int xVal = deviceX[letterOffset * (targetWordLength + 1) + tid];

		cVar = deviceDistances[(i - 1) * (targetWordLength + 1) + xVal - 1];

		__syncthreads();

		if (tid == 0)
		{
			dvar = i;
			t = DELETE;
		}

		else
		{

			if (sharedMemory[threadIdx.x] == letter)
			{
				dvar = aVar;
				t = SKIP;
			}
			else
			{
				bVar = dvar;

				if (xVal == 0) {
					dvar = 1 + GPU::ResolveTransformation(aVar, bVar, i + tid - 1, &t);
				}
				else {
					dvar = 1 + GPU::ResolveTransformation(aVar, bVar, cVar + (tid - 1 - xVal), &t);
				}
			}
		}

		deviceDistances[i * (targetWordLength + 1) + tid] = dvar;
		deviceTransformations[i * (targetWordLength + 1) + tid] = t;
	}

	if (threadIdx.x == 0) {
		deviceNextColumn[0] += blockDim.x;
	}
}