#include "DeviceRawData.cuh"

DeviceRawData::DeviceRawData(unsigned sourceWordLength, unsigned targetWordLength) :
	SourceWordLength{ sourceWordLength },
	TargetWordLength{ targetWordLength }
{
	CUDACHECK(cudaMalloc((void**)&DeviceAlphabet, (AlphabetLength + 1) * sizeof(char)));
	CUDACHECK(cudaMalloc((void**)&DeviceSourceWord, (SourceWordLength + 1) * sizeof(char)));
	CUDACHECK(cudaMalloc((void**)&DeviceTargetWord, (TargetWordLength + 1) * sizeof(char)));
	CUDACHECK(cudaMalloc((void**)&DeviceTransformations, (SourceWordLength + 1) * (TargetWordLength + 1) * sizeof(char)));

	CUDACHECK(cudaMalloc((void**)&DeviceX, AlphabetLength * (TargetWordLength + 1) * sizeof(int)));
	CUDACHECK(cudaMalloc((void**)&DeviceDistances, (SourceWordLength + 1) * (TargetWordLength + 1) * sizeof(int)));
	CUDACHECK(cudaMalloc((void**)&DeviceNextColumn, sizeof(int)));
}

DeviceRawData::~DeviceRawData()
{
	CUDACHECK(cudaFree(DeviceAlphabet));
	CUDACHECK(cudaFree(DeviceTransformations));
	CUDACHECK(cudaFree(DeviceDistances));
	CUDACHECK(cudaFree(DeviceX));
	CUDACHECK(cudaFree(DeviceTargetWord));
	CUDACHECK(cudaFree(DeviceSourceWord));
	CUDACHECK(cudaFree(DeviceNextColumn));
}

void DeviceRawData::FromHost(const std::string& sourceWord, const std::string& targetWord, int* hostNextColumn)
{
	CUDACHECK(cudaMemcpy(DeviceAlphabet, Alphabet, (AlphabetLength + 1) * sizeof(char), cudaMemcpyHostToDevice));
	CUDACHECK(cudaMemcpy(DeviceSourceWord, sourceWord.c_str(), (SourceWordLength + 1) * sizeof(char), cudaMemcpyHostToDevice));
	CUDACHECK(cudaMemcpy(DeviceTargetWord, targetWord.c_str(), (TargetWordLength + 1) * sizeof(char), cudaMemcpyHostToDevice));

	CUDACHECK(cudaMemcpy(DeviceNextColumn, hostNextColumn, sizeof(int), cudaMemcpyHostToDevice));
}

void DeviceRawData::ToHost(int** hostDistances, char** hostTransformations)
{
	CUDACHECK(cudaMemcpy(
		*hostDistances,
		DeviceDistances,
		(SourceWordLength + 1) * (TargetWordLength + 1) * sizeof(int),
		cudaMemcpyDeviceToHost
	));

	CUDACHECK(cudaMemcpy(
		*hostTransformations,
		DeviceTransformations,
		(SourceWordLength + 1) * (TargetWordLength + 1) * sizeof(char),
		cudaMemcpyDeviceToHost
	));
}