#pragma once

#include "../LevenshteinDistanceBase.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define THREADS_IN_ONE_BLOCK 512u
#define WARP_SIZE 32

const int AlphabetLength = 26;

namespace GPU
{
	__global__
		void PopulateDeviceX(
			int* deviceX,
			char* deviceAlphabet,
			int alphabetLength,
			char* deviceTargetWord,
			int targetWordLength
		);

	__global__
		void PopulateDeviceDistances(
			int* deviceDistances,
			char* deviceTransformations,
			int* deviceX,
			char* deviceSourceWord,
			int sourceWordLength,
			char* deviceTargetWord,
			int targetWordLength,
			int warpsCount,
			int* deviceNextColumn
		);

	__device__
		int ResolveTransformation(int s, int i, int d, char* transformation);

	class LevenshteinDistance
		: public LevenshteinDistanceBase
	{
	public:
		int CalculateLevenshteinDistance(
			const std::string& sourceWord,
			const std::string& targetWord,
			std::string& transformation
		) override;

		const char Alphabet[AlphabetLength + 1] = { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
						 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
						 'w', 'x', 'y', 'z', '\0' };
		
		char* DeviceAlphabet{};
		char* DeviceSourceWord{};
		unsigned SourceWordLength{};
		char* DeviceTargetWord{};
		unsigned TargetWordLength{};
		int* DeviceX{};
		int* DeviceDistances{};
		char* DeviceTransformations{};
		int* DeviceNextColumn{};

	private:
		std::string RetrieveTransformation(char* transformations, int m, int n);
	};
}