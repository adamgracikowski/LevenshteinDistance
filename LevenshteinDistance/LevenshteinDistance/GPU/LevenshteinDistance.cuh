#pragma once

#include "../LevenshteinDistanceBase.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define THREADS_IN_ONE_BLOCK 512u
#define WARP_SIZE 32

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
			std::string& transformation,
			bool showTables = false
		) override;

	private:
		std::string RetrieveTransformation(char* transformations, int m, int n);

		template<typename T>
		void PrintMatrix(T* matrix, const std::string& sourceWord, const std::string& targetWord)
		{
			const int width = 3;
			const size_t rows = sourceWord.length() + 1;
			const size_t columns = targetWord.length() + 1;

			auto PrintColumnHeaders = [&]()
				{
					std::cout << "      ";

					for (size_t j = 0; j < targetWord.length(); ++j) {
						std::cout << std::setw(width) << targetWord[j];
					}
					std::cout << std::endl;
				};

			auto PrintRow = [&](size_t rowIndex)
				{
					if (rowIndex == 0) {
						std::cout << "   ";
					}
					else {
						std::cout << std::setw(3) << sourceWord[rowIndex - 1];
					}

					for (size_t colIndex = 0; colIndex < columns; ++colIndex) {
						std::cout << std::setw(3) << matrix[rowIndex * columns + colIndex];
					}

					std::cout << std::endl;
				};

			PrintColumnHeaders();

			for (size_t i = 0; i < rows; ++i) {
				PrintRow(i);
			}

			std::cout << std::endl;
		}
	};
}