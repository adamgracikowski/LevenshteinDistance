#pragma once

#include "LevenshteinDistanceBase.h"

#include <algorithm>
#include <iostream>

namespace CPU {
	class LevenshteinDistance
		: public LevenshteinDistanceBase
	{
	public:
		int CalculateLevenshteinDistance(
			const std::string& sourceWord,
			const std::string& targetWord,
			std::string& transformation
		) override;

	private:

		int ResolveTransformation(int s, int i, int d, char& transformation);

		void PopulateDynamically(
			Matrix<int>& distances,
			Matrix<char>& transformations,
			const std::string& sourceWord,
			const std::string& targetWord);

		std::string RetrieveTransformation(Matrix<char>& transformations, int m, int n);

		template<typename T>
		void PrintMatrix(const Matrix<T>& matrix) {
			for (const auto& row : matrix) {
				for (const auto& cell : row) {
					std::cout << cell << ' ';
				}
				std::cout << '\n';
			}
		}
	};
}