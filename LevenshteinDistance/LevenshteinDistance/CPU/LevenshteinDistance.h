#pragma once

#include "../LevenshteinDistanceBase.h"

#include <algorithm>

namespace CPU {
	/// <summary>
	/// A class for calculating the Levenshtein Distance on the CPU.
	/// </summary>
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
		/// <summary>
		/// Resolves the transformation operation at a specific matrix cell.
		/// </summary>
		/// <param name="s">Cost of substitution.</param>
		/// <param name="i">Cost of insertion.</param>
		/// <param name="d">Cost of deletion.</param>
		/// <param name="transformation">A reference to a char that will hold the operation ('s', 'i', 'd', etc.).</param>
		/// <returns>The minimum cost among substitution, insertion, or deletion.</returns>
		int ResolveTransformation(int s, int i, int d, char& transformation);

		/// <summary>
		/// Populates the distance and transformation matrices dynamically.
		/// </summary>
		/// <param name="distances">The matrix storing the edit distances.</param>
		/// <param name="transformations">The matrix storing the edit operations.</param>
		/// <param name="sourceWord">The source string.</param>
		/// <param name="targetWord">The target string.</param>
		void PopulateDynamically(
			Matrix<int>& distances,
			Matrix<char>& transformations,
			const std::string& sourceWord,
			const std::string& targetWord);

		/// <summary>
		/// Retrieves the sequence of transformations from the transformation matrix.
		/// </summary>
		/// <param name="transformations">The matrix storing the edit operations.</param>
		/// <param name="m">The number of rows in the matrix (length of sourceWord + 1).</param>
		/// <param name="n">The number of columns in the matrix (length of targetWord + 1).</param>
		/// <returns>A string representing the sequence of transformations.</returns>
		std::string RetrieveTransformation(Matrix<char>& transformations, int m, int n);

		/// <summary>
		/// Prints a matrix for debugging or visualization.
		/// </summary>
		/// <typeparam name="T">The type of the matrix elements (e.g., int or char).</typeparam>
		/// <param name="matrix">The matrix to be printed.</param>
		/// <param name="sourceWord">The source string used for row headers.</param>
		/// <param name="targetWord">The target string used for column headers.</param>
		template<typename T>
		void PrintMatrix(const Matrix<T>& matrix, const std::string& sourceWord, const std::string& targetWord) 
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
						std::cout << std::setw(3) << matrix[rowIndex][colIndex];
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