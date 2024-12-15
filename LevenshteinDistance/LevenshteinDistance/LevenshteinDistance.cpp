#include "LevenshteinDistance.h"

#define DEBUG

int CPU::LevenshteinDistance::CalculateLevenshteinDistance(
	const std::string& sourceWord, 
	const std::string& targetWord, 
	std::string& transformation)
{
	auto m = sourceWord.length();
	auto n = targetWord.length();

	auto distances = Matrix<int>(m + 1, std::vector<int>(n + 1, 0));
	auto transformations = Matrix<char>(m + 1, std::vector<char>(n + 1, 0));

	PopulateDynamically(distances, transformations, sourceWord, targetWord);

	transformation = RetrieveTransformation(transformations, m, n);

	return distances[m][n];
}

void CPU::LevenshteinDistance::PopulateDynamically(
	Matrix<int>& distances,
	Matrix<char>& transformations,
	const std::string& sourceWord, 
	const std::string& targetWord)
{
	auto m{ sourceWord.length() };
	auto n{ targetWord.length() };

	for (auto i{ 0 }; i <= m; ++i)
	{
		distances[i][0] = i;
		transformations[i][0] = DELETE;
	}
	for (auto i{ 1 }; i <= n; ++i)
	{
		distances[0][i] = i;
		transformations[0][i] = INSERT;
	}

	char currentTransformation{};

	for (auto j{ 1 }; j <= n; j++)
	{
		for (auto i{ 1 }; i <= m; i++)
		{
			auto isDifferent = sourceWord[i - 1] != targetWord[j - 1] ? 1 : 0;

			distances[i][j] = ResolveTransformation(
				distances[i - 1][j - 1] + isDifferent, // substitution
				distances[i][j - 1] + 1,			   // insertion
				distances[i - 1][j] + 1,               // deletion
				currentTransformation
			);

			if (currentTransformation == SUBSTITUTE && !isDifferent) {
				currentTransformation = SKIP;
			}

			transformations[i][j] = currentTransformation;
		}
	}
#ifdef DEBUG
	PrintMatrix<int>(distances);
	PrintMatrix<char>(transformations);
#endif // DEBUG

}

int CPU::LevenshteinDistance::ResolveTransformation(int s, int i, int d, char& transformation)
{
	auto result{ s };

	transformation = SUBSTITUTE;

	if (i < result)
	{
		result = i;
		transformation = INSERT;
	}

	if (d < result)
	{
		result = d;
		transformation = DELETE;
	}

	return result;
}

std::string CPU::LevenshteinDistance::RetrieveTransformation(Matrix<char>& transformations,	int m, int n)
{
	std::string transformation{};

	int i{ m }, j{ n };

	while (i != 0 || j != 0)
	{
		transformation.push_back(transformations[i][j]);

		if (transformations[i][j] == DELETE) {
			i--;
		}
		else if (transformations[i][j] == INSERT) {
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