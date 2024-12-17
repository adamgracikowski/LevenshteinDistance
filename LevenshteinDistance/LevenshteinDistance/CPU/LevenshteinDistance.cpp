#include "LevenshteinDistance.h"
#include "../Timers/TimerManager.h"

int CPU::LevenshteinDistance::CalculateLevenshteinDistance(
	const std::string& sourceWord, 
	const std::string& targetWord, 
	std::string& transformation,
	bool showTables)
{
	auto& timerManager = Timers::TimerManager::GetInstance();

	auto m = static_cast<int>(sourceWord.length());
	auto n = static_cast<int>(targetWord.length());

	auto distances = Matrix<int>(m + 1, std::vector<int>(n + 1, 0));
	auto transformations = Matrix<char>(m + 1, std::vector<char>(n + 1, 0));

	std::cout << "Starting computation..." << std::endl;

	std::cout << "-> Finding distance..." << std::endl;

	timerManager.FindDistanceTimer.Start();
	PopulateDynamically(distances, transformations, sourceWord, targetWord);
	timerManager.FindDistanceTimer.Stop();

	std::cout << std::setw(35) << std::left << "    Elapsed time: "
		<< timerManager.FindDistanceTimer.ElapsedMiliseconds() << " ms" << std::endl;

	std::cout << " -> Retrieving transformation..." << std::endl;

	timerManager.RetrieveTransformationTimer.Start();
	transformation = RetrieveTransformation(transformations, m, n);
	timerManager.RetrieveTransformationTimer.Stop();

	std::cout << std::setw(35) << std::left << "    Elapsed time: "
		<< timerManager.RetrieveTransformationTimer.ElapsedMiliseconds() << " ms" << std::endl;

	if (showTables) {
		std::cout << std::endl << "Distances:" << std::endl << std::endl;
		PrintMatrix(distances, sourceWord, targetWord);

		std::cout << std::endl << "Transformations:" << std::endl << std::endl;
		PrintMatrix(transformations, sourceWord, targetWord);

		std::cout << std::endl;
	}

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