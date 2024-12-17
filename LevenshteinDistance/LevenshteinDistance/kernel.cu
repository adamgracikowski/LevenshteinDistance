
#include "ProgramParameters.h"
#include "DataManager.h"
#include "Timers/TimerManager.h"

#include "CPU/LevenshteinDistance.h"
#include "GPU/LevenshteinDistance.cuh"

#include <iomanip>

#define SHOW_TABLES_SIZE 20

void DisplayProgramParameters(ProgramParameters parameters);

void DisplayWords(const std::string& source, const std::string& target);

void DisplaySummary(const std::string& computationMethod);

void TransformWords(std::string& sourceWord, std::string& targetWord);

int main(int argc, char* argv[])
{
	auto parameters = ParseProgramParameters(argc, argv);

	// Check if the parameters are successfully parsed
	if (!parameters.Success) {
		return 1;
	}

	DisplayProgramParameters(parameters);

	try {
		auto dataManager = DataManager{};

		auto loaded = dataManager.LoadDataFromInputFile(parameters.DataFormat, parameters.InputFile);
		auto& sourceWord = loaded.first;
		auto& targetWord = loaded.second;

		// Transform both source and target words to lowercase before computation
		TransformWords(sourceWord, targetWord);

		DisplayWords(sourceWord, targetWord);

		// Check if the word lengths are small enough to display tables
		auto showTables = sourceWord.length() <= SHOW_TABLES_SIZE && targetWord.length() <= SHOW_TABLES_SIZE;

		std::string transformation{};

		if (parameters.ComputationMethod == CPU_COMPUTATION_METHOD) {
			auto lev = CPU::LevenshteinDistance{};
			lev.CalculateLevenshteinDistance(sourceWord, targetWord, transformation, showTables);
		}
		else {
			auto lev = GPU::LevenshteinDistance{};
			lev.CalculateLevenshteinDistance(sourceWord, targetWord, transformation, showTables);
		}

		std::cout << transformation << std::endl;

		dataManager.SaveDataToOutputFile(parameters.OutputFile, TXT_FORMAT, transformation);

		DisplaySummary(parameters.ComputationMethod);
	}
	catch (const std::exception& e) {
		// Handle any exceptions that occur during computation
		std::cerr << e.what() << std::endl;
	}

	return 0;
}

void DisplayProgramParameters(ProgramParameters parameters)
{
	std::cout << std::setw(25) << std::left << "Data format: "
		<< parameters.DataFormat << std::endl;
	std::cout << std::setw(25) << std::left << "Computation method: "
		<< parameters.ComputationMethod << std::endl << std::endl;
}

void DisplayWords(const std::string& source, const std::string& target)
{
	const auto width = 50;

	auto m = source.length();
	auto n = target.length();

	if (m < width) {
		std::cout << std::setw(25) << std::left << "Source word: "
			<< source << std::endl;
	}
	else {
		std::cout << std::setw(25) << std::left << "Source word length: "
			<< source.length() << std::endl;
	}

	if (n < width) {
		std::cout << std::setw(25) << std::left << "Target word: "
			<< target << std::endl;
	}
	else {
		std::cout << std::setw(25) << std::left << "Target word length: "
			<< target.length() << std::endl;
	}

	std::cout << std::endl;
}

void DisplaySummary(const std::string& computationMethod)
{
	const int width{ 40 };
	auto& timerManager = Timers::TimerManager::GetInstance();

	auto totalExecutionTime{ 0.0f };
	auto totalAlgorithmTime{ 0.0f };

	auto displayTime = [&](const std::string& title, const float elapsed)
		{
			totalExecutionTime += elapsed;
			std::cout << std::setw(width) << std::left << title << elapsed << " ms." << std::endl;
		};

	std::cout << std::endl;
	displayTime("Loading data from the input file: ", timerManager.LoadDataFromInputFileTimer.TotalElapsedMiliseconds());

	if (computationMethod == CPU_COMPUTATION_METHOD) {
		displayTime("Finding distance: ", timerManager.FindDistanceTimer.TotalElapsedMiliseconds());
		totalAlgorithmTime += timerManager.FindDistanceTimer.TotalElapsedMiliseconds();

		displayTime("Retrieving transformation: ", timerManager.RetrieveTransformationTimer.TotalElapsedMiliseconds());
		totalAlgorithmTime += timerManager.RetrieveTransformationTimer.TotalElapsedMiliseconds();
	}
	else {
		displayTime("Host to device transfer: ", timerManager.Host2DeviceDataTransferTimer.TotalElapsedMiliseconds());
		totalAlgorithmTime += timerManager.Host2DeviceDataTransferTimer.TotalElapsedMiliseconds();

		displayTime("Populating X: ", timerManager.PopulateDeviceXTimer.TotalElapsedMiliseconds());
		totalAlgorithmTime += timerManager.PopulateDeviceXTimer.TotalElapsedMiliseconds();

		displayTime("Populating distances: ", timerManager.PopulateDeviceDistancesTimer.TotalElapsedMiliseconds());
		totalAlgorithmTime += timerManager.PopulateDeviceDistancesTimer.TotalElapsedMiliseconds();

		displayTime("Device to host transfer: ", timerManager.Device2HostDataTransferTimer.TotalElapsedMiliseconds());
		totalAlgorithmTime += timerManager.Device2HostDataTransferTimer.TotalElapsedMiliseconds();
	}

	std::cout << std::setw(40) << std::left << "Saving data to output file time: "
		<< timerManager.SaveDataToOutputFileTimer.TotalElapsedMiliseconds() << " ms." << std::endl << std::endl;

	std::cout << std::setw(width) << std::left << "Total algorithm execution: " << totalExecutionTime << " ms." << std::endl;
	std::cout << std::setw(width) << std::left << "Total execution: " << totalExecutionTime << " ms." << std::endl;
}

void TransformWords(std::string& sourceWord, std::string& targetWord)
{
	std::transform(
		sourceWord.begin(),
		sourceWord.end(),
		sourceWord.begin(),
		[](unsigned char c) {
			return tolower(c);
		}
	);

	std::transform(
		targetWord.begin(),
		targetWord.end(),
		targetWord.begin(),
		[](unsigned char c) {
			return tolower(c);
		}
	);
}