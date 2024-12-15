
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ProgramParameters.h"
#include "DataManager.h"
#include "CPU/LevenshteinDistance.h"
#include "GPU/LevenshteinDistance.cuh"

#include <iomanip>

void DisplayProgramParameters(ProgramParameters parameters) {
	std::cout << std::setw(25) << std::left << "Data format: "
		<< parameters.DataFormat << std::endl;
	std::cout << std::setw(25) << std::left << "Computation method: "
		<< parameters.ComputationMethod << std::endl << std::endl;
}

void DisplayWords(const std::string& source, const std::string& target) {
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

int main(int argc, char* argv[])
{
	auto parameters = ParseProgramParameters(argc, argv);

	if (!parameters.Success) {
		return 1;
	}

	DisplayProgramParameters(parameters);

	try {
		auto dataManager = DataManager{};

		auto loaded = dataManager.LoadDataFromInputFile(parameters.DataFormat, parameters.InputFile);
		auto& sourceWord = loaded.first;
		auto& targetWord = loaded.second;

		std::string source(sourceWord.begin(), sourceWord.end());
		std::string target(targetWord.begin(), targetWord.end());

		DisplayWords(source, target);

		std::string transformation{};

		if (parameters.ComputationMethod == CPU_COMPUTATION_METHOD) {
			auto lev = CPU::LevenshteinDistance{};
			lev.CalculateLevenshteinDistance(source, target, transformation);
		}
		else {
			auto lev = GPU::LevenshteinDistance{};
			lev.CalculateLevenshteinDistance(source, target, transformation);
		}

		dataManager.SaveDataToOutputFile(parameters.OutputFile, TXT_FORMAT, transformation);
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
	}

	return 0;
}