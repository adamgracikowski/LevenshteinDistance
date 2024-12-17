#include "DataManager.h"
#include "Timers/TimerManager.h"

#include <iostream>
#include <fstream>
#include <sstream>

std::pair<std::string, std::string> DataManager::LoadDataFromBinaryFile(const std::string& path)
{
	std::ifstream file(path, std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("Could not open: " + path);
	}

	int n = 0, m = 0;

	file.read(reinterpret_cast<char*>(&n), sizeof(n));
	file.read(reinterpret_cast<char*>(&m), sizeof(m));

	if (!file) {
		throw std::runtime_error("Error while reading the values of 'n' and 'm' parameters.");
	}

	std::string sourceWord(n, ' ');
	file.read(sourceWord.data(), n);

	if (!file) {
		throw std::runtime_error("Error while reading the source word.");
	}

	std::string targetWord(m, ' ');
	file.read(targetWord.data(), m);

	if (!file) {
		throw std::runtime_error("Error while reading the target word.");
	}

	return { sourceWord, targetWord };
}

std::pair<std::string, std::string> DataManager::LoadDataFromTextFile(const std::string& path)
{
	std::ifstream file(path);

	if (!file.is_open()) {
		throw std::runtime_error("Could not open: " + path);
	}

	int n = 0, m = 0;

	std::string line;

	if (!std::getline(file, line)) {
		throw std::runtime_error("Error while reading the values of 'n' and 'm' parameters.");
	}

	std::istringstream iss(line);
	if (!(iss >> n >> m)) {
		throw std::runtime_error("Error while parsing the values of 'n' and 'm' parameters.");
	}

	if (!std::getline(file, line)) {
		throw std::runtime_error("Error while reading the source word.");
	}

	std::string sourceWord(line.begin(), line.end());

	if (!std::getline(file, line)) {
		throw std::runtime_error("Error while reading the target word.");
	}

	std::string targetWord(line.begin(), line.end());

	return { sourceWord, targetWord };
}

std::pair<std::string, std::string> DataManager::LoadDataFromInputFile(const std::string& dataFormat, const std::string& inputFile)
{
	auto& timerManager = Timers::TimerManager::GetInstance();

	std::cout << "Loading data from the input file..." << std::endl << std::endl;

	timerManager.LoadDataFromInputFileTimer.Start();

	if (dataFormat == "txt") {
		return LoadDataFromTextFile(inputFile);
	}
	else if (dataFormat == "bin") {
		return LoadDataFromBinaryFile(inputFile);
	}
	else {
		timerManager.LoadDataFromInputFileTimer.Stop();

		throw std::runtime_error("Invalid format: " + dataFormat);
	}

	timerManager.LoadDataFromInputFileTimer.Stop();
}

void DataManager::SaveDataToOutputFile(const std::string& outputFile, const std::string& dataFormat, const std::string& transformation)
{
	auto& timerManager = Timers::TimerManager::GetInstance();

	std::cout << std::endl << "Saving results to the output file..." << std::endl;

	timerManager.SaveDataToOutputFileTimer.Start();

	if (dataFormat == "txt") {
		return SaveDataToTextFile(outputFile, transformation);
	}
	else if (dataFormat == "bin") {
		return SaveDataToBinaryFile(outputFile, transformation);
	}
	else {
		timerManager.SaveDataToOutputFileTimer.Stop();

		throw std::runtime_error("Invalid format: " + dataFormat);
	}

	timerManager.SaveDataToOutputFileTimer.Stop();
}

void DataManager::SaveDataToBinaryFile(const std::string& outputFile, const std::string& transformation)
{
	std::ofstream file(outputFile, std::ios::binary);

	if (!file.is_open()) {
		throw std::runtime_error("Could not open: " + outputFile);
	}

	size_t length = transformation.size();

	file.write(reinterpret_cast<const char*>(&length), sizeof(length));
	file.write(transformation.data(), length);
}

void DataManager::SaveDataToTextFile(const std::string& outputFile, const std::string& transformation)
{
	std::ofstream file(outputFile);

	if (!file.is_open()) {
		throw std::runtime_error("Could not open: " + outputFile);
	}

	file << transformation.size() << std::endl;
	file << transformation << std::endl;
}