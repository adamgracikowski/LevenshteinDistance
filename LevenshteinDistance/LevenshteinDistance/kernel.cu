
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

struct ProgramParameters {
    std::string DataFormat{};
    std::string ComputationMethod{};
    std::string InputFile{};
    std::string OutputFile{};
    bool Success{};
};

ProgramParameters ParseProgramParameters(int argc, char* argv[]) {
    ProgramParameters parameters{};

    if (argc != 5) {
        std::cerr << "Usage: LevenshteinDistance data_format computation_method input_file output_file" << std::endl;
        return parameters;
    }

    parameters.DataFormat = argv[1];
    parameters.ComputationMethod = argv[2];
    parameters.InputFile = argv[3];
    parameters.OutputFile = argv[4];

    if (parameters.DataFormat != "txt" && parameters.DataFormat != "bin") {
        std::cerr << "Invalid data format. Use 'txt' or 'bin'." << std::endl;
        return parameters;
    }

    if (parameters.ComputationMethod != "cpu" && parameters.ComputationMethod != "gpu") {
        std::cerr << "Invalid computation method. Use 'cpu' or 'gpu'." << std::endl;
    }

    parameters.Success = true;
    return parameters;
}

std::pair<std::vector<char>, std::vector<char>> LoadDataFromBinaryFile(const std::string& path)
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

    std::vector<char> sourceWord(n);
    file.read(sourceWord.data(), n);

    if (!file) {
        throw std::runtime_error("Error while reading the source word.");
    }

    std::vector<char> targetWord(m);
    file.read(targetWord.data(), m);

    if (!file) {
        throw std::runtime_error("Error while reading the target word.");
    }

    return { sourceWord, targetWord };
}

std::pair<std::vector<char>, std::vector<char>> LoadDataFromTextFile(const std::string& path)
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

    std::vector<char> sourceWord(line.begin(), line.end());

    if (!std::getline(file, line)) {
        throw std::runtime_error("Error while reading the target word.");
    }

    std::vector<char> targetWord(line.begin(), line.end());

    return { sourceWord, targetWord };
}

std::pair<std::vector<char>, std::vector<char>> LoadDataFromInputFile(const std::string& dataFormat, const std::string& inputFile) 
{
    std::cout << "Loading data from the input file..." << std::endl;

    if (dataFormat == "txt") {
        return LoadDataFromTextFile(inputFile);
    }
    else if (dataFormat == "bin") {
        return LoadDataFromBinaryFile(inputFile);
    }
    else {
        throw std::runtime_error("Invalid format: " + dataFormat);
    }
}

int main(int argc, char* argv[])
{
    auto parameters = ParseProgramParameters(argc, argv);

    if (!parameters.Success) {
        return 1;
    }

    try {
        auto loaded = LoadDataFromInputFile(parameters.DataFormat, parameters.InputFile);
        auto& sourceWord = loaded.first;
        auto& targetWord = loaded.second;

        std::string source(sourceWord.begin(), sourceWord.end());
        std::string target(targetWord.begin(), targetWord.end());

        std::cout << source << std::endl;
        std::cout << target << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}