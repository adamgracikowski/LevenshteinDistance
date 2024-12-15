#pragma once

#include <string>
#include <iostream>

#define BIN_FORMAT "bin"
#define TXT_FORMAT "txt"
#define CPU_COMPUTATION_METHOD "cpu"
#define GPU_COMPUTATION_METHOD "gpu"

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

	if (parameters.DataFormat != TXT_FORMAT && 
		parameters.DataFormat != BIN_FORMAT) {
		std::cerr << "Invalid data format. Use 'txt' or 'bin'." << std::endl;
		return parameters;
	}

	if (parameters.ComputationMethod != CPU_COMPUTATION_METHOD && 
		parameters.ComputationMethod != GPU_COMPUTATION_METHOD) {
		std::cerr << "Invalid computation method. Use 'cpu' or 'gpu'." << std::endl;
		return parameters;
	}

	parameters.Success = true;
	return parameters;
}