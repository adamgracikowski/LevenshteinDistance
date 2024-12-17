#pragma once

#include "../CudaCheck.cuh"

#include <string>

const char Alphabet[] = "abcdefghijklmnopqrstuvwxyz";
constexpr int AlphabetLength = sizeof(Alphabet) - 1;

struct DeviceRawData
{
public:
	char* DeviceAlphabet{};
	char* DeviceSourceWord{};
	unsigned SourceWordLength{};
	char* DeviceTargetWord{};
	unsigned TargetWordLength{};
	int* DeviceX{};
	int* DeviceDistances{};
	char* DeviceTransformations{};
	int* DeviceNextColumn{};

	DeviceRawData(unsigned sourceWordLength, unsigned targetWordLength);

	~DeviceRawData();

	void FromHost(const std::string& sourceWord, const std::string& targetWord, int* hostNextColumn);

	void ToHost(int** hostDistances, char** hostTransformations);
};