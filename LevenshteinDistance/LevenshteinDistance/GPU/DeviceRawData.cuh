#pragma once

#include "../CudaCheck.cuh"

#include <string>

/// <summary>
/// Global constant defining the alphabet used in processing.
/// </summary>
const char Alphabet[] = "abcdefghijklmnopqrstuvwxyz";

/// <summary>
/// Length of the Alphabet array.
/// </summary>
constexpr int AlphabetLength = sizeof(Alphabet) - 1;

/// <summary>
/// A structure for managing raw device (GPU) data.
/// </summary>
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

	/// <summary>
	/// Constructor to allocate GPU memory for the required data.
	/// 
	/// This constructor initializes the GPU memory for storing source and target words, their lengths, and associated matrices.
	/// </summary>
	/// <param name="sourceWordLength">The length of the source word.</param>
	/// <param name="targetWordLength">The length of the target word.</param>
	DeviceRawData(unsigned sourceWordLength, unsigned targetWordLength);

	~DeviceRawData();

	/// <summary>
	/// Transfers data from host (CPU) to device (GPU).
	/// </summary>
	void FromHost(const std::string& sourceWord, const std::string& targetWord, int* hostNextColumn);

	/// <summary>
	/// Transfers data from device (GPU) to host (CPU).
	/// </summary>
	/// <param name="hostDistances">Double pointer to store the distances matrix in host memory.</param>
	/// <param name="hostTransformations">Double pointer to store the transformations matrix in host memory.</param>
	void ToHost(int** hostDistances, char** hostTransformations);
};