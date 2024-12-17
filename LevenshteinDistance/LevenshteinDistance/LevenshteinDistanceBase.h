#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

#define DELETE 'd'		// Represents a deletion operation
#define INSERT 'i'		// Represents an insertion operation
#define SUBSTITUTE 's'  // Represents a substitution operation
#define SKIP '-'		// Represents no operation (when characters match)

/// <summary>
/// Template alias for a 2D matrix using std::vector
/// </summary>
template<typename T>
using Matrix = std::vector<std::vector<T>>;

/// <summary>
/// Base class for calculating the Levenshtein Distance.
/// </summary>
class LevenshteinDistanceBase {
public:
	/// <summary>
	/// Function to calculate the Levenshtein Distance
	/// </summary>
	/// <param name="sourceWord">The starting string (source) to be transformed.</param>
	/// <param name="targetWord">The desired string (target) to transform into.</param>
	/// <param name="transformation">A string that will store the sequence of operations (INSERT, DELETE, SUBSTITUTE, or SKIP).</param>
	/// <param name="showTables">Optional flag to display intermediate results. Default is false.</param>
	/// <returns></returns>
	virtual int CalculateLevenshteinDistance(
		const std::string& sourceWord, 
		const std::string& targetWord, 
		std::string& transformation,
		bool showTables = false
	) = 0;
};