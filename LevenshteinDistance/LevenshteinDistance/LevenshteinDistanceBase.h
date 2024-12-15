#pragma once

#include <string>
#include <vector>

#define DELETE 'd'
#define INSERT 'i'
#define SUBSTITUTE 's'
#define SKIP '-'

template<typename T>
using Matrix = std::vector<std::vector<T>>;

class LevenshteinDistanceBase {
public:
	virtual int CalculateLevenshteinDistance(
		const std::string& sourceWord, 
		const std::string& targetWord, 
		std::string& transformation
	) = 0;
};