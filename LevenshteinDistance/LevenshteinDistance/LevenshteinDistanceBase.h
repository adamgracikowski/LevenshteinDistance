#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

#define DELETE 'd'
#define INSERT 'i'
#define SUBSTITUTE 's'
#define SKIP '-'

const char Alphabet[] = "abcdefghijklmnopqrstuvwxyz";
constexpr int AlphabetLength = sizeof(Alphabet) - 1;

template<typename T>
using Matrix = std::vector<std::vector<T>>;

class LevenshteinDistanceBase {
public:
	virtual int CalculateLevenshteinDistance(
		const std::string& sourceWord, 
		const std::string& targetWord, 
		std::string& transformation,
		bool showTables = false
	) = 0;
};