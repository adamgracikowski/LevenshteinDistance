#pragma once

#include "../LevenshteinDistanceBase.h"

namespace GPU {
	class LevenshteinDistance
		: public LevenshteinDistanceBase
	{
	public:
		int CalculateLevenshteinDistance(
			const std::string& sourceWord,
			const std::string& targetWord,
			std::string& transformation
		) override;
	};
}