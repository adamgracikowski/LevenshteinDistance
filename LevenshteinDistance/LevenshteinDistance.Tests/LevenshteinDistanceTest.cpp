#include "pch.h"
#include "../LevenshteinDistance/LevenshteinDistance.cpp"

class LevenshteinDistanceTest : public ::testing::Test {
protected:
    CPU::LevenshteinDistance lev;

    void RunTest(
        const std::string& source, 
        const std::string& target,
        const std::string& expectedTransformation, 
        int expectedDistance) 
    {
        std::string actualTransformation;
        int actualDistance = lev.CalculateLevenshteinDistance(source, target, actualTransformation);

        ASSERT_EQ(expectedDistance, actualDistance);
        ASSERT_EQ(expectedTransformation, actualTransformation);
    }
};

TEST_F(LevenshteinDistanceTest, EmptyWords) {
    RunTest("", "", "", 0);
}

TEST_F(LevenshteinDistanceTest, EmptySourceNonEmptyTarget) {
    RunTest("", "abc", "iii", 3);
}

TEST_F(LevenshteinDistanceTest, EmptyTargetNonEmptySource) {
    RunTest("abc", "", "ddd", 3);
}

TEST_F(LevenshteinDistanceTest, IdenticalWords) {
    RunTest("abc", "abc", "---", 0);
}