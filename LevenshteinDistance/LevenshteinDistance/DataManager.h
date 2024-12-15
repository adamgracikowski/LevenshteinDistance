#pragma once

#include <vector>
#include <string>

class DataManager
{
public:
    std::pair<std::vector<char>, std::vector<char>> LoadDataFromBinaryFile(const std::string& path);

    std::pair<std::vector<char>, std::vector<char>> LoadDataFromTextFile(const std::string& path);

    std::pair<std::vector<char>, std::vector<char>> LoadDataFromInputFile(const std::string& dataFormat, const std::string& inputFile);

    void SaveDataToOutputFile(const std::string& path, const std::string& dataFormat, const std::string& transformation);
    
    void SaveDataToBinaryFile(const std::string& path, const std::string& transformation);
    
    void SaveDataToTextFile(const std::string& path, const std::string& transformation);
};