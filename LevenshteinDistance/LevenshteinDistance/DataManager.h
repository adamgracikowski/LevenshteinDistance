#pragma once

#include <vector>
#include <string>

/// <summary>
/// Provides functionality to load and save data.
/// </summary>
class DataManager
{
public:
    std::pair<std::string, std::string> LoadDataFromBinaryFile(const std::string& path);

    std::pair<std::string, std::string> LoadDataFromTextFile(const std::string& path);

    std::pair<std::string, std::string> LoadDataFromInputFile(const std::string& dataFormat, const std::string& inputFile);

    void SaveDataToOutputFile(const std::string& path, const std::string& dataFormat, const std::string& transformation, const int editDistance);
    
    void SaveDataToBinaryFile(const std::string& path, const std::string& transformation, const int editDistance);
    
    void SaveDataToTextFile(const std::string& path, const std::string& transformation, const int editDistance);
};