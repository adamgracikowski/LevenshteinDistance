#pragma once

#include <vector>
#include <string>

class DataManager
{
public:
    std::pair<std::string, std::string> LoadDataFromBinaryFile(const std::string& path);

    std::pair<std::string, std::string> LoadDataFromTextFile(const std::string& path);

    std::pair<std::string, std::string> LoadDataFromInputFile(const std::string& dataFormat, const std::string& inputFile);

    void SaveDataToOutputFile(const std::string& path, const std::string& dataFormat, const std::string& transformation);
    
    void SaveDataToBinaryFile(const std::string& path, const std::string& transformation);
    
    void SaveDataToTextFile(const std::string& path, const std::string& transformation);
};