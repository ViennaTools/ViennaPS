#ifndef PS_POINT_DATA_HPP
#define PS_POINT_DATA_HPP

#include <lsPointData.hpp>

template <typename T> using psPointData = lsPointData<T>;

// template <typename NumericType>
// class psCellData
// {
// private:
//     using CellDataType = std::vector<std::vector<NumericType>>;

//     std::vector<std::string> labels;
//     CellDataType data;

// public:
//     void insterNextCellData(std::vector<NumericType> &passedData, std::string
//     dataLabel)
//     {
//         data.push_back(passedData);
//         labels.push_back(dataLabel);
//     }

//     void insterNextCellData(std::vector<NumericType> &&passedData,
//     std::string dataLabel)
//     {
//         data.push_back(std::move(passedData));
//         labels.push_back(dataLabel);
//     }

//     CellDataType &getCellData()
//     {
//         return data;
//     }

//     std::vector<NumericType> *getCellData(std::string label)
//     {
//         if (auto it = std::find(labels.begin(), labels.end(), label); it !=
//         labels.end())
//         {
//             auto idx = it - labels.begin();
//             return &data[idx];
//         }
//         return nullptr;
//     }
// };

#endif