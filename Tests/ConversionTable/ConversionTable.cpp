#include <fstream>
#include <iostream>

#include <csConversionTable.hpp>

int main() {
  csConversionTable table;

  double overSqrt2 = 1.0 / std::sqrt(2.0);

  for (double ls = -0.5; ls <= 0.5; ls += 0.1) {
    std::cout << table.getFillingFraction(ls, overSqrt2, overSqrt2)
              << std::endl;
  }

  // constexpr unsigned numberOfValues = 3;

  // csConversionTableFactory<numberOfValues, numberOfValues, numberOfValues>
  // table;

  // std::ofstream outFile;
  // outFile.open ("table.hpp");

  // outFile << "double table";
  // outFile << "[" << numberOfValues + 1 << "]";
  // outFile << "[" << numberOfValues + 1 << "]";
  // outFile << "[" << numberOfValues + 1 << "]";
  // outFile << " = { " << std::endl;
  // for(unsigned i = 0; i < numberOfValues + 1; ++i) {
  //   outFile << "{ ";
  //   for(unsigned j = 0; j < numberOfValues + 1; ++j) {
  //     outFile << "{ ";
  //     for(unsigned k = 0; k < numberOfValues + 1; ++k) {
  //       outFile << table.table[i][j][k];
  //       if(k != numberOfValues) {
  //         outFile << ", ";
  //       } else {
  //         outFile << "}" << std::endl;
  //       }
  //     }
  //     if(j != numberOfValues) {
  //       outFile << ", ";
  //     } else {
  //       outFile << "}" << std::endl;
  //     }
  //   }
  //   if(i != numberOfValues) {
  //     outFile << ", ";
  //   } else {
  //     outFile << "}" << std::endl;
  //   }
  // }
  // outFile << ";" << std::endl;

  // outFile.close();

  return 0;
}
