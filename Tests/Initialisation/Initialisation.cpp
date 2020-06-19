#include <iostream>

#include <psDomain.hpp>

class myCellType : public cellBase {};

int main() {

  psDomain<myCellType, float, 3> myDomain;

  std::cout << "success" << std::endl;

  myDomain.print();

  return 0;
}
