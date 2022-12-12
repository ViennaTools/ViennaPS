#include "Application.hpp"

int main(int argc, char **argv) {

  Application<3> app(argc, argv);
  app.run();

  return 0;
}