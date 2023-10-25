#include "Application.hpp"

int main(int argc, char **argv) {

  Application<VIENNAPS_APP_DIM> app(argc, argv);
  app.run();

  return 0;
}