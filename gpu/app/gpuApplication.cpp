#include "gpuApplication.hpp"
#include "interrupt.hpp"

int main(int argc, char **argv) {

  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = sig_to_exception;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);

  gpuApplication app(argc, argv);
  try {
    app.run();
  } catch (InterruptException &e) {
    std::cerr << "Aborting. Saving last geometry." << '\n';
    app.printGeometry("geometryOnAbort");
    return 1;
  }

  return 0;
}