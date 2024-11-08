#pragma once

#include <exception>
#include <signal.h>
#include <stdlib.h>

class InterruptException : public std::exception {
public:
  InterruptException(int s) : S(s) {}
  int S;
};

void sig_to_exception(int s) { throw InterruptException(s); }