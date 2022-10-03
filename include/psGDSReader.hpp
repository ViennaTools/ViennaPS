#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <psDomain.hpp>
#include <psGDSUtils.hpp>

#ifndef endian_swap_long
#define endian_swap_long(w)                                                    \
  (((w & 0xff) << 24) | ((w & 0xff00) << 8) | ((w & 0xff0000) >> 8) |          \
   ((w & 0xff000000) >> 24))
#endif
#ifndef endian_swap_short
#define endian_swap_short(w) (((w & 0xff) << 8) | ((w & 0xff00) >> 8))
#endif

template <typename NumericType, int D> class psGDSReader {
  using PSPtrType = psSmartPointer<psDomain<NumericType, D>>;

  FILE *filePtr = nullptr;
  PSPtrType domain = nullptr;
  std::string fileName;

public:
  psGDSReader() {}
  psGDSReader(PSPtrType passedDomain, std::string passedFileName)
      : domain(passedDomain), fileName(passedFileName) {}

  void setDomain(PSPtrType passedDomain) { domain = passedDomain; }

  void setFileName(std::string passedFileName) { fileName = passedFileName; }

  void apply() {
    filePtr = fopen(fileName.c_str(), "rb");
    if (!filePtr) {
      std::cerr << "Could not open GDS file." << std::endl;
      return;
    }

    unsigned char recordType, dataType;

    while (!feof(filePtr)) {
      currentrecordLen = getTwoByteSignedInt();
      fread(&recordType, 1, 1, filePtr);
      fread(&dataType, 1, 1, filePtr);
      currentrecordLen -= 4;

      switch (recordType) {
      case psGDSRecordNumbers::rnHeader:
        std::cout << "Found header\n";
        parseHeader();
        break;

      case psGDSRecordNumbers::rnBgnLib:
        std::cout << "BGNLIB\n";
        while (currentrecordLen) {
          getTwoByteSignedInt();
        }
        break;

      case psGDSRecordNumbers::rnLibName:
        std::cout << "LIBNAME ";
        parseLibName();
        break;

      default:
        break;
      }
    }
  }

private:
  int16_t currentrecordLen = 0;

  char *getAsciiString() {
    char *str = NULL;

    if (currentrecordLen > 0) {
      currentrecordLen +=
          currentrecordLen %
          2; /* Make sure length is even, why would you do this? */
      str = new char[currentrecordLen + 1];

      fread(str, 1, currentrecordLen, filePtr);
      str[currentrecordLen] = 0;
      currentrecordLen = 0;
    }

    return str;
  }

  int16_t getTwoByteSignedInt() {
    int16_t value;

    fread(&value, 2, 1, filePtr);

    currentrecordLen -= 2;

#if __BYTE_ORDER == __LITTLE_ENDIAN
    return endian_swap_short(value);
#else
    return value;
#endif
  }

  void parseHeader() {
    short version;
    version = getTwoByteSignedInt();
    std::cout << "Version " << version << std::endl;
  }

  void parseLibName() {
    char *str;
    str = GetAsciiString();
    if (_libname) {
      delete[] _libname;
      _libname = NULL;
    }
    _libname = new char[strlen(str) + 1];
    if (_libname) {
      strcpy(_libname, str);
      v_printf(3, " (\"%s\")\n", _libname);
    } else {
      v_printf(1, "\nUnable to allocate memory for string (%d)\n",
               strlen(str) + 1);
    }
    delete[] str;
  }
};