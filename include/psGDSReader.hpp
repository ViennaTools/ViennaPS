#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <psDomain.hpp>
#include <psGDSGeometry.hpp>
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
  using PSPtrType = psSmartPointer<psGDSGeometry<NumericType, D>>;

  FILE *filePtr = nullptr;
  PSPtrType geometry = nullptr;
  std::string fileName;

public:
  psGDSReader() {}
  psGDSReader(PSPtrType passedGeometry, std::string passedFileName)
      : geometry(passedGeometry), fileName(passedFileName) {}

  void setgeometry(PSPtrType passedGeometry) { geometry = passedGeometry; }

  void setFileName(std::string passedFileName) { fileName = passedFileName; }

  void apply() {
    parseFile();
    geometry->checkReferences();
    geometry->calculateBoundingBoxes();
    geometry->preBuildStructures();
  }

private:
  psGDSStructure<NumericType> currentStructure;

  int16_t currentRecordLen = 0;
  int16_t currentLayer;
  int16_t currentDataType;
  int16_t currentPlexNumber;
  int16_t currentSTrans;
  double currentMag = 0;
  double currentAngle = 0;
  int16_t arrayCols, arrayRows;
  bool ignore = false;
  psGDSElementType currentElement;
  double units; // units in micron
  double userUnits;

  // unsused
  float currentWidth;
  char *tempStr;

  void resetCurrentStructure() {
    currentStructure.name = "";
    currentStructure.elements.clear();
    currentStructure.sRefs.clear();
    currentStructure.aRefs.clear();
    currentStructure.containsLayers.clear();
    currentStructure.boundaryElements = 0;
    currentStructure.boxElements = 0;

    currentStructure.elementBoundingBox[0][0] =
        std::numeric_limits<NumericType>::max();
    currentStructure.elementBoundingBox[0][1] =
        std::numeric_limits<NumericType>::max();

    currentStructure.elementBoundingBox[1][0] =
        std::numeric_limits<NumericType>::lowest();
    currentStructure.elementBoundingBox[1][1] =
        std::numeric_limits<NumericType>::lowest();
  }

  char *readAsciiString() {
    char *str = NULL;

    if (currentRecordLen > 0) {
      currentRecordLen += currentRecordLen % 2;
      str = new char[currentRecordLen + 1];

      (void)!fread(str, 1, currentRecordLen, filePtr);
      str[currentRecordLen] = 0;
      currentRecordLen = 0;
    }

    return str;
  }

  int16_t readTwoByteSignedInt() {
    int16_t value;
    (void)!fread(&value, 2, 1, filePtr);

    currentRecordLen -= 2;

#if __BYTE_ORDER == __LITTLE_ENDIAN
    return endian_swap_short(value);
#else
    return value;
#endif
  }

  int32_t readFourByteSignedInt() {
    int32_t value;
    (void)!fread(&value, 4, 1, filePtr);

    currentRecordLen -= 4;

#if __BYTE_ORDER == __LITTLE_ENDIAN
    return endian_swap_long(value);
#else
    return value;
#endif
  }

  double readEightByteReal() {
    unsigned char value;
    unsigned char b8, b2, b3, b4, b5, b6, b7;
    double sign = 1.0;
    double exponent;
    double mant;

    (void)!fread(&value, 1, 1, filePtr);
    if (value & 128) {
      value -= 128;
      sign = -1.0;
    }
    exponent = (double)value;
    exponent -= 64.0;
    mant = 0.0;

    (void)!fread(&b2, 1, 1, filePtr);
    (void)!fread(&b3, 1, 1, filePtr);
    (void)!fread(&b4, 1, 1, filePtr);
    (void)!fread(&b5, 1, 1, filePtr);
    (void)!fread(&b6, 1, 1, filePtr);
    (void)!fread(&b7, 1, 1, filePtr);
    (void)!fread(&b8, 1, 1, filePtr);

    mant += b8;
    mant /= 256.0;
    mant += b7;
    mant /= 256.0;
    mant += b6;
    mant /= 256.0;
    mant += b5;
    mant /= 256.0;
    mant += b4;
    mant /= 256.0;
    mant += b3;
    mant /= 256.0;
    mant += b2;
    mant /= 256.0;

    currentRecordLen -= 8;

    return sign * (mant * std::pow(16.0, exponent));
  }

  void parseHeader() {
    short version;
    version = readTwoByteSignedInt();
  }

  void parseLibName() {
    char *str;
    str = readAsciiString();
    geometry->setLibName(str);
    delete[] str;
  }

  void parseUnits() {
    userUnits = readEightByteReal();
    units = readEightByteReal();
    units = units * 1.0e6; /*in micron*/
  }

  void parseStructureName() {
    char *str = readAsciiString();

    if (str) {
      currentStructure.name = str;
      delete[] str;
    }
  }

  void parseSName() {
    // parse the structure reference
    char *str = readAsciiString();
    if (str) {
      if (currentElement == elSRef) {
        currentStructure.sRefs.back().strName = str;
      } else if (currentElement == elARef) {
        currentStructure.aRefs.back().strName = str;
      }
      delete[] str;
    }
  }

  void parseXYBoundary() {
    float X, Y;
    unsigned int numPoints = currentRecordLen / 8;
    auto &currentElPointCloud = currentStructure.elements.back().pointCloud;

    // do not include the last point since it
    // is just a copy of the first
    for (unsigned int i = 0; i < numPoints - 1; i++) {
      X = units * (float)readFourByteSignedInt();
      Y = units * (float)readFourByteSignedInt();

      currentElPointCloud.push_back(std::array<NumericType, 3>{
          static_cast<NumericType>(X), static_cast<NumericType>(Y),
          NumericType(0)});

      if (X < currentStructure.elementBoundingBox[0][0]) {
        currentStructure.elementBoundingBox[0][0] = X;
      }
      if (X > currentStructure.elementBoundingBox[1][0]) {
        currentStructure.elementBoundingBox[1][0] = X;
      }
      if (Y < currentStructure.elementBoundingBox[0][1]) {
        currentStructure.elementBoundingBox[0][1] = Y;
      }
      if (Y > currentStructure.elementBoundingBox[1][1]) {
        currentStructure.elementBoundingBox[1][1] = Y;
      }
    }
    readFourByteSignedInt(); // parse remaining points
    readFourByteSignedInt();
  }

  void parseXYIgnore() {
    unsigned int numPoints = currentRecordLen / 8;
    for (unsigned int i = 0; i < numPoints * 2; i++) {
      readFourByteSignedInt();
    }
  }

  void parseXYRef() {
    bool flipped = ((uint16_t)(currentSTrans & 0x8000) == (uint16_t)0x8000);

    if (currentElement == elSRef) {
      float X = units * (float)readFourByteSignedInt();
      float Y = units * (float)readFourByteSignedInt();
      currentStructure.sRefs.back().refPoint[0] = static_cast<NumericType>(X);
      currentStructure.sRefs.back().refPoint[1] = static_cast<NumericType>(Y);
      currentStructure.sRefs.back().refPoint[2] = static_cast<NumericType>(0);

      currentStructure.sRefs.back().magnification =
          static_cast<NumericType>(currentMag);
      currentStructure.sRefs.back().angle =
          static_cast<NumericType>(currentAngle);
      currentStructure.sRefs.back().flipped = flipped;
    } else {
      for (size_t i = 0; i < 3; i++) {
        float X = units * (float)readFourByteSignedInt();
        float Y = units * (float)readFourByteSignedInt();
        currentStructure.aRefs.back().refPoints[i][0] =
            static_cast<NumericType>(X);
        currentStructure.aRefs.back().refPoints[i][1] =
            static_cast<NumericType>(Y);
        currentStructure.aRefs.back().refPoints[i][2] =
            static_cast<NumericType>(0);
      }

      currentStructure.aRefs.back().magnification =
          static_cast<NumericType>(currentMag);
      currentStructure.aRefs.back().angle =
          static_cast<NumericType>(currentAngle);
      currentStructure.aRefs.back().flipped = flipped;

      currentStructure.aRefs.back().arrayDims[0] = arrayRows;
      currentStructure.aRefs.back().arrayDims[1] = arrayCols;
    }

    currentAngle = 0;
    currentMag = 0;
    currentSTrans = 0;
  }

  void parseFile() {
    filePtr = fopen(fileName.c_str(), "rb");
    if (!filePtr) {
      std::cerr << "Could not open GDS file." << std::endl;
      return;
    }

    unsigned char recordType, dataType;
    resetCurrentStructure();

    while (!feof(filePtr)) {
      currentRecordLen = readTwoByteSignedInt();
      (void)!fread(&recordType, 1, 1, filePtr);
      (void)!fread(&dataType, 1, 1, filePtr);
      currentRecordLen -= 4;

      switch (recordType) {
      case psGDSRecordNumbers::Header:
        parseHeader();
        break;

      case psGDSRecordNumbers::BgnLib:
        while (currentRecordLen)
          readTwoByteSignedInt(); // read modification date/time
        break;

      case psGDSRecordNumbers::LibName:
        parseLibName();
        break;

      case psGDSRecordNumbers::Units:
        parseUnits();
        break;

      case psGDSRecordNumbers::EndLib:
        fseek(filePtr, 0, SEEK_END);
        return;

      case psGDSRecordNumbers::BgnStr: // begin structure
        assert(currentStructure.name == "" &&
               currentStructure.elements.empty() &&
               currentStructure.sRefs.empty() &&
               currentStructure.aRefs
                   .empty()); // assert current structure is reset
        while (currentRecordLen)
          readTwoByteSignedInt(); // read modification date/time
        break;

      case psGDSRecordNumbers::StrName:
        parseStructureName();
        break;

      case psGDSRecordNumbers::EndStr: // current structure finished
        geometry->insertNextStructure(currentStructure);
        resetCurrentStructure();
        break;

      case psGDSRecordNumbers::EndEl:
        ignore = false;
        break;

      case psGDSRecordNumbers::Boundary:
        currentStructure.elements.push_back(
            psGDSElement<NumericType>{elBoundary});
        currentElement = elBoundary;
        currentStructure.boundaryElements++;
        break;

      case psGDSRecordNumbers::Box:
        currentStructure.elements.push_back(psGDSElement<NumericType>{elBox});
        currentElement = elBox;
        currentStructure.boxElements++;
        break;

      case psGDSRecordNumbers::BoxType: // ignore
        readTwoByteSignedInt();
        break;

      case psGDSRecordNumbers::Layer:
        currentLayer = readTwoByteSignedInt();
        if (!ignore) {
          assert(currentStructure.elements.size() > 0);
          currentStructure.elements.back().layer = currentLayer;
          currentStructure.containsLayers.insert(currentLayer);
        }
        break;

      case psGDSRecordNumbers::Plex:
        currentPlexNumber = readFourByteSignedInt();
        if (!ignore) {
          assert(currentStructure.elements.size() > 0);
          currentStructure.elements.back().plexNumber = currentPlexNumber;
        }
        break;

      case psGDSRecordNumbers::XY:
        if (currentElement == elBoundary || currentElement == elBox) {
          parseXYBoundary();
        } else if (currentElement == elSRef || currentElement == elARef) {
          parseXYRef();
        } else {
          parseXYIgnore();
        }
        break;

      case psGDSRecordNumbers::SRef:
        currentElement = elSRef;
        currentStructure.sRefs.push_back(psGDSSRef<NumericType>{});
        break;

      case psGDSRecordNumbers::ARef:
        currentElement = elARef;
        currentStructure.aRefs.push_back(psGDSARef<NumericType>{});
        break;

      case psGDSRecordNumbers::SName:
        parseSName();
        break;

      case psGDSRecordNumbers::STrans:
        currentSTrans = readTwoByteSignedInt();
        break;

      case psGDSRecordNumbers::Mag:
        currentMag = readEightByteReal();
        break;

      case psGDSRecordNumbers::Angle:
        currentAngle = readEightByteReal();
        break;

      case psGDSRecordNumbers::ColRow:
        arrayCols = readTwoByteSignedInt();
        arrayRows = readTwoByteSignedInt();
        break;

      case psGDSRecordNumbers::Text: // ignore
        currentElement = elText;
        ignore = true;
        break;

      case psGDSRecordNumbers::TextType: // ignore
        readTwoByteSignedInt();
        break;

      case psGDSRecordNumbers::Presentation: // ignore
        readTwoByteSignedInt();
        break;

      case psGDSRecordNumbers::String: // ignore
        tempStr = readAsciiString();
        delete[] tempStr;
        break;

      case psGDSRecordNumbers::Path: // ignore
        currentElement = elPath;
        ignore = true;
        break;

      case psGDSRecordNumbers::PathType: // ignore
        readTwoByteSignedInt();
        break;

      case psGDSRecordNumbers::Width: // ignore (only used in path and text)
        currentWidth = (float)readFourByteSignedInt();
        if (currentWidth > 0) {
          currentWidth *= units;
        }
        break;

      case psGDSRecordNumbers::DataType: // unimportant and should be zero
        currentDataType = readTwoByteSignedInt();
        if (currentDataType != 0)
          std::cout << "WARNING: unsupported argument in DATATYPE" << std::endl;
        break;

      case psGDSRecordNumbers::Node: // ignore
        currentElement = elNone;
        while (currentRecordLen) {
          readTwoByteSignedInt();
        }
        ignore = true;
        break;

      case psGDSRecordNumbers::ElFlags: // ignore
        while (currentRecordLen)
          readTwoByteSignedInt();
        break;

      case psGDSRecordNumbers::ElKey: // ignore
        while (currentRecordLen)
          readTwoByteSignedInt();
        break;

      case psGDSRecordNumbers::RefLibs: // ignore
        tempStr = readAsciiString();
        delete[] tempStr;
        break;

      case psGDSRecordNumbers::Fonts: // ignore
        tempStr = readAsciiString();
        delete[] tempStr;
        break;

      case psGDSRecordNumbers::Generations: // ignore
        while (currentRecordLen)
          readTwoByteSignedInt();
        break;
      case psGDSRecordNumbers::AttrTable: // ignore
        tempStr = readAsciiString();
        delete[] tempStr;
        break;

      case psGDSRecordNumbers::StypTable: // ignore
        readTwoByteSignedInt();
        break;

      case psGDSRecordNumbers::StrType: // ignore
        tempStr = readAsciiString();
        delete[] tempStr;
        break;

      case psGDSRecordNumbers::LinkType: // ignore
        while (currentRecordLen)
          readTwoByteSignedInt();
        break;

      case psGDSRecordNumbers::LinkKeys: // ignore
        while (currentRecordLen)
          readFourByteSignedInt();
        break;

      case psGDSRecordNumbers::NodeType: // ignore
        while (currentRecordLen)
          readTwoByteSignedInt();
        break;

      case psGDSRecordNumbers::PropAttr: // ignore
        while (currentRecordLen)
          readTwoByteSignedInt();
        break;

      case psGDSRecordNumbers::PropValue: // ignore
        tempStr = readAsciiString();
        delete[] tempStr;
        break;

      case psGDSRecordNumbers::BgnExtn: // ignore
        readFourByteSignedInt();
        break;

      case psGDSRecordNumbers::EndExtn: // ignore
        readFourByteSignedInt();
        break;

      case psGDSRecordNumbers::TapeNum: // ignore
        while (currentRecordLen)
          readTwoByteSignedInt();
        break;

      case psGDSRecordNumbers::TapeCode: // ignore
        while (currentRecordLen)
          readTwoByteSignedInt();
        break;

      case psGDSRecordNumbers::StrClass: // ignore
        while (currentRecordLen)
          readTwoByteSignedInt();
        break;

      case psGDSRecordNumbers::Reserved: // ignore
        /* Empty */
        break;

      case psGDSRecordNumbers::Format: // ignore
        while (currentRecordLen)
          readTwoByteSignedInt();
        break;

      case psGDSRecordNumbers::Mask: // ignore
        tempStr = readAsciiString();
        delete[] tempStr;
        break;

      case psGDSRecordNumbers::EndMasks: // ignore
        /* Empty */
        break;

      case psGDSRecordNumbers::LibDirSize: // ignore
        while (currentRecordLen)
          readTwoByteSignedInt();
        break;

      case psGDSRecordNumbers::SrfName: // ignore
        tempStr = readAsciiString();
        delete[] tempStr;
        break;

      case psGDSRecordNumbers::LibSecur: // ignore
        while (currentRecordLen)
          readTwoByteSignedInt();
        break;

      case psGDSRecordNumbers::Border: // ignore
        /* Empty */
        break;

      case psGDSRecordNumbers::SoftFence: // ignore
        /* Empty */
        break;

      case psGDSRecordNumbers::HardFence: // ignore
        /* Empty */
        break;

      case psGDSRecordNumbers::SoftWire: // ignore
        /* Empty */
        break;

      case psGDSRecordNumbers::HardWire: // ignore
        /* Empty */
        break;

      case psGDSRecordNumbers::PathPort: // ignore
        /* Empty */
        break;

      case psGDSRecordNumbers::NodePort: // ignore
        /* Empty */
        break;

      case psGDSRecordNumbers::UserConstraint: // ignore
        /* Empty */
        break;

      case psGDSRecordNumbers::SpacerError: // ignore
        /* Empty */
        break;

      case psGDSRecordNumbers::Contact: // ignore
        /* Empty */
        break;

      default:
        std::cerr << "Unknown record type!" << std::endl;
        return;
      }
    }
  }
};