#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>

#include "psGDSGeometry.hpp"
#include "psGDSUtils.hpp"

/// This class reads a GDS file and creates a psGDSGeometry object. It is a
/// very simple implementation and does not support all GDS features.
template <typename NumericType, int D = 3> class psGDSReader {
  using GeometryType = psSmartPointer<psGDSGeometry<NumericType, D>>;

  FILE *filePtr = nullptr;
  GeometryType geometry = nullptr;
  std::string fileName;

public:
  psGDSReader() {}
  psGDSReader(GeometryType passedGeometry, std::string passedFileName)
      : geometry(passedGeometry), fileName(std::move(passedFileName)) {}

  void setGeometry(GeometryType passedGeometry) { geometry = passedGeometry; }

  void setFileName(std::string passedFileName) {
    fileName = std::move(passedFileName);
  }

  void apply() {
    if constexpr (D == 2) {
      psLogger::getInstance()
          .addWarning("Cannot import 2D geometry from GDS file.")
          .print();
      return;
    }

    parseFile();
    geometry->finalize();
  }

private:
  psGDS::Structure<NumericType> currentStructure;

  int16_t currentRecordLen = 0;
  int16_t currentLayer;
  int16_t currentDataType;
  int16_t currentPlexNumber;
  int16_t currentSTrans;
  double currentMag = 0;
  double currentAngle = 0;
  int16_t arrayCols, arrayRows;
  bool ignore = false;
  psGDS::ElementType currentElement;
  double units; // units in micron
  double userUnits;

  // unused
  float currentWidth;
  char *tempStr;

  bool contains(int32_t X, int32_t Y,
                const std::vector<std::array<int32_t, 2>> &uniPoints) {
    for (const auto &p : uniPoints) {
      if (p[0] == X && p[1] == Y)
        return true;
    }

    return false;
  }

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
    std::array<unsigned char, 7> bytes;
    double sign = 1.0;
    double exponent;
    double mant;

    (void)!fread(&value, 1, 1, filePtr);
    if (value & 128) {
      value -= 128;
      sign = -1.0;
    }
    exponent = static_cast<double>(value);
    exponent -= 64.0;
    mant = 0.0;

    for (int i = 0; i < 7; i++) {
      (void)!fread(&bytes[i], 1, 1, filePtr);
    }

    for (int i = 6; i >= 0; i--) {
      mant += static_cast<double>(bytes[i]);
      mant /= 256.0;
    }

    currentRecordLen -= 8;

    return sign * (mant * std::pow(16.0, exponent));
  }

  void parseHeader() {
    short version;
    version = readTwoByteSignedInt();
    psLogger::getInstance()
        .addDebug("GDS Version: " + std::to_string(version))
        .print();
  }

  void parseLibName() {
    char *str;
    str = readAsciiString();
    psLogger::getInstance()
        .addDebug("GDS Library name: " + std::string(str))
        .print();
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
      if (currentElement == psGDS::ElementType::elSRef) {
        currentStructure.sRefs.back().strName = str;
      } else if (currentElement == psGDS::ElementType::elARef) {
        currentStructure.aRefs.back().strName = str;
      }
      delete[] str;
    }
  }

  void parseXYBoundary() {
    float X, Y;
    unsigned int numPoints = currentRecordLen / 8;
    auto &currentElPointCloud = currentStructure.elements.back().pointCloud;
    std::vector<std::array<int32_t, 2>> uniquePoints;

    // do not include the last point since it
    // is just a copy of the first
    for (unsigned int i = 0; i < numPoints - 1; i++) {
      auto pX = readFourByteSignedInt();
      auto pY = readFourByteSignedInt();

      if (!contains(pX, pY, uniquePoints)) {
        uniquePoints.push_back({pX, pY});

        X = units * (float)pX;
        Y = units * (float)pY;

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

    if (currentElement == psGDS::ElementType::elSRef) {
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
      psLogger::getInstance().addError("Could not open GDS file.").print();
      return;
    }

    unsigned char recordType, dataType;
    resetCurrentStructure();

    while (!feof(filePtr)) {
      currentRecordLen = readTwoByteSignedInt();
      (void)!fread(&recordType, 1, 1, filePtr);
      (void)!fread(&dataType, 1, 1, filePtr);
      currentRecordLen -= 4;

      switch (static_cast<psGDS::RecordNumbers>(recordType)) {
      case psGDS::RecordNumbers::Header:
        parseHeader();
        break;

      case psGDS::RecordNumbers::BgnLib:
        while (currentRecordLen)
          readTwoByteSignedInt(); // read modification date/time
        break;

      case psGDS::RecordNumbers::LibName:
        parseLibName();
        break;

      case psGDS::RecordNumbers::Units:
        parseUnits();
        break;

      case psGDS::RecordNumbers::EndLib:
        fseek(filePtr, 0, SEEK_END);
        return;

      case psGDS::RecordNumbers::BgnStr: // begin structure
        assert(currentStructure.name == "" &&
               currentStructure.elements.empty() &&
               currentStructure.sRefs.empty() &&
               currentStructure.aRefs
                   .empty()); // assert current structure is reset
        while (currentRecordLen)
          readTwoByteSignedInt(); // read modification date/time
        break;

      case psGDS::RecordNumbers::StrName:
        parseStructureName();
        break;

      case psGDS::RecordNumbers::EndStr: // current structure finished
        geometry->insertNextStructure(currentStructure);
        resetCurrentStructure();
        break;

      case psGDS::RecordNumbers::EndEl:
        ignore = false;
        break;

      case psGDS::RecordNumbers::Boundary:
        currentStructure.elements.push_back(
            psGDS::Element<NumericType>{psGDS::ElementType::elBoundary});
        currentElement = psGDS::ElementType::elBoundary;
        currentStructure.boundaryElements++;
        break;

      case psGDS::RecordNumbers::Box:
        currentStructure.elements.push_back(
            psGDS::Element<NumericType>{psGDS::ElementType::elBox});
        currentElement = psGDS::ElementType::elBox;
        currentStructure.boxElements++;
        break;

      case psGDS::RecordNumbers::BoxType: // ignore
        readTwoByteSignedInt();
        break;

      case psGDS::RecordNumbers::Layer:
        currentLayer = readTwoByteSignedInt();
        if (!ignore) {
          assert(currentStructure.elements.size() > 0);
          currentStructure.elements.back().layer = currentLayer;
          currentStructure.containsLayers.insert(currentLayer);
        }
        break;

      case psGDS::RecordNumbers::Plex:
        currentPlexNumber = readFourByteSignedInt();
        if (!ignore) {
          assert(currentStructure.elements.size() > 0);
          currentStructure.elements.back().plexNumber = currentPlexNumber;
        }
        break;

      case psGDS::RecordNumbers::XY:
        if (currentElement == psGDS::ElementType::elBoundary ||
            currentElement == psGDS::ElementType::elBox) {
          parseXYBoundary();
        } else if (currentElement == psGDS::ElementType::elSRef ||
                   currentElement == psGDS::ElementType::elARef) {
          parseXYRef();
        } else {
          parseXYIgnore();
        }
        break;

      case psGDS::RecordNumbers::SRef:
        currentElement = psGDS::ElementType::elSRef;
        currentStructure.sRefs.push_back(psGDS::SRef<NumericType>{});
        break;

      case psGDS::RecordNumbers::ARef:
        currentElement = psGDS::ElementType::elARef;
        currentStructure.aRefs.push_back(psGDS::ARef<NumericType>{});
        break;

      case psGDS::RecordNumbers::SName:
        parseSName();
        break;

      case psGDS::RecordNumbers::STrans:
        currentSTrans = readTwoByteSignedInt();
        break;

      case psGDS::RecordNumbers::Mag:
        currentMag = readEightByteReal();
        break;

      case psGDS::RecordNumbers::Angle:
        currentAngle = readEightByteReal();
        break;

      case psGDS::RecordNumbers::ColRow:
        arrayCols = readTwoByteSignedInt();
        arrayRows = readTwoByteSignedInt();
        break;

      case psGDS::RecordNumbers::Text: // ignore
        currentElement = psGDS::ElementType::elText;
        ignore = true;
        break;

      case psGDS::RecordNumbers::TextType: // ignore
        readTwoByteSignedInt();
        break;

      case psGDS::RecordNumbers::Presentation: // ignore
        readTwoByteSignedInt();
        break;

      case psGDS::RecordNumbers::String: // ignore
        tempStr = readAsciiString();
        delete[] tempStr;
        break;

      case psGDS::RecordNumbers::Path: // ignore
        currentElement = psGDS::ElementType::elPath;
        ignore = true;
        break;

      case psGDS::RecordNumbers::PathType: // ignore
        readTwoByteSignedInt();
        break;

      case psGDS::RecordNumbers::Width: // ignore (only used in path and text)
        currentWidth = (float)readFourByteSignedInt();
        if (currentWidth > 0) {
          currentWidth *= units;
        }
        break;

      case psGDS::RecordNumbers::DataType: // unimportant and should be zero
        currentDataType = readTwoByteSignedInt();
        if (currentDataType != 0)
          psLogger::getInstance()
              .addWarning("Unsupported argument in DATATYPE")
              .print();
        break;

      case psGDS::RecordNumbers::Node: // ignore
        currentElement = psGDS::ElementType::elNone;
        while (currentRecordLen) {
          readTwoByteSignedInt();
        }
        ignore = true;
        break;

      case psGDS::RecordNumbers::ElFlags: // ignore
        while (currentRecordLen)
          readTwoByteSignedInt();
        break;

      case psGDS::RecordNumbers::ElKey: // ignore
        while (currentRecordLen)
          readTwoByteSignedInt();
        break;

      case psGDS::RecordNumbers::RefLibs: // ignore
        tempStr = readAsciiString();
        delete[] tempStr;
        break;

      case psGDS::RecordNumbers::Fonts: // ignore
        tempStr = readAsciiString();
        delete[] tempStr;
        break;

      case psGDS::RecordNumbers::Generations: // ignore
        while (currentRecordLen)
          readTwoByteSignedInt();
        break;
      case psGDS::RecordNumbers::AttrTable: // ignore
        tempStr = readAsciiString();
        delete[] tempStr;
        break;

      case psGDS::RecordNumbers::StypTable: // ignore
        readTwoByteSignedInt();
        break;

      case psGDS::RecordNumbers::StrType: // ignore
        tempStr = readAsciiString();
        delete[] tempStr;
        break;

      case psGDS::RecordNumbers::LinkType: // ignore
        while (currentRecordLen)
          readTwoByteSignedInt();
        break;

      case psGDS::RecordNumbers::LinkKeys: // ignore
        while (currentRecordLen)
          readFourByteSignedInt();
        break;

      case psGDS::RecordNumbers::NodeType: // ignore
        while (currentRecordLen)
          readTwoByteSignedInt();
        break;

      case psGDS::RecordNumbers::PropAttr: // ignore
        while (currentRecordLen)
          readTwoByteSignedInt();
        break;

      case psGDS::RecordNumbers::PropValue: // ignore
        tempStr = readAsciiString();
        delete[] tempStr;
        break;

      case psGDS::RecordNumbers::BgnExtn: // ignore
        readFourByteSignedInt();
        break;

      case psGDS::RecordNumbers::EndExtn: // ignore
        readFourByteSignedInt();
        break;

      case psGDS::RecordNumbers::TapeNum: // ignore
        while (currentRecordLen)
          readTwoByteSignedInt();
        break;

      case psGDS::RecordNumbers::TapeCode: // ignore
        while (currentRecordLen)
          readTwoByteSignedInt();
        break;

      case psGDS::RecordNumbers::StrClass: // ignore
        while (currentRecordLen)
          readTwoByteSignedInt();
        break;

      case psGDS::RecordNumbers::Reserved: // ignore
        /* Empty */
        break;

      case psGDS::RecordNumbers::Format: // ignore
        while (currentRecordLen)
          readTwoByteSignedInt();
        break;

      case psGDS::RecordNumbers::Mask: // ignore
        tempStr = readAsciiString();
        delete[] tempStr;
        break;

      case psGDS::RecordNumbers::EndMasks: // ignore
        /* Empty */
        break;

      case psGDS::RecordNumbers::LibDirSize: // ignore
        while (currentRecordLen)
          readTwoByteSignedInt();
        break;

      case psGDS::RecordNumbers::SrfName: // ignore
        tempStr = readAsciiString();
        delete[] tempStr;
        break;

      case psGDS::RecordNumbers::LibSecur: // ignore
        while (currentRecordLen)
          readTwoByteSignedInt();
        break;

      case psGDS::RecordNumbers::Border: // ignore
        /* Empty */
        break;

      case psGDS::RecordNumbers::SoftFence: // ignore
        /* Empty */
        break;

      case psGDS::RecordNumbers::HardFence: // ignore
        /* Empty */
        break;

      case psGDS::RecordNumbers::SoftWire: // ignore
        /* Empty */
        break;

      case psGDS::RecordNumbers::HardWire: // ignore
        /* Empty */
        break;

      case psGDS::RecordNumbers::PathPort: // ignore
        /* Empty */
        break;

      case psGDS::RecordNumbers::NodePort: // ignore
        /* Empty */
        break;

      case psGDS::RecordNumbers::UserConstraint: // ignore
        /* Empty */
        break;

      case psGDS::RecordNumbers::SpacerError: // ignore
        /* Empty */
        break;

      case psGDS::RecordNumbers::Contact: // ignore
        /* Empty */
        break;

      default:
        psLogger::getInstance()
            .addWarning("Unknown record type in GDS file.")
            .print();
        return;
      }
    }
  }
};
