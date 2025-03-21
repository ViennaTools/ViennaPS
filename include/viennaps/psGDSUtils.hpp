#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#ifndef endian_swap_long
#define endian_swap_long(w)                                                    \
  (((w & 0xff) << 24) | ((w & 0xff00) << 8) | ((w & 0xff0000) >> 8) |          \
   ((w & 0xff000000) >> 24))
#endif
#ifndef endian_swap_short
#define endian_swap_short(w) (((w & 0xff) << 8) | ((w & 0xff00) >> 8))
#endif

namespace viennaps::GDS {

enum class ElementType {
  elBoundary,
  elBox,
  elPath,
  elSRef,
  elARef,
  elText,
  elNone
};

enum class RecordNumbers {
  Header, /* 0 */
  BgnLib,
  LibName,
  Units,
  EndLib,
  BgnStr,
  StrName,
  EndStr,
  Boundary,
  Path,
  SRef, /* 10 */
  ARef,
  Text,
  Layer,
  DataType,
  Width,
  XY,
  EndEl,
  SName,
  ColRow,
  TextNode, /* 20 */
  Node,
  TextType,
  Presentation,
  Spacing,
  String,
  STrans,
  Mag,
  Angle,
  UInteger,
  UString, /* 30 */
  RefLibs,
  Fonts,
  PathType,
  Generations,
  AttrTable,
  StypTable,
  StrType,
  ElFlags,
  ElKey,
  LinkType, /* 40 */
  LinkKeys,
  NodeType,
  PropAttr,
  PropValue,
  Box,
  BoxType,
  Plex,
  BgnExtn,
  EndExtn,
  TapeNum, /* 50 */
  TapeCode,
  StrClass,
  Reserved,
  Format,
  Mask,
  EndMasks,
  LibDirSize,
  SrfName,
  LibSecur,
  Border, /* 60 */
  SoftFence,
  HardFence,
  SoftWire,
  HardWire,
  PathPort,
  NodePort,
  UserConstraint,
  SpacerError,
  Contact /* 69 */
};

template <class T> struct Element {
  ElementType elementType;
  int16_t layer;
  int32_t plexNumber = -1;
  std::vector<std::array<T, 3>> pointCloud;

  // Member function to print pointCloud
  void print() const {
    std::cout << "Point Cloud Data:\n";
    for (const auto& point : pointCloud) {
        std::cout << "(" << point[0] << ", " << point[1] << ", " << point[2] << ")\n";
    }
  }
};

template <class T> struct SRef {
  std::string strName;
  T angle = 0;
  T magnification = 0;
  bool flipped = false;
  std::array<T, 3> refPoint;
};

template <class T> struct ARef {
  std::string strName;
  T angle = 0;
  T magnification = 0;
  bool flipped = false;
  std::array<std::array<T, 3>, 3> refPoints;
  std::array<int16_t, 2> arrayDims;
};

template <class T> struct Structure {
  std::vector<Element<T>> elements;
  std::vector<SRef<T>> sRefs;
  std::vector<ARef<T>> aRefs;
  std::string name;
  int boundaryElements = 0;
  int boxElements = 0;
  std::array<std::array<T, 2>, 2> elementBoundingBox;
  std::array<std::array<T, 2>, 2> boundingBox;
  bool isRef = false;
  std::set<int16_t> containsLayers;

  std::array<T, 2> getElementExtent() const {
    return {elementBoundingBox[1][0] - elementBoundingBox[0][0],
            elementBoundingBox[1][1] - elementBoundingBox[0][1]};
  }

  void printBoundingBox() const {
    std::cout << "Structure " << name << ": (" << boundingBox[0][0] << ", "
              << boundingBox[0][1] << ") - (" << boundingBox[1][0] << ", "
              << boundingBox[1][1] << ")" << std::endl;
  }

  void print() const {
    std::cout << name << ":\n\nBoundary elements: " << boundaryElements
              << std::endl;
    for (auto &e : elements) {
      if (e.elementType == ElementType::elBoundary) {
        std::cout << "---------------------------\n";
        std::cout << "Layer: " << e.layer << "\n";
        if (e.plexNumber > 0) {
          std::cout << "Plex number: " << e.plexNumber << std::endl;
        }
        std::cout << "Points: ";
        for (auto &p : e.pointCloud) {
          std::cout << "(" << p[0] << ", " << p[1] << ") ";
        }
        std::cout << "\n---------------------------\n";
      }
    }

    std::cout << "\nBox elements: " << boxElements << std::endl;
    for (auto &e : elements) {
      if (e.elementType == ElementType::elBox) {
        std::cout << "---------------------------\n";
        std::cout << "Layer: " << e.layer << "\n";
        if (e.plexNumber > 0) {
          std::cout << "Plex number: " << e.plexNumber << std::endl;
        }
        std::cout << "Points: ";
        for (auto &p : e.pointCloud) {
          std::cout << "(" << p[0] << ", " << p[1] << ") ";
        }
        std::cout << "\n---------------------------\n";
      }
    }
  }
};

void saveContoursToCSV(const std::vector<std::pair<double, double>> &contours, const std::string &filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
      std::cerr << "Error opening file: " << filename << std::endl;
      return;
  }

  for (const auto &point : contours) {
      file << point.first << "," << point.second << "\n";
  }

  file.close();
}

} // namespace viennaps::GDS
