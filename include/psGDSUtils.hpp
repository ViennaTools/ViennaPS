#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <lsGeometries.hpp>

typedef enum psGDSElementType {
  elBoundary,
  elBox,
  elPath,
  elSRef,
  elARef,
  elText,
  elNone
} psGDSElementType;

enum psGDSRecordNumbers {
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

template <class T> struct psGDSElement {
  psGDSElementType elementType;
  int16_t layer;
  int32_t plexNumber = -1;
  std::vector<std::array<T, 3>> pointCloud;
};

template <class T> struct psGDSSRef {
  std::string strName;
  T angle = 0;
  T magnification = 0;
  bool flipped = false;
  std::array<T, 3> refPoint;
};

template <class T> struct psGDSARef {
  std::string strName;
  T angle = 0;
  T magnification = 0;
  bool flipped = false;
  std::array<std::array<T, 3>, 3> refPoints;
  std::array<int16_t, 2> arrayDims;
};

template <class T> struct psGDSStructure {
  std::vector<psGDSElement<T>> elements;
  std::vector<psGDSSRef<T>> sRefs;
  std::vector<psGDSARef<T>> aRefs;
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

  void print() {
    std::cout << name << ":\n\nBoundary elements: " << boundaryElements
              << std::endl;
    for (auto &e : elements) {
      if (e.elementType == elBoundary) {
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
      if (e.elementType == elBox) {
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
