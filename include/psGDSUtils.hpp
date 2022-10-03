#pragma once

#include <cstdint>

typedef enum psGDSElementType {
  elBoundary,
  elBox,
  elPath,
  elSRef,
  elARef,
  elText,
  elNone
} psGDSElementType;

/* Two consecutive zero bytes are a null word */

typedef uint8_t psByte;

typedef struct psGDSHeader {
  short RecordLength;
  psByte RecordType;
  psByte DataType;
  psByte *Data;
} psGDSHeader;

// Words are 16 bit.
enum psGDSDataTypes {
  dtNoData,
  dtBitArray,
  // A word containing bits or group of bits that represent data.
  // A bit array allows one wid to contain more than one piece of
  // information.
  dtTwoByteSignedInt,
  // One word in 2s-complement representation.
  // Range is -32,768 to 32,767
  // S - sign, M - magnitude
  // SMMMMMMM MMMMMMMM
  dtFourByteSignedInt,
  // Two words in 2s-complement representation.
  // Range is -2,147,483,648 to 2,147,483,647
  // S - sign, M - magnitude
  // SMMMMMMM MMMMMMMM MMMMMMMM MMMMMMMM
  dtFourByteReal,
  dtEightByteReal,
  // 2 and 4 word floating point representation.
  // (Mantissa) * (16 raised to the true value of exponent field)
  //
  // Exponent is greater by 64 than the actual exponent.
  // So if E=65, then the exponent=1.
  //
  // Mantissa is always a positive fraction greater than or equal
  // to 1/16 and less than 1.
  //
  // S - sign, E - exponent, M - mantissa
  // SEEEEEEE MMMMMMMM MMMMMMMM MMMMMMMM
  dtAsciiString
  // A collection of bytes representing ascii characters. All odd
  // length strings are padded with a null character and the byte
  // count for the record containg this string includes the null
  // character.
};

enum psGDSRecordNumbers {
  rnHeader, /* 0 */
  rnBgnLib,
  rnLibName,
  rnUnits,
  rnEndLib,
  rnBgnStr,
  rnStrName,
  rnEndStr,
  rnBoundary,
  rnPath,
  rnSRef, /* 10 */
  rnARef,
  rnText,
  rnLayer,
  rnDataType,
  rnWidth,
  rnXY,
  rnEndEl,
  rnSName,
  rnColRow,
  rnTextNode, /* 20 */
  rnNode,
  rnTextType,
  rnPresentation,
  rnSpacing,
  rnString,
  rnSTrans,
  rnMag,
  rnAngle,
  rnUInteger,
  rnUString, /* 30 */
  rnRefLibs,
  rnFonts,
  rnPathType,
  rnGenerations,
  rnAttrTable,
  rnStypTable,
  rnStrType,
  rnElFlags,
  rnElKey,
  rnLinkType, /* 40 */
  rnLinkKeys,
  rnNodeType,
  rnPropAttr,
  rnPropValue,
  rnBox,
  rnBoxType,
  rnPlex,
  rnBgnExtn,
  rnEndExtn,
  rnTapeNum, /* 50 */
  rnTapeCode,
  rnStrClass,
  rnReserved,
  rnFormat,
  rnMask,
  rnEndMasks,
  rnLibDirSize,
  rnSrfName,
  rnLibSecur,
  rnBorder, /* 60 */
  rnSoftFence,
  rnHardFence,
  rnSoftWire,
  rnHardWire,
  rnPathPort,
  rnNodePort,
  rnUserConstraint,
  rnSpacerError,
  rnContact /* 69 */
};