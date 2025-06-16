#pragma once

namespace viennaps::units {

class Length {
  static int unit_;

public:
  enum : int {
    METER,
    CENTIMETER,
    MILLIMETER,
    MICROMETER,
    NANOMETER,
    ANGSTROM,
    UNDEFINED
  };

  Length() = default;

  static void setUnit(const int passedUnit) { unit_ = passedUnit; }

  static void setUnit(const std::string &unit) {
    if (unit == "meter" || unit == "m") {
      unit_ = METER;
    } else if (unit == "centimeter" || unit == "cm") {
      unit_ = CENTIMETER;
    } else if (unit == "millimeter" || unit == "mm") {
      unit_ = MILLIMETER;
    } else if (unit == "micrometer" || unit == "um") {
      unit_ = MICROMETER;
    } else if (unit == "nanometer" || unit == "nm") {
      unit_ = NANOMETER;
    } else if (unit == "angstrom" || unit == "A") {
      unit_ = ANGSTROM;
    } else {
      throw std::invalid_argument(
          "The value must be one of the following: meter, centimeter, "
          "millimeter, micrometer, nanometer, angstrom");
    }
  }

  static int getUnit() { return unit_; }

  static Length &getInstance() {
    static Length instance;
    return instance;
  }

  // delete constructors to result in better error messages by compilers
  Length(const Length &) = delete;
  void operator=(const Length &) = delete;

  static double convertMeter() {
    switch (unit_) {
    case METER:
      return 1.;
    case CENTIMETER:
      return 1e-2;
    case MILLIMETER:
      return 1e-3;
    case MICROMETER:
      return 1e-6;
    case NANOMETER:
      return 1e-9;
    case ANGSTROM:
      return 1e-10;
    case UNDEFINED: {
      Logger::getInstance().addWarning("Length unit is not defined.").print();
      return 1.;
    }
    default:
      Logger::getInstance().addError("Invalid length unit.").print();
    }

    return 0.;
  }

  static double convertCentimeter() {
    switch (unit_) {
    case METER:
      return 1e2;
    case CENTIMETER:
      return 1.;
    case MILLIMETER:
      return 1e-1;
    case MICROMETER:
      return 1e-4;
    case NANOMETER:
      return 1e-7;
    case ANGSTROM:
      return 1e-8;
    case UNDEFINED: {
      Logger::getInstance().addWarning("Length unit is not defined.").print();
      return 1.;
    }
    default:
      Logger::getInstance().addError("Invalid length unit.").print();
    }

    return 0.;
  }

  static double convertMillimeter() {
    switch (unit_) {
    case METER:
      return 1e3;
    case CENTIMETER:
      return 1e1;
    case MILLIMETER:
      return 1.;
    case MICROMETER:
      return 1e-3;
    case NANOMETER:
      return 1e-6;
    case ANGSTROM:
      return 1e-7;
    case UNDEFINED: {
      Logger::getInstance().addWarning("Length unit is not defined.").print();
      return 1.;
    }
    default:
      Logger::getInstance().addError("Invalid length unit.").print();
    }

    return 0.;
  }

  static double convertMicrometer() {
    switch (unit_) {
    case METER:
      return 1e6;
    case CENTIMETER:
      return 1e4;
    case MILLIMETER:
      return 1e3;
    case MICROMETER:
      return 1.;
    case NANOMETER:
      return 1e-3;
    case ANGSTROM:
      return 1e-4;
    case UNDEFINED: {
      Logger::getInstance().addWarning("Length unit is not defined.").print();
      return 1.;
    }
    default:
      Logger::getInstance().addError("Invalid length unit.").print();
    }

    return 0.;
  }

  static double convertNanometer() {
    switch (unit_) {
    case METER:
      return 1e9;
    case CENTIMETER:
      return 1e7;
    case MILLIMETER:
      return 1e6;
    case MICROMETER:
      return 1e3;
    case NANOMETER:
      return 1.;
    case ANGSTROM:
      return 1e-1;
    case UNDEFINED: {
      Logger::getInstance().addWarning("Length unit is not defined.").print();
      return 1.;
    }
    default:
      Logger::getInstance().addError("Invalid length unit.").print();
    }

    return 0.;
  }

  static double convertAngstrom() {
    switch (unit_) {
    case METER:
      return 1e10;
    case CENTIMETER:
      return 1e8;
    case MILLIMETER:
      return 1e7;
    case MICROMETER:
      return 1e4;
    case NANOMETER:
      return 1e1;
    case ANGSTROM:
      return 1.;
    case UNDEFINED: {
      Logger::getInstance().addWarning("Length unit is not defined.").print();
      return 1.;
    }
    default:
      Logger::getInstance().addError("Invalid length unit.").print();
    }

    return 0.;
  }

  static std::string toString() {
    switch (unit_) {
    case METER:
      return "meter";
    case CENTIMETER:
      return "centimeter";
    case MILLIMETER:
      return "millimeter";
    case MICROMETER:
      return "micrometer";
    case NANOMETER:
      return "nanometer";
    case ANGSTROM:
      return "angstrom";
    case UNDEFINED:
      return "";
    default:
      Logger::getInstance().addError("Invalid length unit.").print();
    }

    return "error-length-unit";
  }

  static std::string toShortString() {
    switch (unit_) {
    case METER:
      return "m";
    case CENTIMETER:
      return "cm";
    case MILLIMETER:
      return "mm";
    case MICROMETER:
      return "um";
    case NANOMETER:
      return "nm";
    case ANGSTROM:
      return "A";
    case UNDEFINED:
      return "";
    default:
      Logger::getInstance().addError("Invalid length unit.").print();
    }

    return "error-length-unit";
  }
};

inline int Length::unit_ = Length::UNDEFINED;

class Time {
  static int unit_;

public:
  enum : int { MINUTE, SECOND, MILLISECOND, UNDEFINED };

  Time() = default;

  static void setUnit(const int passedUnit) { unit_ = passedUnit; }

  static void setUnit(const std::string &unit) {
    if (unit == "second" || unit == "s") {
      unit_ = SECOND;
    } else if (unit == "minute" || unit == "min") {
      unit_ = MINUTE;
    } else if (unit == "millisecond" || unit == "ms") {
      unit_ = MILLISECOND;
    } else {
      throw std::invalid_argument("The value must be one of the following: "
                                  "second, minute, millisecond");
    }
  }

  static int getUnit() { return unit_; }

  static Time &getInstance() {
    static Time instance;
    return instance;
  }

  // delete constructors to result in better error messages by compilers
  Time(const Time &) = delete;
  void operator=(const Time &) = delete;

  static double convertMinute() {
    switch (unit_) {
    case MINUTE:
      return 1.;
    case SECOND:
      return 1. / 60.;
    case MILLISECOND:
      return 1. / 60000.;
    case UNDEFINED: {
      Logger::getInstance().addWarning("Time unit is not defined.").print();
      return 1.;
    }
    default:
      Logger::getInstance().addError("Invalid time unit.").print();
    }

    return 0.;
  }

  static double convertSecond() {
    switch (unit_) {
    case MINUTE:
      return 60.;
    case SECOND:
      return 1.;
    case MILLISECOND:
      return 1e-3;
    case UNDEFINED: {
      Logger::getInstance().addWarning("Time unit is not defined.").print();
      return 1.;
    }
    default:
      Logger::getInstance().addError("Invalid time unit.").print();
    }

    return 0.;
  }

  static double convertMillisecond() {
    switch (unit_) {
    case MINUTE:
      return 60000.;
    case SECOND:
      return 1e3;
    case MILLISECOND:
      return 1.;
    case UNDEFINED: {
      Logger::getInstance().addWarning("Time unit is not defined.").print();
      return 1.;
    }
    default:
      Logger::getInstance().addError("Invalid time unit.").print();
    }

    return 0.;
  }

  static std::string toString() {
    switch (unit_) {
    case MINUTE:
      return "minute";
    case SECOND:
      return "second";
    case MILLISECOND:
      return "millisecond";
    case UNDEFINED:
      return "";
    default:
      Logger::getInstance().addError("Invalid time unit.").print();
    }

    return "error-time-unit";
  }

  static std::string toShortString() {
    switch (unit_) {
    case MINUTE:
      return "min";
    case SECOND:
      return "s";
    case MILLISECOND:
      return "ms";
    case UNDEFINED:
      return "";
    default:
      Logger::getInstance().addError("Invalid time unit.").print();
    }

    return "error-time-unit";
  }
};

inline int Time::unit_ = Time::UNDEFINED;

}; // namespace viennaps::units