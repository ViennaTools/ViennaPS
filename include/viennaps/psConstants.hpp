#pragma once

#include <cmath>

namespace viennaps {

namespace constants {

static constexpr double kB = 8.617333262 * 1e-5; // eV / K
static constexpr double roomTemperature = 300.;  // K
static constexpr double N_A = 6.0221367e3; //  Avogadro's number in 10^20 mol^-1
static constexpr double R = 8.314;         // Ideal gas constant in J/(mol K)

inline double torrToPascal(double torr) { return torr * 133.322; }

inline double celsiusToKelvin(double celsius) { return celsius + 273.15; }

// p: pressure in torr
// T: temperature in Celsius
// d: diameter of the gas molecule in angstroms
// Result: mean free path in um
inline double gasMeanFreePath(double p, double T, double d) {
  T = constants::celsiusToKelvin(T);
  p = constants::torrToPascal(p);
  return constants::R * T /
         (std::sqrt(2.) * M_PI * d * d * constants::N_A * p) * 1e6; // in um
}

// T: temperature in Celsius
// m: molar mass in amu
// Result: mean velocity in um/s
inline double gasMeanThermalVelocity(double T, double m) {
  T = constants::celsiusToKelvin(T);
  m = m * 1e-3;                                               // amu to kg / mol
  return std::sqrt(8. * constants::R * T / (M_PI * m)) * 1e6; // in um/s
}

namespace Si {

// sputtering coefficients in Ar
static constexpr double Eth_sp_Ar = 20.; // eV
static constexpr double Eth_sp_Ar_sqrt = 4.47213595499958;
static constexpr double A_sp = 0.0337;
static constexpr double B_sp = 9.3;

// chemical etching
static constexpr double K = 0.029997010728956663;
static constexpr double E_a = 0.108; // eV

// density
static constexpr double rho = 5.02; // 1e22 atoms/cm³

} // namespace Si

namespace SiO2 {

// sputtering coefficients in Ar
static constexpr double Eth_sp_Ar = 18.; // eV
static constexpr double Eth_sp_Ar_sqrt = 4.242640687119285;
static constexpr double A_sp = 0.0139;

// chemical etching
static constexpr double K = 0.002789491704544977;
static constexpr double E_a = 0.168; // eV

// density
static constexpr double rho = 2.3; // 1e22 atoms/cm³

} // namespace SiO2

namespace Polymer {
// density
static constexpr double rho = 2; // 1e22 atoms/cm³
} // namespace Polymer

namespace Si3N4 {
// density
static constexpr double rho = 2.3; // 1e22 atoms/cm³
} // namespace Si3N4

namespace TiN {
// density
static constexpr double rho = 5.0804; // 1e22 atoms/cm³
} // namespace TiN

namespace Cl {
// density
static constexpr double rho = 0.0042399; // 1e22 atoms/cm³
} // namespace Cl

namespace Mask {
// density
static constexpr double rho = 500.; // 1e22 atoms/cm³
} // namespace Mask

namespace Ion {
static constexpr double inflectAngle = 1.55334303;
static constexpr double n_l = 10.;
static constexpr double A = 1. / (1. + n_l * (M_PI_2 / inflectAngle - 1.));
static constexpr double minAngle = 1.3962634;
} // namespace Ion

} // namespace constants
} // namespace viennaps
