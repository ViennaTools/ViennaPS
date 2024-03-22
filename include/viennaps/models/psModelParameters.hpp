#pragma once

#include <cmath>

namespace psParameters {

static constexpr double kB = 8.617333262 * 1e-5; // eV / K
static constexpr double roomTemperature = 300.;  // K

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

} // namespace psParameters
