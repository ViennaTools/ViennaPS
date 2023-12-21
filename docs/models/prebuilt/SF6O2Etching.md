---
layout: default
title: SF<sub>6</sub>O<sub>2</sub> Etching
parent: Pre-Built Models
grand_parent: Process Models
nav_order: 6
sf6o2: SF<sub>6</sub>O<sub>2</sub>
---
<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']]
  }
};
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
</script>

# {{ page.sf6o2 }} Etching
{: .fs-9 .fw-500}

```c++
#include <psSF6O2Etching.hpp>
```
---

The model for {{ page.sf6o2 }} etching of silicon is based on a model by Belen et al. [^1] and summarized here. For implementation details refer to [here](#implementation).
To describe the feature scale simulation of {{ page.sf6o2 }} plasma etching, the surface rates of both ions and neutral particles, specifically fluorine and oxygen, and the surface coverages of neutrals, are considered. Ray tracing is used for the calculation of the surface rates, which are used for calculating coverages during each time step.
The etch rate is determined the three physical phenomena:
- Chemical etching
- Physical sputtering
- Ion-enhanced etching

In the process of chemical etching, the fluorine from the SF<sub>6</sub> species reacts with the exposed silicon surface.
Physical sputtering is caused by high-energy ions impacting the surface. Due to an applied bias, ions strike the wafer surface with a high enough kinetic energy, $E_{ion} > E_{th}$ to break the existing bonds in the silicon wafer or other exposed materials.
Lastly, ion-enhanced etching, also known as reactive ion etching (RIE), combines the two previous effects. Since silicon surfaces that are saturated with fluorine are more prone to physical sputtering, the threshold energy for releasing the silicon atom $E_{th}$ is significantly reduced compared to non-fluorinated surfaces. Therefore, ion-enhanced etching provides an etch rate that is larger than the sum of the chemical etching and sputtering.

The surface can be covered in fluorine or oxygen. The physical model keeps track of these coverages, given by $\theta_F$ and $\theta_O$, respectively, by calculating the flux-induced rates and considering the coverages from the previous time step. They are calculated with Langmuir–Hinshelwood-type surface site balance equations, given by:

\begin{equation}
    \sigma_{Si}\cfrac{d\theta_{F}}{dt}=\gamma_{F}\Gamma_{F}\left(1-\theta_{F}-\theta_{O}\right)-k\sigma_{Si}\theta_{F}-2Y_{ie}\Gamma_{i}\theta_{F}
    \label{equ:thetaF}
\end{equation}

\begin{equation}
    \sigma_{Si}\cfrac{d\theta_{O}}{dt}=\gamma_{O}\Gamma_{O}\left(1-\theta_{F}-\theta_{O}\right)-\beta\sigma_{Si}\theta_{O}-Y_{O}\Gamma_{i}\theta_{O}
    \label{equ:thetaO}
\end{equation}

The term $\sigma_{Si}$ represents the density of silicon at the surface point $\vec{x}$ which is not included in the equations for legibility; $\Gamma_F$, $\Gamma_O$, and $\Gamma_i$ are the emitted fluorine, oxygen, and ion fluxes, respectively; $\gamma_F$ and $\gamma_O$ are the sticking coefficients for fluorine and oxygen on a non-covered silicon substrate, respectively; $k$ is the chemical etch reaction rate constant; $\beta$ is the oxygen recombination rate constant; and $Y_{ie}$ and $Y_O$ are the total ion-enhanced and oxygen etching yields, respectively. $Y_{ie}$ and $Y_O$ are yield functions that are dependent on the ion energies in the reactor. 

Since the surface movement is significantly smaller than the considered fluxes, it can be assumed that it does not impact the calculation. With this assumption of a pseudo-steady-state, the coverage equations can be set equal to zero, resulting in the following surface coverage equations:
\begin{equation}
    \theta_{F}=\left[1+\left(\cfrac{k\sigma_{Si}+2Y_{ie}\Gamma_{i}}{\gamma_{F}\Gamma_{F}}\right)\left(1+\cfrac{\gamma_{O}\Gamma_{O}}{\beta\sigma_{Si}+Y_{O}\Gamma_{i}}\right)\right]^{-1}
\end{equation}

\begin{equation}
    \theta_{O}=\left[1+\left(\cfrac{\beta\sigma_{Si}+Y_{ie}\Gamma_{i}}{\gamma_{O}\Gamma_{O}}\right)\left(1+\cfrac{\gamma_{F}\Gamma_{F}}{k\sigma_{Si}+2Y_{ie}\Gamma_{i}}\right)\right]^{-1}
\end{equation}

The reason that pseudo-steady-state can be assumed is that the incoming fluxes of all involved particles are in the order of 10$^{16}$--10$^{19}$\,cm<sup>-1</sup>s<sup>-1</sup>, which is significantly larger than the surface etch rate ER, which is typically in the range of several nanometers per second. The oxygen particles do not take part in surface removal; instead, they occupy an area on the top surface layer and inhibit the effects of chemical etching by fluorine. Relating it to the parameters in the equation, the presence of oxygen (denoted by its flux $\Gamma_{O}$) tends to reduce $\theta_{F}$. Increasing the oxygen flux $\Gamma_O$ increases the overall expression in the square brackets, which means $\theta_{F}$ decreases. Since oxygen has a passivating effect, the etching of silicon proceeds only due to its reaction with fluorine and physical sputtering due to the incoming ion flux. At locations where oxygen coverage is high, only ion sputtering takes place. This brings us to the expression for the etch rate (ER), which is used to move the surface

\begin{equation}
    \textrm{ER}=\cfrac{1}{\rho_{Si}}\left(\cfrac{k\sigma_{Si}\theta_{F}}{4}+Y_{p}\Gamma_{i}+Y_{ie}\Gamma_{i}\theta_{F}\right),
\end{equation}
where $\rho_{Si}$ is the silicon density. The first, second, and third terms in the brackets of the etch rate equation represent the chemical etching, physical sputtering, and ion-enhanced etching, respectively.

## Implementation

```c++
psSF6O2Etching(const double ionFlux, const double etchantFlux,
                const double oxygenFlux, const NumericType meanEnergy /* eV */,
                const NumericType sigmaEnergy /* eV */, 
                const NumericType ionExponent = 100.,
                const NumericType oxySputterYield = 2.,
                const NumericType etchStopDepth =
                    std::numeric_limits<NumericType>::lowest())
```

| Parameter           | Description                                                               | Type           |
|----------------------|--------------------------------------------------------------------------|----------------|
| `ionFlux`           | Ion flux for the {{ page.sf6o2 }} etching process.                                   | `double`       |
| `etchantFlux`       | Etchant flux for the {{ page.sf6o2 }} etching process.                               | `double`       |
| `oxygenFlux`        | Oxygen flux for the {{ page.sf6o2 }} etching process.                                | `double`       |
| `meanEnergy`        | Mean energy of particles in electronvolts (eV).                           | `NumericType`  |
| `sigmaEnergy`       | Energy distribution standard deviation in electronvolts (eV).             | `NumericType`  |
| `ionExponent`       | (Optional) Exponent power cosine source distribution of the ions. Default is set to 100.       | `NumericType`  |
| `oxySputterYield`   | (Optional) Oxygen sputtering yield. Default is set to 2.                  | `NumericType`  |
| `etchStopDepth`     | (Optional) Depth at which etching should stop. Default is negative infinity.| `NumericType`  |

{: .note}
> All flux values are units 10<sup>16</sup> / cm<sup>2</sup>.

## Related Examples

* [Hole Etching](https://github.com/ViennaTools/ViennaPS/tree/master/examples/holeEtching)

---

[^1]: Rodolfo Jun Belen, Sergi Gomez, Mark Kiehlbauch, David Cooperberg, Eray S. Aydil; Feature-scale model of Si etching in SF<sub>6</sub> plasma and comparison with experiments. J. Vac. Sci. Technol. A 1 January 2005; 23 (1): 99–113. [https://doi.org/10.1116/1.1830495](https://doi.org/10.1116/1.1830495)