---
layout: default
title: Fluorocarbon Etching
parent: Pre-Built Models
grand_parent: Process Models
nav_order: 8
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

# Fluorocarbon Etching
{: .fs-9 .fw-500}

```c++
#include <psFluorocarbonEtching.hpp>
```
CPU only
{: .label .label-yellow}
---

Our model assumes that, in any complex plasma etch process, there are three fundamental types of particles:
etchant, depositing polymer particles and ions. Due to the long etch times, compared to surface reaction time scales, we can safely assume that each of these substances’ concentrations will reach a steady state on the surface. Therefore, the surface coverages of all involved particle types $\phi_x$, where $x$ represents etchant (e), polymer (p), etchant on polymer (ep), and ions (i), are expressed by the following equations:
\begin{equation}
  \frac{d \phi_e}{dt}=J_{e} S_{e}\left(1-\phi_{e}-\phi_{p}\right)-k_{i e} J_{i} Y_{ie} \phi_{e}-k_{e v} J_{e v} \phi_{e} \approx 0;
\end{equation}
\begin{equation}
  \frac{d \phi_{p}}{d t}=J_{p} S_{p}-J_{i} Y_{p} \phi_{p} \phi_{p e} \approx 0;
\end{equation}
\begin{equation}
  \frac{d \phi_{p e}}{d t}=J_{e} S_{p e}\left(1-\phi_{p e}\right)-J_{i} Y_{p} \phi_{p e} \approx 0.
\end{equation}
Here, $J_x$ and $S_x$ represent the different particle fluxes and sticking probabilities, respectively. $Y_{ie}$ is the ion-enhanced etching yield for etchant particles, $Y_p$ is the ion-enhanced etching yield on polymer, $Y_{sp}$ gives the physical ion sputtering yield, and $k_{ie}$ and $k_{ev}$ are the stoichiometric factors for ion-enhanced etching and evaporation, respectively, which are determined by the chemical etching reaction. 
By solving these steady state equations for the coverages, one can determine etch or deposition rates on the surface. If deposition of polymer dominates, the surface normal velocity is positive and is given by
\begin{equation}
  v=\frac{1}{\rho_{p}}\left(J_{p} S_{p}-Y_{p} J_{i} \phi_{p e}\right), 
\end{equation}
where $\rho_p$ is the atomic polymer density. 
The first term $J_{p} S_{p}$ gives the rate of polymer particles reaching and adsorbing on the surface, while the second term $Y_{p} J_{i} \phi_{p e}$ describes the removal of polymer by ion-enhanced etching.
Together, these terms describe the deposition of polymer material on the surface, which acts as passivation layer for the chemical etching process. 
If, on the other hand, etching of the substrate dominates, the negative surface velocity of the substrate is given by
\begin{equation}
  v=\frac{1}{\rho_{m}}\left[J_{e v} \phi_{e}+J_{i} Y_{ie} \phi_{e}+J_{i} Y_{sp}\left(1-\phi_{e}\right)\right], 
\end{equation}
where $\rho_m$ is the atomic density of the etched material and depends on which layer in the stack is being etched. 
Each term accounts for a different type of surface reaction.
The first term, $J_{e v} \phi_{e}$, describes the chemical etching process, where etchants bind chemically with the substrate to form volatile etch products which dissolve thermally from the surface. Thus, the evaporation flux $J_{ev}$ is a parameter proportional to the etchant flux $J_e$ and depends on the chemical gas and surface composition and temperature of the etching plasma. It is given by
\begin{equation}
  J_{ev} = K e^{-E_a/k_B T}J_e,
\end{equation}
where $K$ is a process parameter describing the volatility of the chemical etching process, $E_a$ is the activation energy for thermal etching, $k_B$ is the Boltzmann constant, and $T$ is the temperature.
The second term, $J_{i} Y_{ie} \phi_{e}$, describes the contribution of ion-enhanced etching. In this surface reaction, volatile etch products which do not dissolve from the surface thermally, absorb energy from impinging ions and consequently dissolve from the surface. 
Finally, the last term, $J_{i} Y_{sp}\left(1-\phi_{e}\right)$, describes physical sputtering of the substrate by highly energetic ions. 
Since both chemical and ion-enhanced etching involve etchants, they are proportional to the etchant coverage $\phi_e$, while physical ion sputtering takes place directly on the substrate and is thus proportional to the fraction of the surface not covered by the etchant.

## Ions

Each ion is endowed with an initial energy and direction upon creation on the source plane. The assignment of initial energies is governed by a normal distribution, characterized by a mean energy value and an energy sigma, thus allowing for stochastic variations in the initial energy states of the ions. The distribution of initial ion directions is given by a power cosine source distribution, which is defined by the exponent of the power cosine distribution. 

Upon impact with the surface, an energy- and angle-dependent yield is computed, which contributes to the specific surface point's rate. The yield is expressed as:
\begin{equation}
    Y(E,\theta) = A\left(\sqrt{E} - \sqrt{E_{th}}\right)f(\theta),
\end{equation}
where $E$ denotes the particle energy and $\theta$ its incident angle. Here, $A$ represents a yield coefficient, and $E_{\text{th}}$ denotes the material's threshold energy for physical sputtering. The function $f(\theta)$, which characterizes the angle-dependence of the yield. For sputtering, the function is given by:
\begin{equation}
    f(\theta) = (1 + B_{sp}(1-\cos^2(\theta)))\cos(\theta),
\end{equation}
while for ion-enhanced etching, the function is given by:
\begin{equation}
    f(\theta) = \cos(\theta).
\end{equation}

The ions can also reflect from the surface. Their energy loss during reflection is described by the model proposed by Belen et al [^1]. The current ray energy is multiplied by a factor $E_\textrm{ref}$ ($0 \leq E_\textrm{ref} \leq 1$) which depends on the incoming angle $\theta$ in the following way:
\begin{equation}
E_{\textrm{ref}}= 1-(1-A)\frac{\frac{\pi}{2}-\theta}{\frac{\pi}{2}-\theta_\textrm{inflect}} \quad \text{ if } \theta \geqslant \theta_\textrm{inflect}
\end{equation}
\begin{equation}
E_{\textrm{ref}}=\mathrm{A}\left(\frac{\theta}{\theta_\textrm{inflect}}\right)^{n_l} \quad \text{ if } \theta<\theta_\textrm{inflect},
\end{equation}
where $A = (1 + n(\frac{\pi}{2 \theta_\textrm{inflect}} - 1))^{-1}$. 

Ions striking the surface at an angle denoted by $\theta$ relative to the surface normal undergo reflection, where the angular dispersion is characterized by a cosine function centered around the direction of specular reflection defined by $\theta_\textrm{spec}$.
This reflection process distinguishes between ions approaching the surface at glancing angles, which undergo nearly perfect specular reflection, and those striking the surface perpendicularly, which undergo nearly diffuse reflection.

\begin{equation}
    \mathrm{P}(\phi) \propto \cos \left(\frac{\pi}{2} \frac{\phi}{\frac{\pi}{2}-\theta_\textrm{spec}}\right) \quad \text{ if } \theta_\textrm{inc} \leqslant \theta_\textrm{min}
\end{equation}

\begin{equation}
  \mathrm{P}(\phi) \propto \cos \left(\frac{\pi}{2} \frac{\phi}{\frac{\pi}{2}-\theta_\textrm{min}}\right) \quad \text{ if } \theta_{\textrm{inc}}>\theta_{\textrm{min}}
\end{equation}

The ray's reflected direction is randomly chosen from a cone around the specular direction. The opening angle of this cone is given by the incidence angle $\theta$. 

<img src="{% link assets/images/coned_specular.png %}" alt="drawing" width="500"/>

---

## New parameterization (arbitrary materials)

The **FluorocarbonParameters** struct now accepts an arbitrary set of materials.
Each entry has full per-material reaction and transport properties.

```c++
/// C++
template <typename NumericType>
struct FluorocarbonParameters {
  struct MaterialParameters {
    Material id = Material::Undefined;

    // density (1e22 atoms/cm³)
    NumericType density = 2.2;

    // sticking
    NumericType beta_p = 0.26;
    NumericType beta_e = 0.9;

    // sputtering / ion-enhanced terms
    NumericType Eth_sp = 18.; // eV
    NumericType Eth_ie = 4.;  // eV
    NumericType A_sp  = 0.0139;
    NumericType B_sp  = 9.3;
    NumericType A_ie  = 0.0361;

    // chemical etching volatility
    NumericType K   = 0.002789491704544977;
    NumericType E_a = 0.168; // eV
  };

  std::vector<MaterialParameters> materials;

  // fluxes (1e15 /cm² /s)
  NumericType ionFlux     = 56.;
  NumericType etchantFlux = 500.;
  NumericType polyFlux    = 100.;

  NumericType delta_p       = 1.;
  NumericType etchStopDepth = std::numeric_limits<NumericType>::lowest();

  NumericType temperature = 300.; // K
  NumericType k_ie = 2.;
  NumericType k_ev = 2.;

  struct IonType {
    NumericType meanEnergy = 100.; // eV
    NumericType sigmaEnergy = 10.; // eV
    NumericType exponent = 500.;
    NumericType inflectAngle = 1.55334303;
    NumericType n_l = 10.;
    NumericType minAngle = 1.3962634;
  } Ions;

  void addMaterial(const MaterialParameters &m) { materials.push_back(m); }

  MaterialParameters getMaterialParameters(Material m) const; // lookup + error if missing
};
```

**Key changes**

* You can add **any** `Material` from the global enum and set per-material parameters.
* Parameters are applied to the model via the **unified `setParameters(...)`** API on the process/model.
* Direct field mutation on the process or model is deprecated.

{: .note }
Time and length units must be set before initializing the model. See [Units]({% link misc/units.md %}).

---

## Usage

### C++

```c++
// Create model and parameter pack
using T = double;
auto model = SmartPointer<FluorocarbonEtching<T, 3>>::New();

ps::FluorocarbonParameters<T> params;

// Silicon
ps::FluorocarbonParameters<T>::MaterialParameters si;
si.id = ps::Material::Si;
si.density = 5.5;      // example override
params.addMaterial(si);

// SiO2
ps::FluorocarbonParameters<T>::MaterialParameters sio2;
sio2.id = ps::Material::SiO2;
sio2.density = 2.2;
params.addMaterial(sio2);

// Si3N4
ps::FluorocarbonParameters<T>::MaterialParameters si3n4;
si3n4.id = ps::Material::Si3N4;
si3n4.density = 2.3;
params.addMaterial(si3n4);

// Polymer (passivation)
ps::FluorocarbonParameters<T>::MaterialParameters pol;
pol.id = ps::Material::Polymer;
pol.density = 2.0;
pol.beta_e = 0.6;
pol.A_ie = 0.0361 * 2.0;
params.addMaterial(pol);

// Mask
ps::FluorocarbonParameters<T>::MaterialParameters mask;
mask.id = ps::Material::Mask;
mask.density = 500.0;
mask.beta_e = 0.1;
mask.beta_p = 0.01;
mask.Eth_sp = 20.0;
params.addMaterial(mask);

// Global flux and ion settings (optional overrides)
params.ionFlux = 56.;
params.etchantFlux = 500.;
params.polyFlux = 100.;

// Apply parameters via unified API
model->setParameters(params);
```

### Python

```python
import viennaps as vps

model = vps.FluorocarbonEtching()

params = vps.FluorocarbonParameters()

matSi = vps.FluorocarbonMaterialParameters()
matSi.id = vps.Material.Si
matSi.density = 5.5
params.addMaterial(matSi)

matSiO2 = vps.FluorocarbonMaterialParameters()
matSiO2.id = vps.Material.SiO2
matSiO2.density = 2.2
params.addMaterial(matSiO2)

matSi3N4 = vps.FluorocarbonMaterialParameters()
matSi3N4.id = vps.Material.Si3N4
matSi3N4.density = 2.3
params.addMaterial(matSi3N4)

matPoly = vps.FluorocarbonMaterialParameters()
matPoly.id = vps.Material.Polymer
matPoly.density = 2.0
matPoly.beta_e = 0.6
matPoly.A_ie = 0.0361 * 2.0
params.addMaterial(matPoly)

matMask = vps.FluorocarbonMaterialParameters()
matMask.id = vps.Material.Mask
matMask.density = 500.0
matMask.beta_e = 0.1
matMask.beta_p = 0.01
matMask.Eth_sp = 20.0
params.addMaterial(matMask)

# Optional global settings
params.ionFlux = 56.0
params.etchantFlux = 500.0
params.polyFlux = 100.0

# Apply via unified API
model.setParameters(params)
```

---

## Notes and tips

* Add only the materials that appear in your stack. Unlisted materials will use defaults unless you define them.
* Use the global `Material` enum introduced in the **Material Mapping** docs to ensure consistent IDs.
* For multi-material stacks, tuning (K, E_a), thresholds, and sticking allows selective etching vs passivation.
* The model works with any flux engine. Choose via `setFluxEngineType(...)` on the process using this model.

## Related examples

* [Stack Etching](https://github.com/ViennaTools/ViennaPS/tree/master/examples/stackEtching)

