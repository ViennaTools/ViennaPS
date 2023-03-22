# Configuration File Documentation

The configuration file must obey a certain structure in order to be parsed correctly. An example for a configuration file can be seen in **SampleConfig.txt**. The configuration file is parsed line by line and each succesfully parsed line is executed immediately.

## Commands
Every line has to start with a command statement, followed by a list of parameters for the command. Possible commands are:

- **INIT**: Initialize the simulation domain.
- **GEOMETRY** _\<GeometryType>_: Create or import a geometry. Possible options for creating geometries are: _Trench_, _Hole_ and _Plane_. It is also possible to import geometries from _.lvst_ files by specifying _Import_ or to import layers from GDSII file format by secifying _GDS_ (only possible in 3D mode). Parameters for the geometries are described below.
- **PROCESS** _\<ProcessType>_: Run a process. Possible process types are: _Deposition_, _GeometricUniformDeposition_, _SF6O2Etching_ and _DirectionalEtching_. Parameters for the processes are described below.
- **PLANARIZE**: Planarize the geometry at a given height.
- **OUTPUT** _\<fileName>_: Print the surface of the geometry in _fileName.vtp_ file format.

## Parameters

Every parameter is specified by _paramName=paramValue_ (there must not be any blank space before or after the _=_ sign) after the command statement.
All parameters which are parsed additional to a command are described below. For any parameter which is not specified, a default value is assigned.

---
**INIT**
<dl>
  <dt>xExtent</dt>
  <dd>width of the domain in x-direction (numeric value, default: 1)</dd>
  <dt>yExtent</dt>
  <dd>width of domain in y-direction (this is ignored in 2D mode, numeric value, default: 1)</dd>
  <dt>resolution</dt>
  <dd>distance between grid points in the domain (numeric value, default: 0.02)</dd>
  <dt>printIntermediate</dt>
  <dd>output intermediate disk meshes for each process step (boolean, default: 0)</dd>
  <dt>periodic</dt>
  <dd>use periodic boundary conditions (boolean, default: 0)</dd>
</dl>

---
**GEOMETRY Trench**
<dl>
  <dt>width</dt>
  <dd> width of the trench (numeric value, default: 0.2)</dd>
  <dt>depth</dt>
  <dd>depth of the trench (numeric value, default: 0.5)</dd>
  <dt>zPos</dt>
  <dd>offset of the trench in z-direction (y-direction in 2D mode, numeric value, default: 0)</dd>
  <dt>tapering</dt>
  <dd>tapering angle of the trench in degrees (numeric value, default: 0)</dd>
  <dt>mask</dt>
  <dd>creates a mask from the trench, such that the bottom represents a different material and a selective etching or deposition process can be used (boolean, default: 0)
</dl>

---
**GEOMETRY Hole**
<dl>
  <dt>radius</dt>
  <dd>radius of the hole (numeric value, default: 0.2)</dd>
  <dt>depth</dt>
  <dd>depth of the hole (numeric value, default: 0.2)</dd>
  <dt>zPos</dt>
  <dd>offset of the hole in z-direction (y-direction in 2D mode, numeric value, default: 0)</dd>
  <dt>tapering</dt>
  <dd>tapering angle of the hole in degrees (numeric value, default: 0)</dd>
  <dt>mask</dt>
  <dd>creates a mask from the hole, such that the bottom represents a different material and a selective etching or deposition process can be used (boolean, default: 0)
</dl>

---
**GEOMETRY Plane**
<dl>
  <dt>zPos</dt>
  <dd>offset of the plane in z-direction (y-direction in 2D mode, numeric value, default: 0)</dd>
</dl>

---
**GEOMETRY Import**
<dl>
  <dt>file</dt>
  <dd>file name of ViennaLS geometry files. The file name is assumed to be appended with "_layer(i).lvst" (string, no default value) </dd>
  <dt>layers</dt>
  <dd>number of layers to read (integer value, default: 0)</dd>
</dl>

---
**GEOMETRY GDS**
<dl>
  <dt>file</dt>
  <dd>file name of GDSII file (string, no default value) </dd>
  <dt>layer</dt>
  <dd>specify which layer of the file should be read (integer value, default: 0)</dd>
  <dt>maskHeight</dt>
  <dd>height of the layer in z-direction (numeric value, default: 0.1)</dd>
  <dt>zPos</dt>
  <dd>offset of the layer in z-direction (numeric value, default: 0)</dd>
  <dt>invert</dt>
  <dd>invert the mask (boolean, default: 0)</dd> 
  <dt>xPadding</dt>
  <dd>padding on the boundary of the mask in x-direction (numeric value, default: 0)</dd> 
  <dt>yPadding</dt>
  <dd>padding on the boundary of the mask in y-direction (numeric value, default: 0)</dd> 
  <dt>pointOrder</dt>
  <dd>store points order in GDS file. Can vary depending on what GDS editor is used to create to file. If geometry can not be read (timeout error), try changing the value to 1 (boolean, default: 0)</dd>
</dl>

---
**PROCESS Deposition**
<dl>
  <dt>time</dt>
  <dd>process time (numeric value, default: 1)</dd>
  <dt>rate</dt>
  <dd>deposition rate (numeric value, default: 1)</dd>
  <dt>sticking</dt>
  <dd>sticking coefficient (numeric value, default: 1)</dd>
  <dt>cosineExponent</dt>
  <dd>Exponent for cosine distribution of initial ray directions (numeric value, default: 1)</dd>
  <dt>raysPerPoint</dt>
  <dd>number of rays traced per grid point in the surface geometry (integer value, default: 3000)</dd>
</dl>

---
**PROCESS SphereDistribution**
<dl>
  <dt>radius</dt>
  <dd>radius used for sphere distribution (numeric value, default: 1, can be negative for etching)</dd>
</dl>

---
**PROCESS BoxDistribution**
<dl>
  <dt>halfAxisX</dt>
  <dd>half the width of the box in x-direction (numeric value, default: 1, can be negative for etching)</dd>
  <dt>halfAxisY</dt>
  <dd>half the width of the box in y-direction (numeric value, default: 1, can be negative for etching)</dd>
  <dt>halfAxisZ</dt>
  <dd>half the width of the box in z-direction (numeric value, default: 1, can be negative for etching)</dd>
</dl>

---
**PROCESS SF6O2Etching**
<dl>
  <dt>time</dt>
  <dd>process time (numeric value, default: 1)</dd>
  <dt>ionFlux</dt>
  <dd>total flux of ions in plasma (numeric value, default: 2e16)</dd>
  <dt>ionEnergy</dt>
  <dd>mean ion energy (numeric value, default: 100)</dd>
  <dt>etchantFlux</dt>
  <dd>total flux of etchant species in plasma (numeric value, default: 4.5e16)</dd>
  <dt>oxygenFlux</dt>
  <dd>total flux of oxygen in plasma (numeric value, default: 1e18)</dd>
  <dt>A_O</dt>
  <dd>factor for ion etching yield on oxygen (numeric value, default: 3)</dd>
  <dt>raysPerPoint</dt>
  <dd>number of rays traced per grid point in the surface geometry (integer value, default: 3000)</dd>
  <dt>maskId</dt>
  <dd>ID of mask material (integer value, default: 0)</dd> 
</dl>

---
**PROCESS DirectionalEtching**
<dl>
  <dt>time</dt>
  <dd>process time (numeric value, default: 1)</dd>
  <dt>direction</dt>
  <dd>primal direction of etching (string, default: negZ, negY in 2D mode) 
  <dt>directionalRate</dt>
  <dd>etching rate in primal direction (numeric value, default: 1)</dd>
  <dt>isotropicRate</dt>
  <dd>isotropic etching rate (numeric value, default: 0)</dd>
  <dt>maskId</dt>
  <dd>ID of mask material (integer value, default: 0)</dd> 
</dl>

---
**PROCESS Isotropic**
<dl>
  <dt>time</dt>
  <dd>process time (numeric value, default: 1)</dd>
  <dt>rate</dt>
  <dd>process rate, can be negative for etching (numeric value, default: 0) 
  <dt>maskId</dt>
  <dd>ID of mask material (integer value, default: 0)</dd> 
</dl>

---
**PROCESS WetEtching**
Wet etching process in 30% KOH solution at 70°C.
<dl>
  <dt>time</dt>
  <dd>process time (numeric value, default: 1)</dd>
  <dt>maskId</dt>
  <dd>ID of mask material (integer value, default: 0)</dd> 
</dl>

---
**PLANARIZE**
<dl>
  <dt>height</dt>
  <dd>height of planarization on the z-axis (y-axis in 2D, numeric value, default: 0)</dd>
</dl>

---
