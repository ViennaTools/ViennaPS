import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # ViennaPS - Process Simulation Library

        <div style="text-align: center;">
            <img src="https://viennatools.github.io/ViennaPS/assets/images/logo.png" width="150" alt="ViennaPS Logo"/>
        </div>

        ---

        **ViennaPS** is a header-only C++ library for **topography simulation** in microelectronic fabrication processes. It models the evolution of 2D and 3D surfaces during etching, deposition, and related steps.

        <img src="https://viennatools.github.io/ViennaPS/assets/images/banner.png" width="100%" alt="ViennaPS Banner"/>
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Key Features

        | Feature | Description |
        |---------|-------------|
        | **Level-Set Methods** | Advanced surface evolution tracking |
        | **Monte Carlo Ray Tracing** | Accurate flux calculation |
        | **2D & 3D Support** | Full dimensional flexibility |
        | **Physical Models** | Realistic process simulation |
        | **Fast Emulation** | Quick approximation approaches |
        | **Python Bindings** | Easy integration with Python workflows |
        | **GPU Acceleration** | OptiX-based GPU ray tracing |

        ### Core Framework (v4.0.0)

        - **Modular Flux Engine**: `AUTO`, `CPU_DISK`, `GPU_DISK`, `GPU_LINE`, `GPU_TRIANGLE`
        - **Unified Python Package**: `viennaps` with `viennaps.d2` and `viennaps.d3` modules
        - **Extended Material List**: Common semiconductor materials included
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## Process Simulation Examples

        ViennaPS provides a comprehensive set of example simulations demonstrating various fabrication processes.
        """
    )
    return


@app.cell
def _(mo):
    # Create tabs for different example categories
    etching_examples = mo.md(
        r"""
        ### Etching Processes

        <table>
        <tr>
        <td style="text-align: center; padding: 20px;">
            <img src="https://viennatools.github.io/ViennaPS/assets/examples/holeEtching.png" width="300"/><br/>
            <b>Hole Etching</b><br/>
            SF₆/O₂ plasma etching with ion bombardment
        </td>
        <td style="text-align: center; padding: 20px;">
            <img src="https://viennatools.github.io/ViennaPS/assets/examples/BoschProcess.png" width="300"/><br/>
            <b>Bosch Process</b><br/>
            Deep reactive ion etching (DRIE)
        </td>
        </tr>
        <tr>
        <td style="text-align: center; padding: 20px;">
            <img src="https://viennatools.github.io/ViennaPS/assets/examples/wetEtching.png" width="300"/><br/>
            <b>Cantilever Wet Etching</b><br/>
            Crystallographic direction influence
        </td>
        <td style="text-align: center; padding: 20px;">
            <img src="https://viennatools.github.io/ViennaPS/assets/examples/stackEtching.png" width="300"/><br/>
            <b>Stack Etching</b><br/>
            Multi-layer material etching
        </td>
        </tr>
        <tr>
        <td style="text-align: center; padding: 20px;">
            <img src="https://viennatools.github.io/ViennaPS/assets/examples/IBE.png" width="300"/><br/>
            <b>Ion Beam Etching</b><br/>
            Directional ion bombardment
        </td>
        <td style="text-align: center; padding: 20px;">
            <img src="https://viennatools.github.io/ViennaPS/assets/examples/blazedGrating.png" width="300"/><br/>
            <b>Blazed Gratings Etching</b><br/>
            Angled surface patterning
        </td>
        </tr>
        </table>
        """
    )

    deposition_examples = mo.md(
        r"""
        ### Deposition Processes

        <table>
        <tr>
        <td style="text-align: center; padding: 20px;">
            <img src="https://viennatools.github.io/ViennaPS/assets/examples/trenchDepo.png" width="300"/><br/>
            <b>Trench Deposition</b><br/>
            Particle deposition with varying sticking probability
        </td>
        <td style="text-align: center; padding: 20px;">
            <img src="https://viennatools.github.io/ViennaPS/assets/examples/TEOS.png" width="300"/><br/>
            <b>TEOS Trench Deposition</b><br/>
            Plasma-enhanced CVD process
        </td>
        </tr>
        <tr>
        <td style="text-align: center; padding: 20px;">
            <img src="https://viennatools.github.io/ViennaPS/assets/examples/ALD.png" width="300"/><br/>
            <b>Atomic Layer Deposition</b><br/>
            Self-limiting surface reactions
        </td>
        <td style="text-align: center; padding: 20px;">
            <img src="https://viennatools.github.io/ViennaPS/assets/examples/sputterDepo.png" width="300"/><br/>
            <b>Sputter Deposition</b><br/>
            Physical vapor deposition
        </td>
        </tr>
        </table>
        """
    )

    special_examples = mo.md(
        r"""
        ### Special Processes

        <table>
        <tr>
        <td style="text-align: center; padding: 20px;">
            <img src="https://viennatools.github.io/ViennaPS/assets/examples/epitaxy.png" width="300"/><br/>
            <b>Selective Epitaxy</b><br/>
            SiGe growth on Si substrate
        </td>
        <td style="text-align: center; padding: 20px;">
            <img src="https://viennatools.github.io/ViennaPS/assets/examples/FinFET.png" width="300"/><br/>
            <b>Process Emulation</b><br/>
            Fast FinFET process approximation
        </td>
        </tr>
        <tr>
        <td style="text-align: center; padding: 20px;">
            <img src="https://viennatools.github.io/ViennaPS/assets/examples/DRAM.png" width="300"/><br/>
            <b>DRAM Wiggling</b><br/>
            High aspect ratio structure deformation
        </td>
        <td style="text-align: center; padding: 20px;">
            <img src="https://viennatools.github.io/ViennaPS/assets/examples/faradayCageEtching.png" width="300"/><br/>
            <b>Faraday Cage Etching</b><br/>
            Ion trajectory modification
        </td>
        </tr>
        </table>
        """
    )

    mo.ui.tabs({
        "Etching": etching_examples,
        "Deposition": deposition_examples,
        "Special": special_examples
    })
    return deposition_examples, etching_examples, special_examples


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## Physical Process Models

        ViennaPS includes several pre-built physical models for common fabrication processes.

        ### TEOS Plasma-Enhanced CVD

        The TEOS PE CVD process models deposition with plasma-enhanced ion bombardment:

        $$v = R_{\text{rad}} \cdot \Gamma_{\text{rad}}^{o_{\text{rad}}} + R_{\text{ion}} \cdot \Gamma_{\text{ion}}^{o_{\text{ion}}}$$

        Where:
        - $R_{\text{rad}}, R_{\text{ion}}$ are the rates of radicals and ions
        - $\Gamma_{\text{rad}}, \Gamma_{\text{ion}}$ are the fluxes
        - $o_{\text{rad}}, o_{\text{ion}}$ are the reaction orders

        | Parameter | Description | Default |
        |-----------|-------------|---------|
        | `radicalSticking` | Sticking probability of TEOS radicals | 1.0 |
        | `radicalRate` | Rate of TEOS radicals | 1.0 |
        | `ionRate` | Rate of ions | 1.0 |
        | `ionExponent` | Power cosine source distribution exponent | 1.0 |
        | `ionSticking` | Sticking probability of ions | 1.0 |
        | `radicalOrder` | Reaction order of radicals | 1.0 |
        | `ionOrder` | Reaction order of ions | 1.0 |
        | `ionMinAngle` | Minimum specular reflection angle | 0.0 |
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## Simulation Results Gallery

        ### Trench Deposition Process
        <img src="https://raw.githubusercontent.com/ViennaTools/ViennaPS/master/assets/deposition.png" width="100%"/>

        *Particle deposition within a trench geometry showing variations in particle sticking probability*

        ---

        ### SF₆/O₂ Hole Etching Results
        <img src="https://raw.githubusercontent.com/ViennaTools/ViennaPS/master/assets/sf6o2_results.png" width="100%"/>

        *Hole etching process with SF₆/O₂ plasma chemistry showing different flux configurations*

        ---

        ### Bosch Process Comparison
        <img src="https://raw.githubusercontent.com/ViennaTools/ViennaPS/master/assets/bosch_process.png" width="100%"/>

        *Deep reactive ion etching: process emulation vs. simple simulation vs. physical simulation*

        ---

        ### Wet Etching - Cantilever Structure
        <img src="https://raw.githubusercontent.com/ViennaTools/ViennaPS/master/assets/wet_etching.png" width="100%"/>

        *Wet etching process showing crystallographic direction influence on cantilever formation*

        ---

        ### Selective Epitaxy
        <img src="https://raw.githubusercontent.com/ViennaTools/ViennaPS/master/assets/epitaxy.png" width="100%"/>

        *SiGe growth on Si substrate with crystallographic direction influence*

        ---

        ### GDS Mask Import
        <img src="https://raw.githubusercontent.com/ViennaTools/ViennaPS/master/assets/masks.png" width="100%"/>

        *GDS mask import, blurring, rotation, scaling, flipping, proximity correction, and extrusion*
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## Installation

        ### Python (Recommended)

        ```bash
        pip install ViennaPS
        ```

        ### C++ with CMake (using CPM)

        ```cmake
        cmake_minimum_required(VERSION 3.15)
        project(MyViennaPSProject)

        include(cmake/CPM.cmake)

        CPMAddPackage(
            NAME ViennaPS
            VERSION 4.2.1
            GIT_REPOSITORY "https://github.com/ViennaTools/ViennaPS.git")

        add_executable(my_executable main.cpp)
        target_link_libraries(my_executable PRIVATE ViennaTools::ViennaPS)
        ```

        ### Dependencies

        - **VTK**: `sudo apt install libvtk9.1 libvtk9-dev` (Linux) or `brew install vtk` (macOS)
        - **Embree**: `sudo apt install libembree-dev` (Linux) or `brew install embree` (macOS)

        ### Supported Platforms

        - Windows (Visual Studio)
        - Linux (g++ / clang)
        - macOS (XCode)
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## Python Usage Example

        ```python
        import viennaps as vps

        # Set dimension (2D or 3D)
        vps.setDimension(3)

        # Import from 3D module
        from viennaps.d3 import Domain, Process, TEOSPECVD

        # Create domain and geometry
        domain = Domain()
        domain.makeTrench(width=5.0, height=10.0, depth=20.0)

        # Configure TEOS PE CVD process
        model = TEOSPECVD(
            radicalSticking=0.1,
            radicalRate=1.0,
            ionRate=0.5,
            ionExponent=100.0
        )

        # Run process simulation
        process = Process()
        process.setDomain(domain)
        process.setProcessModel(model)
        process.setProcessDuration(10.0)
        process.apply()

        # Save result
        domain.saveSurfaceMesh("result.vtp")
        ```
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## Resources

        - **Documentation**: [viennatools.github.io/ViennaPS](https://viennatools.github.io/ViennaPS/)
        - **GitHub Repository**: [github.com/ViennaTools/ViennaPS](https://github.com/ViennaTools/ViennaPS)
        - **Contact**: [viennatools@iue.tuwien.ac.at](mailto:viennatools@iue.tuwien.ac.at)

        ---

        *Developed at the [Institute for Microelectronics](http://www.iue.tuwien.ac.at/) at TU Wien*

        *Licensed under the MIT License*
        """
    )
    return


if __name__ == "__main__":
    app.run()
