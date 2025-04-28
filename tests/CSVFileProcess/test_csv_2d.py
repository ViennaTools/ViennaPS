import numpy as np
import os
import viennaps2d as vps

vps.Logger.setLogLevel(vps.LogLevel.WARNING)

def write_csv(filename, etch=False):
    rng = np.random.default_rng(42)
    x = np.linspace(-50, 50, 51)
    rates = rng.uniform(1.0, 2.0, size=x.shape)
    if etch:
        rates = -rates
    np.savetxt(filename, np.column_stack((x, rates)), delimiter=",", comments="")

def run2D():
    for etch in [False, True]:
        write_csv("rates2D.csv", etch)
        for mode in ["linear", "idw", "custom"]:
            domain = vps.Domain()
            vps.MakeTrench(
                domain=domain,
                gridDelta=1.0,
                xExtent=10.0,
                yExtent=10.0,
                trenchWidth=4.0,
                trenchDepth=5.0,
                taperingAngle=0.0,
                baseHeight=0.0,
                periodicBoundary=False,
                makeMask=etch,
                material=vps.Material.Si,
            ).apply()

            if not etch:
                domain.duplicateTopLevelSet(vps.Material.SiO2)

            model = vps.CSVFileProcess("rates2D.csv", direction=[0.0, -1.0, 0.0], offset=[0.0, 0.0])
            print(f"Testing {mode} interpolation | Etch: {etch}")
            model.setInterpolationMode(mode)
            if mode == "custom":
                model.setCustomInterpolator(lambda coord: 1.0 + 0.5 * np.sin(coord[0]))
            elif mode == "idw":
                model.setIDWNeighbors(4)

            vps.Process(domain, model, 1.0).apply()
        os.remove("rates2D.csv")

    print("2D CSVFileProcess tests passed")

if __name__ == "__main__":
    run2D()
