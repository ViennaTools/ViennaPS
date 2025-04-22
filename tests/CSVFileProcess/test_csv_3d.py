import numpy as np
import os
import viennaps3d as vps

vps.Logger.setLogLevel(vps.LogLevel.WARNING)

def write_csv(filename, etch=False):
    rng = np.random.default_rng(42)
    x = np.linspace(-50, 50, 51)
    y = np.linspace(-50, 50, 51)
    xx, yy = np.meshgrid(x, y)
    rates = rng.uniform(1.0, 2.0, size=xx.shape)
    if etch:
        rates = -rates
    np.savetxt(filename, np.column_stack((xx.ravel(), yy.ravel(), rates.ravel())),
               delimiter=",", comments="")

def run3D():
    for etch in [False, True]:
        write_csv("rates3D.csv", etch)
        for mode in ["linear", "idw", "custom"]:
            domain = vps.Domain()
            vps.MakeHole(
                domain=domain,
                gridDelta=1.0,
                xExtent=10.0,
                yExtent=10.0,
                holeRadius=2.0,
                holeDepth=5.0,
                taperingAngle=0.0,
                baseHeight=0.0,
                periodicBoundary=False,
                makeMask=etch,
                material=vps.Material.Si,
                holeShape=vps.HoleShape.Full,
            ).apply()

            if not etch:
                domain.duplicateTopLevelSet(vps.Material.SiO2)

            model = vps.CSVFileProcess("rates3D.csv", direction=[0.0, 0.0, -1.0], offset=[0.0, 0.0])

            print(f"Testing {mode} interpolation | Etch: {etch}")
            model.setInterpolationMode(mode)
            if mode == "custom":
                model.setCustomInterpolator(lambda coord: 1.0 + 0.5 * np.sin(coord[0] + coord[1]))
            elif mode == "idw":
                model.setIDWNeighbors(8)

            vps.Process(domain, model, 1.0).apply()
        os.remove("rates3D.csv")

    print("3D CSVFileProcess tests passed")

if __name__ == "__main__":
    run3D()
