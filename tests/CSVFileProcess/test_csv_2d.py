import numpy as np
import os
import viennaps2d as vps

vps.Logger.setLogLevel(vps.LogLevel.WARNING)

def write_csv(filename, etch=False):
    rng = np.random.default_rng(42)
    x = np.linspace(-50, 50, 101)
    rates = rng.uniform(1.0, 2.0, size=x.shape)
    if etch:
        rates = -rates
        filename = filename + "_etch.csv"
    else:
        filename = filename + "_deposit.csv"
    np.savetxt(filename, np.column_stack((x, rates)), delimiter=",", header="x,rate", comments="")

def run2D():
    for etch in [False, True]:
        write_csv("rates2D", etch)
        for custom in [False, True]:
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
                model = vps.CSVFileProcess("rates2D_deposit.csv", direction=[0.0, -1.0, 0.0], offset=[0.0])
            else:
                model = vps.CSVFileProcess("rates2D_etch.csv", direction=[0.0, -1.0, 0.0], offset=[0.0])

            if custom:
                model.setInterpolationMode(vps.Interpolation.CUSTOM)
                model.setCustomInterpolator(lambda coord: 1.0 + 0.5 * np.sin(coord[0]))
            else:
                model.setInterpolationMode(vps.Interpolation.LINEAR)

            vps.Process(domain, model, 1.0).apply()

    # os.remove("rates2D.csv")
    print("2D CSVFileProcess tests passed")

if __name__ == "__main__":
    run2D()
