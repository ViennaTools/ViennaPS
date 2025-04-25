from dataclasses import dataclass, field
import csv
import os
import math

def read_config_file(path):
    config = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            # Try to cast to number, else keep as string
            try:
                if "." in value or "e" in value.lower():
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass
            config[key] = value
    return config

@dataclass
class Parameters:
    # Domain
    gridDelta: float = 2.5
    periodicBoundary: int = 0

    # Geometry
    numLayers: int = 12
    numPillars: int = 5
    layerHeight: float = 20.0
    lateralSpacing: float = 200.0

    maskHeight: float = 50.0
    maskWidth: float = 100.0

    trenchWidth: float = 100.0
    trenchWidthBot: float = 60.0

    overEtch: float = 100.0
    offSet: float = 0.0
    buffer: float = 0.0

    # Utils
    saveVolume: int = 1
    halfGeometry: int = 1

    fileName: str = "default_"
    pathFile: str = ""
    targetFile: str = ""

    def getExtent(self):
        """
        Returns [x_extent, y_extent] based on config or CSV path.
        If a CSV is used, x extent is computed from the file + buffer.
        """
        if self.pathFile and os.path.exists(self.pathFile):
            x_vals = []
            try:
                with open(self.pathFile, newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        if len(row) < 2:
                            continue
                        x = float(row[0])
                        x_vals.append(x)
                if not x_vals:
                    raise ValueError("CSV file is empty or invalid.")
                x_min = min(x_vals) - self.buffer
                x_max = max(x_vals) + self.buffer
                return [x_max - x_min, 0.0]
            except Exception as e:
                print(f"Error reading CSV: {e}")
                raise
        else:
            x_extent = 2 * self.lateralSpacing + self.numPillars * self.maskWidth + (self.numPillars - 1) * self.trenchWidth
            y_extent = self.overEtch + self.numLayers * self.layerHeight + self.maskHeight
            return [x_extent, y_extent]

    @classmethod
    def from_dict(cls, config: dict):
        """
        Construct Parameters from a dictionary, like from JSON or config parser.
        """
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config.items() if k in field_names}
        return cls(**filtered)
