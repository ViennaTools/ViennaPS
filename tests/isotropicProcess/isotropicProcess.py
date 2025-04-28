import subprocess
import sys

print("Running 2D IsotropicProcess test...")
subprocess.check_call([sys.executable, "test_isotropic_2d.py"])

print("Running 3D IsotropicProcess test...")
subprocess.check_call([sys.executable, "test_isotropic_3d.py"])

print("All IsotropicProcess tests passed")
