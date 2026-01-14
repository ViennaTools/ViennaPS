import subprocess
import sys

print("Running 2D DirectionalProcess test...")
subprocess.check_call([sys.executable, "test_directional_2d.py"])

print("Running 3D DirectionalProcess test...")
subprocess.check_call([sys.executable, "test_directional_3d.py"])

print("All DirectionalProcess tests passed")
