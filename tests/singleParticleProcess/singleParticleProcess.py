import subprocess
import sys

print("Running 2D SingleParticleProcess test...")
subprocess.check_call([sys.executable, "test_single_particle_2d.py"])

print("Running 3D SingleParticleProcess test...")
subprocess.check_call([sys.executable, "test_single_particle_3d.py"])

print("All SingleParticleProcess tests passed")
