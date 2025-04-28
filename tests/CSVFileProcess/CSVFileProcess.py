# run_all_csv_tests.py
import subprocess
import sys

print("Running 2D CSVFileProcess tests...")
subprocess.check_call([sys.executable, "test_csv_2d.py"])

print("Running 3D CSVFileProcess tests...")
subprocess.check_call([sys.executable, "test_csv_3d.py"])

print("All CSVFileProcess tests passed!")
