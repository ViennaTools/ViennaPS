import argparse
import os
import subprocess
import sys
from typing import Dict, List


# List of examples to run
EXAMPLES: List[Dict[str, object]] = [
    {
        "name": "Trench_Deposition",
        "dir": "examples/trenchDeposition",
        "script": "trenchDeposition.py",
        "args": ["-D", "2", "config.txt"],
    },
    {
        "name": "Hole_Etching",
        "dir": "examples/holeEtching",
        "script": "holeEtching.py",
        "args": ["-D", "2", "config.txt"],
    },
    {
        "name": "Blazed_Gratings_Etching",
        "dir": "examples/blazedGratingsEtching",
        "script": "blazedGratingsEtching.py",
        "args": ["config.txt"],
    },
    {
        "name": "Bosch_simulate",
        "dir": "examples/boschProcess",
        "script": "boschProcessSimulate.py",
        "args": ["-D", "2", "config.txt"],
    },
    {
        "name": "Bosch_emulate",
        "dir": "examples/boschProcess",
        "script": "boschProcessEmulate.py",
        "args": ["-D", "2", "config.txt"],
    },
    {
        "name": "Ion_Beam_Etching",
        "dir": "examples/ionBeamEtching",
        "script": "ionBeamEtching.py",
        "args": ["-D", "2", "config.txt"],
    },
    {
        "name": "Faraday_Cage_Etching",
        "dir": "examples/faradayCageEtching",
        "script": "faradayCageEtching.py",
        "args": ["-D", "2", "config.txt"],
    },
    {
        "name": "Selective_Epitaxy",
        "dir": "examples/selectiveEpitaxy",
        "script": "selectiveEpitaxy.py",
        "args": ["-D", "3", "config.txt"],
    },
    {
        "name": "Simple_Etching",
        "dir": "examples/simpleEtching",
        "script": "simpleEtching.py",
        "args": [],
    },
    {
        "name": "Stack_Etching",
        "dir": "examples/stackEtching",
        "script": "stackEtching.py",
        "args": ["config.txt"],
    },
    {
        "name": "Single_TEOS",
        "dir": "examples/TEOSTrenchDeposition",
        "script": "singleTEOS.py",
        "args": ["-D", "2", "singleTEOS_config.txt"],
    },
    {
        "name": "Multi_TEOS",
        "dir": "examples/TEOSTrenchDeposition",
        "script": "multiTEOS.py",
        "args": ["-D", "2", "multiTEOS_config.txt"],
    },
    {
        "name": "Sputter_Deposition",
        "dir": "examples/sputterDeposition",
        "script": "sputterDeposition.py",
        "args": ["-D", "3", "config3D.txt"],
    },
    {
        "name": "SiGe_Selective_Etching",
        "dir": "examples/SiGeSelectiveEtching",
        "script": "SiGeEtching.py",
        "args": [],
    },
    {
        "name": "Cantilever_Wet_Etching",
        "dir": "examples/cantileverWetEtching",
        "script": "cantileverWetEtching.py",
        "args": [],
    },
]


def run_example(source_dir: str, example: Dict[str, object], python_exe: str) -> bool:
    example_name = str(example["name"])
    rel_dir = str(example["dir"])
    script = str(example["script"])
    args = [str(arg) for arg in example.get("args", [])]

    working_dir = os.path.join(source_dir, rel_dir)
    script_path = os.path.join(working_dir, script)

    if not os.path.isdir(working_dir):
        print(f"[FAIL] {example_name}: example directory not found: {working_dir}")
        return False

    if not os.path.isfile(script_path):
        print(f"[FAIL] {example_name}: script not found: {script_path}")
        return False

    cmd = [python_exe, script] + args
    print(f"[RUN ] {example_name}: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, cwd=working_dir, check=True)
        print(f"[ OK ] {example_name}")
        return True
    except subprocess.CalledProcessError as error:
        print(f"[FAIL] {example_name}: exited with code {error.returncode}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run configured ViennaPS examples from a source folder"
    )
    parser.add_argument(
        "source_dir",
        nargs="?",
        default=".",
        help="Path to ViennaPS source folder containing .venv and examples/",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with remaining examples after a failure",
    )
    args = parser.parse_args()

    source_dir = os.path.abspath(args.source_dir)
    python_exe = os.path.join(source_dir, ".venv", "bin", "python")

    if not os.path.isfile(python_exe):
        print(f"Python executable not found: {python_exe}")
        return 2

    print(f"Using Python: {python_exe}")
    print(f"Source folder: {source_dir}")
    print(f"Examples to run: {len(EXAMPLES)}")

    passed = 0
    failed = 0

    for example in EXAMPLES:
        success = run_example(source_dir, example, python_exe)
        if success:
            passed += 1
        else:
            failed += 1
            if not args.keep_going:
                print("Stopping after first failure. Use --keep-going to continue.")
                break

    print("\nSummary")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
