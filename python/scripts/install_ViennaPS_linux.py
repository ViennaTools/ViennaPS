#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REQUIRED_GCC = "12"
REQUIRED_NVCC_MAJOR = 12
# UPDATED: Pointing to local directory instead of git URL
VIENNALS_LOCAL_PATH = "/home/filipov/Software/GPU/ViennaLS"

def run(cmd, **kwargs):
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=True, **kwargs)


def run_capture(cmd, **kwargs):
    print("+", " ".join(cmd))
    return subprocess.run(
        cmd, check=True, stdout=subprocess.PIPE, text=True, **kwargs
    ).stdout.strip()


def which_or_fail(name):
    p = shutil.which(name)
    if not p:
        sys.exit(f"{name} is required but not found in PATH.")
    return p


def parse_nvcc_version():
    out = run_capture(["nvcc", "--version"])
    # look for "release 12.3" etc.
    for line in out.splitlines():
        if "release" in line:
            part = line.split("release", 1)[1].strip().split(",")[0].strip()
            # part like "12.3" or "12.0"
            try:
                major = int(part.split(".")[0])
                return major, part
            except Exception:
                break
    sys.exit("Could not parse nvcc version. Need CUDA >= 12.0.")


def ensure_compilers():
    which_or_fail(f"gcc-{REQUIRED_GCC}")
    which_or_fail(f"g++-{REQUIRED_GCC}")


def ensure_cuda():
    which_or_fail("nvcc")
    major, full = parse_nvcc_version()
    if major < REQUIRED_NVCC_MAJOR:
        sys.exit(f"CUDA toolkit version 12.0 or higher is required (found {full}).")
    print(f"CUDA toolkit version: {full}")


def create_or_reuse_venv(venv_dir: Path):
    if venv_dir.exists():
        print(f"{venv_dir} already exists. Reusing the existing virtual environment.")
    else:
        print(f"Creating virtual environment at {venv_dir}")
        run([sys.executable, "-m", "venv", str(venv_dir)])


def venv_paths(venv_dir: Path):
    bindir = "Scripts" if os.name == "nt" else "bin"
    python = venv_dir / bindir / ("python.exe" if os.name == "nt" else "python")
    pip = venv_dir / bindir / ("pip.exe" if os.name == "nt" else "pip")
    return python, pip


def install_build_deps(pip_path: Path):
    """
    Installs build dependencies required for --no-build-isolation.
    """
    print("Installing build dependencies (scikit-build-core, pybind11, cmake, ninja)...")
    run([
        str(pip_path), 
        "install", 
        "scikit-build-core", 
        "pybind11", 
        "cmake", 
        "ninja",
        "pathspec", 
        "packaging"
    ])


def install_viennals(pip_path: Path, verbose: bool):
    """
    Installs ViennaLS directly from the local directory using pip.
    """
    env = os.environ.copy()
    env["CC"] = f"gcc-{REQUIRED_GCC}"
    env["CXX"] = f"g++-{REQUIRED_GCC}"

    print(f"Installing ViennaLS from: {VIENNALS_LOCAL_PATH}")

    # Use pip to handle the install logic internally.
    # --no-build-isolation ensures it uses the build deps we just installed.
    # --force-reinstall ensuring we overwrite any old versions.
    cmd = [
        str(pip_path),
        "install",
        VIENNALS_LOCAL_PATH,
        "--force-reinstall",
        "--no-cache-dir",
        "--no-build-isolation",
    ]

    if verbose:
        cmd.append("-v")

    run(cmd, env=env)


def get_viennaps_dir(viennaps_dir_arg: str | None) -> Path:
    cwd = Path.cwd()
    if viennaps_dir_arg:
        viennaps_dir = Path(viennaps_dir_arg).expanduser().resolve()
    else:
        if cwd.name == "ViennaPS":
            viennaps_dir = cwd
        else:
            script_path = Path(__file__).resolve()
            cwd = script_path.parent.parent
            if cwd.name == "ViennaPS":
                viennaps_dir = cwd
            else:
                try:
                    entered = input(
                        "Please enter the path to the ViennaPS directory: "
                    ).strip()
                except EOFError:
                    entered = ""
                if not entered:
                    sys.exit("No ViennaPS directory provided.")
                viennaps_dir = Path(entered).expanduser().resolve()
    return viennaps_dir


def install_viennaps(
    pip_path: Path,
    viennaps_dir: Path,
    optix_dir: Path | None,
    debug_build: bool,
    gpu_build: bool,
    verbose: bool,
):
    if not viennaps_dir.exists():
        sys.exit(f"ViennaPS directory not found: {viennaps_dir}")
    if not (viennaps_dir / "CMakeLists.txt").exists():
        sys.exit(
            f"{viennaps_dir} does not look like a ViennaPS source directory (missing CMakeLists.txt)."
        )

    # # --- CRITICAL FIX: Clean Build Directory ---
    # # This prevents the "poisoned cache" issue where CMake remembers the old ViennaLS path.
    # build_dir = viennaps_dir / "build"
    # if build_dir.exists():
    #     print(f"Removing dirty build directory: {build_dir}")
    #     shutil.rmtree(build_dir)
    # -------------------------------------------

    env = os.environ.copy()
    env["CC"] = f"gcc-{REQUIRED_GCC}"
    env["CXX"] = f"g++-{REQUIRED_GCC}"

    cmake_args = []
    if gpu_build:
        cmake_args.append("-DVIENNAPS_USE_GPU=ON")

    if debug_build:
        print("Enabling debug build.")
        env["CMAKE_BUILD_TYPE"] = "Debug"
        env["SKBUILD_CMAKE_BUILD_TYPE"] = "Debug"

    if optix_dir is not None:
        env["OptiX_INSTALL_DIR"] = str(optix_dir)

    if cmake_args:
        env["CMAKE_ARGS"] = " ".join(cmake_args)

    cmd = [
        str(pip_path), 
        "install", 
        ".", 
        "--force-reinstall", 
        "--no-cache-dir", 
        "--no-build-isolation"
    ]
    if verbose:
        cmd.append("-v")
    
    run(cmd, cwd=viennaps_dir, env=env)


def main():
    parser = argparse.ArgumentParser(
        description="Dev setup for ViennaPS with GPU support (Linux)."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose pip builds."
    )
    parser.add_argument(
        "--venv", default=".venv", help="Path to the virtual environment directory."
    )
    # These args are ignored now as we force the local path, but kept for compatibility
    parser.add_argument("--viennals-dir", default=None, help="Ignored: Using local path.")
    parser.add_argument("--viennals-version", default=None, help="Ignored: Using local path.")
    
    parser.add_argument(
        "--viennaps-dir",
        default=None,
        help="Path to ViennaPS directory (defaults to current dir if it's ViennaPS).",
    )
    parser.add_argument(
        "--optix",
        default=None,
        help="Path to OptiX installation directory (optional).",
    )
    parser.add_argument(
        "--debug-build",
        action="store_true",
        help="Enable debug build.",
    )
    parser.add_argument(
        "--skip-toolchain-check",
        action="store_true",
        help="Skip checking for required compilers and CUDA.",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU support.",
    )
    args = parser.parse_args()

    if not args.skip_toolchain_check or not args.no_gpu:
        print("Checking toolchain...")
        ensure_compilers()
        ensure_cuda()

    if not args.no_gpu:
        optix_dir = args.optix or os.environ.get("OptiX_INSTALL_DIR")
        if not optix_dir:
            print("No OptiX directory provided. Will auto-download OptiX headers.")
            optix_dir = None
        else:
            optix_dir = Path(optix_dir).expanduser().resolve()
            if not optix_dir.exists():
                sys.exit(f"OptiX directory not found: {optix_dir}")
            print(f"Using OptiX at: {optix_dir}")
    else:
        optix_dir = None

    # venv
    venv_dir = Path(args.venv).expanduser().resolve()
    create_or_reuse_venv(venv_dir)
    venv_python, venv_pip = venv_paths(venv_dir)

    # 1. Install Build Dependencies
    install_build_deps(venv_pip)

    # 2. Install ViennaLS from Local Path
    install_viennals(venv_pip, args.verbose)

    # ViennaPS dir
    viennaps_dir = get_viennaps_dir(args.viennaps_dir)

    # ViennaPS install
    install_viennaps(
        venv_pip,
        viennaps_dir,
        optix_dir,
        args.debug_build,
        not args.no_gpu,
        args.verbose,
    )

    print("\nInstallation complete.")
    bindir = "Scripts" if os.name == "nt" else "bin"
    activate_hint = venv_dir / bindir / ("activate" if os.name != "nt" else "activate.bat")
    print(f"Activate: source {activate_hint}")


if __name__ == "__main__":
    main()