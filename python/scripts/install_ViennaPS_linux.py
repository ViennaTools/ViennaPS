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
DEFAULT_VIENNALS_VERSION = "5.4.0"


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


def ensure_git():
    which_or_fail("git")


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


def pip_show_version(pip_path: Path, pkg: str):
    try:
        out = run_capture([str(pip_path), "show", pkg])
    except subprocess.CalledProcessError:
        return None
    for line in out.splitlines():
        if line.startswith("Version:"):
            return line.split(":", 1)[1].strip()
    return None


def install_viennals(
    pip_path: Path, viennals_dir: Path | None, required_version: str, verbose: bool
):
    env = os.environ.copy()
    env["CC"] = f"gcc-{REQUIRED_GCC}"
    env["CXX"] = f"g++-{REQUIRED_GCC}"

    current = pip_show_version(pip_path, "ViennaLS")
    if current is None:
        print("ViennaLS not installed. A local build is required.")
        if viennals_dir is None:
            ensure_git()
            print(f"Cloning ViennaLS v{required_version}…")
            with tempfile.TemporaryDirectory(prefix="ViennaLS_tmp_install_") as tmp:
                tmp_path = Path(tmp)
                run(
                    [
                        "git",
                        "clone",
                        "https://github.com/ViennaTools/ViennaLS.git",
                        str(tmp_path),
                    ],
                    env=env,
                )
                run(["git", "checkout", f"v{required_version}"], cwd=tmp_path, env=env)
                cmd = [str(pip_path), "install", "."]
                if verbose:
                    cmd.append("-v")
                run(cmd, cwd=tmp_path, env=env)
        else:
            if not (
                viennals_dir.exists() and (viennals_dir / "CMakeLists.txt").exists()
            ):
                sys.exit(f"ViennaLS directory not valid: {viennals_dir}")
            cmd = [str(pip_path), "install", "."]
            if verbose:
                cmd.append("-v")
            run(cmd, cwd=viennals_dir, env=env)
    else:
        print(
            f"ViennaLS already installed ({current}). Local build is required and version should be {required_version}."
        )
        if current != required_version:
            sys.exit(
                f"Version mismatch. Please change to the required version {required_version}.\n"
                "Then re-run this script."
            )
        print("Proceeding with the currently installed ViennaLS.")


def get_viennaps_dir(viennaps_dir_arg: str | None) -> Path:
    cwd = Path.cwd()
    if viennaps_dir_arg:
        viennaps_dir = Path(viennaps_dir_arg).expanduser().resolve()
    else:
        if cwd.name == "ViennaPS":
            viennaps_dir = cwd
        else:
            # try location of this script
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
    env = os.environ.copy()
    env["CC"] = f"gcc-{REQUIRED_GCC}"
    env["CXX"] = f"g++-{REQUIRED_GCC}"

    # GPU on
    cmake_args = []
    if gpu_build:
        cmake_args = ["-DVIENNAPS_USE_GPU=ON"]

    if debug_build:
        print("Enabling debug build.")
        env["CMAKE_BUILD_TYPE"] = "Debug"
        env["SKBUILD_CMAKE_BUILD_TYPE"] = "Debug"

    if optix_dir is not None:
        env["OptiX_INSTALL_DIR"] = str(optix_dir)

    env["CMAKE_ARGS"] = " ".join(cmake_args)
    cmd = [str(pip_path), "install", "--no-deps", "."]
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
    parser.add_argument(
        "--viennals-dir",
        default=None,
        help="Path to a local ViennaLS checkout (optional).",
    )
    parser.add_argument(
        "--viennals-version",
        default=DEFAULT_VIENNALS_VERSION,
        help="ViennaLS version tag to use if cloning.",
    )
    parser.add_argument(
        "--viennaps-dir",
        default=None,
        help="Path to ViennaPS directory (defaults to current dir if it's ViennaPS).",
    )
    parser.add_argument(
        "--optix",
        default=None,
        help="Path to OptiX installation directory (optional - will auto-download OptiX headers if not provided).",
    )
    parser.add_argument(
        "--debug-build",
        action="store_true",
        help="Enable debug build.",
    )
    parser.add_argument(
        "--skip-toolchain-check",
        action="store_true",
        help="Skip checking for required compilers and CUDA (use with caution).",
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

    # OptiX dir
    if not args.no_gpu:
        optix_dir = args.optix or os.environ.get("OptiX_INSTALL_DIR")
        if not optix_dir:
            print("No OptiX directory provided. Will auto-download OptiX headers.")
            print("\nWARNING: OptiX uses a different license than ViennaPS.")
            print(
                "By proceeding with auto-download, you agree to the NVIDIA OptiX license terms."
            )
            print(
                "Please review the OptiX license at: https://developer.nvidia.com/designworks/sdk-samples-tools-software-license-agreement"
            )
            print("If you do not agree, abort now (Ctrl+C).")
            input("Press Enter to continue...")
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

    # ViennaLS
    viennals_dir = (
        Path(args.viennals_dir).expanduser().resolve() if args.viennals_dir else None
    )
    install_viennals(venv_pip, viennals_dir, args.viennals_version, args.verbose)

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

    # Final info
    bindir = "Scripts" if os.name == "nt" else "bin"
    activate_hint = (
        venv_dir / bindir / ("activate" if os.name != "nt" else "activate.bat")
    )
    print("\nInstallation complete.")
    if os.name == "nt":
        print(f"Activate the virtual environment:\n  {activate_hint}")
    else:
        print(f"Activate the virtual environment:\n  source {activate_hint}")
    print("To deactivate:\n  deactivate")


if __name__ == "__main__":
    main()
