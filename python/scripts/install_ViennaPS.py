#!/usr/bin/env python3
import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REQUIRED_NVCC_MAJOR = 12
DEFAULT_VIENNALS_VERSION = "5.5.0"

# Detect OS
IS_WINDOWS = sys.platform == "win32" or os.name == "nt"
IS_LINUX = sys.platform.startswith("linux")
OS_NAME = platform.system()

# Global variable to store required GCC version (determined at runtime)
REQUIRED_GCC = None


def run(cmd, **kwargs):
    print("+", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, **kwargs)


def run_capture(cmd, **kwargs):
    print("+", " ".join(str(c) for c in cmd))
    return subprocess.run(
        cmd, check=True, stdout=subprocess.PIPE, text=True, **kwargs
    ).stdout.strip()


def which_or_fail(name: str) -> str:
    p = shutil.which(name)
    if not p:
        sys.exit(f"{name} is required but was not found in PATH.")
    return p


def parse_nvcc_version():
    """Parse nvcc version and return (major, minor, full_string)."""
    out = run_capture(["nvcc", "--version"])
    # look for "release 12.3" etc.
    for line in out.splitlines():
        if "release" in line:
            part = line.split("release", 1)[1].strip().split(",")[0].strip()
            # part like "12.3" or "12.0"
            try:
                version_parts = part.split(".")
                major = int(version_parts[0])
                minor = int(version_parts[1]) if len(version_parts) > 1 else 0
                return major, minor, part
            except Exception:
                break
    sys.exit("Could not parse nvcc version. Need CUDA >= 12.0.")


def determine_required_gcc_version(
    nvcc_major: int, nvcc_minor: int
) -> list[str] | None:
    """
    Determine the acceptable GCC versions based on nvcc version.

    Rules:
    - nvcc < 12.4: gcc 11 or 12
    - 12.4 <= nvcc < 12.8: gcc 11, 12, or 13
    - nvcc >= 12.8: any gcc version is ok (return None)

    Returns:
        List of acceptable GCC versions (e.g., ["11", "12"]) or None if any version is ok.
    """
    nvcc_version = nvcc_major * 10 + nvcc_minor

    if nvcc_version < 124:
        return ["11", "12"]
    elif nvcc_version < 128:
        return ["11", "12", "13"]
    else:
        return None  # Any GCC version is ok


def get_default_gcc_version() -> tuple[int, int] | None:
    """
    Get the default gcc version (major, minor).
    Returns None if gcc is not found or version cannot be parsed.
    """
    gcc_path = shutil.which("gcc")
    if not gcc_path:
        return None

    try:
        out = run_capture(["gcc", "--version"])
        # First line typically contains version, e.g., "gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
        first_line = out.splitlines()[0]
        # Look for version pattern like "11.4.0"
        import re

        match = re.search(r"(\d+)\.(\d+)", first_line)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2))
            return major, minor
    except Exception:
        pass

    return None


def ensure_compilers():
    """
    Check for required compilers based on OS.
    """
    if IS_WINDOWS:
        ensure_msvc()
    else:
        ensure_gcc_and_gpp()


def ensure_gcc_and_gpp():
    """
    Check that GCC and G++ are available (Linux).
    If specific versions are acceptable based on nvcc, ensure at least one is available.
    """
    global REQUIRED_GCC

    if REQUIRED_GCC is None:
        # No specific version required, just check that gcc exists
        which_or_fail("gcc")
        which_or_fail("g++")
        default_gcc = get_default_gcc_version()
        if default_gcc:
            print(f"Using default GCC version: {default_gcc[0]}.{default_gcc[1]}")
        else:
            print("Using default GCC version (version detection failed)")
    else:
        # Check if any of the acceptable versions is available
        found_version = None
        for gcc_ver in REQUIRED_GCC:
            if shutil.which(f"gcc-{gcc_ver}") and shutil.which(f"g++-{gcc_ver}"):
                found_version = gcc_ver
                break

        if found_version is None:
            # None of the versioned compilers found, check if default gcc is compatible
            default_gcc = get_default_gcc_version()
            if default_gcc and str(default_gcc[0]) in REQUIRED_GCC:
                found_version = str(default_gcc[0])
                print(
                    f"Using default GCC-{found_version} (compatible with CUDA version)"
                )
            else:
                sys.exit(
                    f"ERROR: None of the required GCC versions {REQUIRED_GCC} found.\n"
                    f"Please install one of: "
                    + ", ".join([f"gcc-{v}/g++-{v}" for v in REQUIRED_GCC])
                )
        else:
            print(f"Using GCC-{found_version} (compatible with CUDA version)")
            # Update REQUIRED_GCC to the single version we're using
            REQUIRED_GCC = found_version


def ensure_msvc():
    """
    Check that an MSVC toolchain is available (Windows).
    """
    print("Checking MSVC toolchain...")
    cl_path = shutil.which("cl")
    if not cl_path:
        print(
            "WARNING: 'cl.exe' was not found in PATH.\n"
            "Please run this script from a Visual Studio Developer Command Prompt\n"
            "(e.g. 'x64 Native Tools Command Prompt for VS 2022')."
        )
        # Do not hard-fail here; building will likely fail later anyway.
    else:
        print(f"Found MSVC compiler: {cl_path}")


def ensure_cuda():
    """Check CUDA availability and determine required GCC version on Linux."""
    global REQUIRED_GCC

    which_or_fail("nvcc")
    major, minor, full = parse_nvcc_version()
    if major < REQUIRED_NVCC_MAJOR:
        sys.exit(f"CUDA toolkit version 12.0 or higher is required (found {full}).")
    print(f"CUDA toolkit version: {full}")

    # On Linux, determine required GCC version based on nvcc version
    if IS_LINUX:
        REQUIRED_GCC = determine_required_gcc_version(major, minor)
        if REQUIRED_GCC:
            print(f"CUDA {full} is compatible with GCC: {', '.join(REQUIRED_GCC)}")
        else:
            print(f"CUDA {full} works with any GCC version")


def create_or_reuse_venv(venv_dir: Path):
    _, pip = venv_paths(venv_dir)
    if venv_dir.exists() and pip.exists():
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

    # On Linux, set CC and CXX environment variables
    if IS_LINUX and REQUIRED_GCC is not None:
        env["CC"] = f"gcc-{REQUIRED_GCC}"
        env["CXX"] = f"g++-{REQUIRED_GCC}"

    if viennals_dir is not None:
        print(f"Installing ViennaLS from local directory: {viennals_dir}")
        if not (viennals_dir.exists() and (viennals_dir / "CMakeLists.txt").exists()):
            sys.exit(f"ViennaLS directory not valid: {viennals_dir}")
        cmd = [str(pip_path), "install", ".", "--force-reinstall", "--no-deps"]
        if verbose:
            cmd.append("-v")
        run(cmd, cwd=viennals_dir, env=env)
        return

    current = pip_show_version(pip_path, "ViennaLS")
    if current is None:
        print("ViennaLS not installed. A local build is required.")
        which_or_fail("git")  # git is required to clone
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
        print(
            f"ViennaLS already installed ({current}). "
            f"A local build is required and the version should be {required_version}."
        )
        if current != required_version:
            print(
                "WARNING: Installed ViennaLS version does not match the required version."
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
    viennals_dir: Path | None,
    debug_build: bool,
    gpu_build: bool,
    verbose: bool,
    sanitize: bool = False,
):
    if not viennaps_dir.exists():
        sys.exit(f"ViennaPS directory not found: {viennaps_dir}")
    if not (viennaps_dir / "CMakeLists.txt").exists():
        sys.exit(
            f"{viennaps_dir} does not look like a ViennaPS source directory (missing CMakeLists.txt)."
        )

    env = os.environ.copy()

    # On Linux, set CC and CXX environment variables
    if IS_LINUX and REQUIRED_GCC is not None:
        env["CC"] = f"gcc-{REQUIRED_GCC}"
        env["CXX"] = f"g++-{REQUIRED_GCC}"

    cmake_args: list[str] = []

    # GPU on/off
    if gpu_build:
        cmake_args.append("-DVIENNAPS_USE_GPU=ON")
    else:
        cmake_args.append("-DVIENNAPS_USE_GPU=OFF")

    if viennals_dir:
        cmake_args.append(f'-DCPM_ViennaLS_SOURCE="{str(viennals_dir)}"')

    if sanitize:
        cmake_args.append(
            "-DCMAKE_CXX_FLAGS=-fsanitize=address -fno-omit-frame-pointer"
        )

    env["CMAKE_ARGS"] = " ".join(cmake_args)

    cmd = [str(pip_path), "install", "--no-deps", "."]

    if debug_build:
        print("Enabling debug build.")
        cmd.append("--config-settings=cmake.build-type=Debug")

    if verbose:
        cmd.append("-v")

    # Run the installation
    run(cmd, cwd=viennaps_dir, env=env)


def main():
    parser = argparse.ArgumentParser(
        description=f"Dev setup for ViennaPS with GPU support ({OS_NAME})."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose pip builds."
    )
    parser.add_argument(
        "--venv",
        default=os.environ.get("VIRTUAL_ENV", ".venv"),
        help="Path to the virtual environment directory.",
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
    parser.add_argument(
        "--sanitize",
        action="store_true",
        help="Build with AddressSanitizer (-fsanitize=address).",
    )
    args = parser.parse_args()

    if not args.skip_toolchain_check or not args.no_gpu:
        print("Checking toolchain...")
        ensure_cuda()
        ensure_compilers()

    # venv
    venv_dir = Path(args.venv).expanduser().resolve()
    create_or_reuse_venv(venv_dir)
    venv_python, venv_pip = venv_paths(venv_dir)

    # ViennaLS
    viennals_dir = (
        Path(args.viennals_dir).expanduser().resolve() if args.viennals_dir else None
    )
    install_viennals(venv_pip, viennals_dir, args.viennals_version, args.verbose)

    # ViennaPS
    viennaps_dir = get_viennaps_dir(args.viennaps_dir)
    install_viennaps(
        venv_pip,
        viennaps_dir,
        viennals_dir,
        args.debug_build,
        not args.no_gpu,
        args.verbose,
        args.sanitize,
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
