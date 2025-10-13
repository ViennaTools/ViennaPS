#!/usr/bin/env python3

"""
This script installs the ViennaTools package on various Linux distributions with optional GPU support.
It attempts to install the required dependencies on the system, therefore sudo privileges are required.

Supported distributions:
- Ubuntu 20.04+, Debian 11+
- Fedora 35+, Rocky Linux 8+, AlmaLinux 8+
- Arch Linux, Manjaro
- openSUSE Leap 15.3+, openSUSE Tumbleweed

Features:
- Optional GPU support (enabled by default, can be disabled with --no-gpu)
- Version selection for both ViennaLS and ViennaPS (specific tags or master branch)
- Automatic dependency management for multiple distributions
- Virtual environment setup
"""

import os
import sys
import subprocess
import argparse
import shutil
import platform
from pathlib import Path


def run_command(cmd, shell=False, check=True, capture_output=False):
    """Run a shell command and handle errors."""
    try:
        if capture_output:
            result = subprocess.run(
                cmd, shell=shell, check=check, capture_output=True, text=True
            )
            return result.stdout.strip()
        else:
            subprocess.run(cmd, shell=shell, check=check)
            return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        if capture_output and e.stdout:
            print(f"stdout: {e.stdout}")
        if capture_output and e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def detect_linux_distribution():
    """Detect the Linux distribution and version."""
    try:
        # Try to read /etc/os-release first (most modern distributions)
        if os.path.exists("/etc/os-release"):
            with open("/etc/os-release", "r") as f:
                lines = f.readlines()

            os_info = {}
            for line in lines:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    os_info[key] = value.strip('"')

            distro_id = os_info.get("ID", "").lower()
            distro_name = os_info.get("NAME", "")
            version = os_info.get("VERSION_ID", "")

            print(f"Detected distribution: {distro_name} {version}")
            return distro_id, version, distro_name

        # Fallback to lsb_release
        elif shutil.which("lsb_release"):
            distro_id = run_command(["lsb_release", "-si"], capture_output=True).lower()
            version = run_command(["lsb_release", "-sr"], capture_output=True)
            distro_name = run_command(["lsb_release", "-sd"], capture_output=True)

            print(f"Detected distribution: {distro_name}")
            return distro_id, version, distro_name

        else:
            print(
                "Could not detect Linux distribution. Please ensure /etc/os-release exists or lsb_release is installed."
            )
            sys.exit(1)

    except Exception as e:
        print(f"Failed to detect Linux distribution: {e}")
        sys.exit(1)


def check_distribution_support(distro_id, version):
    """Check if the detected distribution is supported."""
    supported_distros = {
        "ubuntu": {"min_version": "24.04"},
        "debian": {"min_version": "11"},
        "fedora": {"min_version": "35"},
        "rocky": {"min_version": "8"},
        "almalinux": {"min_version": "8"},
        "rhel": {"min_version": "8"},
        "centos": {"min_version": "8"},
        "arch": {"min_version": None},  # Rolling release
        "manjaro": {"min_version": None},  # Rolling release
        "opensuse-leap": {"min_version": "15.3"},
        "opensuse-tumbleweed": {"min_version": None},  # Rolling release
        "sles": {"min_version": "15"},
    }

    if distro_id not in supported_distros:
        print(f"Distribution '{distro_id}' is not officially supported.")
        print("Supported distributions:", ", ".join(supported_distros.keys()))
        response = input("Do you want to continue anyway? (y/N): ").strip().lower()
        if response != "y":
            sys.exit(1)
        return True

    min_version = supported_distros[distro_id]["min_version"]
    if min_version and version:
        try:
            # Simple version comparison (works for most cases)
            if float(version.split(".")[0]) < float(min_version.split(".")[0]):
                print(
                    f"Minimum supported version for {distro_id} is {min_version}, found {version}"
                )
                sys.exit(1)
        except (ValueError, IndexError):
            print(f"Could not parse version {version} for {distro_id}")

    print(f"Distribution {distro_id} {version} is supported.")
    return True


def get_package_manager(distro_id):
    """Determine the package manager for the distribution."""
    if distro_id in ["ubuntu", "debian"]:
        return "apt"
    elif distro_id in ["fedora", "rocky", "almalinux", "rhel", "centos"]:
        return "dnf" if shutil.which("dnf") else "yum"
    elif distro_id in ["arch", "manjaro"]:
        return "pacman"
    elif distro_id in ["opensuse-leap", "opensuse-tumbleweed", "sles"]:
        return "zypper"
    else:
        print(f"Unknown package manager for distribution: {distro_id}")
        return None


def check_gcc_compilers(distro_id, gpu_enabled=True):
    """Check if GCC compilers are installed."""
    # only necessary for GPU builds
    if not gpu_enabled:
        print("GPU build is disabled. Skipping GCC compiler check.")
        return True

    # For Ubuntu, prefer gcc-12/g++-12, for others use default gcc/g++
    if distro_id == "ubuntu":
        compilers = ["gcc-12", "g++-12"]
    else:
        compilers = ["gcc", "g++"]

    missing = [comp for comp in compilers if not shutil.which(comp)]
    if missing:
        print(f"Required compilers not found: {', '.join(missing)}")
        print("They will be installed with system dependencies.")
    else:
        print(f"Found compilers: {', '.join(compilers)}")

    return True


def check_cuda_version(gpu_enabled):
    """Check CUDA version if GPU build is enabled."""
    if not gpu_enabled:
        print("GPU build is disabled. Skipping CUDA version check.")
        return None

    if not shutil.which("nvcc"):
        print(
            "nvcc is required for GPU build but not installed. Please install it first."
        )
        sys.exit(1)

    try:
        nvcc_output = run_command(["nvcc", "--version"], capture_output=True)
        # Extract version from output like "release 12.0, V12.0.76"
        for line in nvcc_output.split("\n"):
            if "release" in line:
                version_part = line.split("release")[1].split(",")[0].strip()
                major_version = float(
                    version_part.split(".")[0] + "." + version_part.split(".")[1]
                )
                if major_version < 12.0:
                    print(
                        "CUDA toolkit version 12.0 or higher is required. Please update your CUDA toolkit."
                    )
                    sys.exit(1)
                print(f"CUDA toolkit version: {version_part}")
                return version_part
    except Exception as e:
        print(f"Failed to check CUDA version: {e}")
        sys.exit(1)


def get_optix_directory(gpu_enabled):
    """Get OptiX directory from environment variable or user input if GPU is enabled."""
    if not gpu_enabled:
        print("GPU build is disabled. OptiX directory not required.")
        return None

    optix_dir = os.environ.get("OptiX_INSTALL_DIR")

    if optix_dir:
        print(f"Using OptiX directory from environment variable: {optix_dir}")
        return optix_dir

    optix_dir = input(
        "Please enter the path to the OptiX directory (e.g., /path/to/Optix): "
    ).strip()
    if not optix_dir:
        print(
            "No OptiX directory specified. Please set the OptiX_INSTALL_DIR environment variable or provide a path."
        )
        sys.exit(1)

    print(f"OptiX directory is set to: {optix_dir}")
    return optix_dir


def get_system_packages(distro_id, pkg_manager):
    """Get system package names for the distribution."""
    packages = {
        "apt": {
            "vtk": "libvtk9-dev",
            "embree": "libembree-dev",
            "venv": "python3-venv",
            "build": ["build-essential", "cmake"],
            "compilers": (
                ["gcc-12", "g++-12"] if distro_id == "ubuntu" else ["gcc", "g++"]
            ),
        },
        "dnf": {
            "vtk": "vtk-devel",
            "embree": "embree-devel",
            "venv": "python3-venv",
            "build": ["gcc", "gcc-c++", "cmake", "make"],
            "compilers": ["gcc", "gcc-c++"],
        },
        "yum": {
            "vtk": "vtk-devel",
            "embree": "embree-devel",
            "venv": "python3-venv",
            "build": ["gcc", "gcc-c++", "cmake", "make"],
            "compilers": ["gcc", "gcc-c++"],
        },
        "pacman": {
            "vtk": "vtk",
            "embree": "embree",
            "venv": "python",
            "build": ["base-devel", "cmake"],
            "compilers": ["gcc"],
        },
        "zypper": {
            "vtk": "vtk-devel",
            "embree": "embree-devel",
            "venv": "python3-venv",
            "build": ["gcc", "gcc-c++", "cmake", "make"],
            "compilers": ["gcc", "gcc-c++"],
        },
    }

    if pkg_manager not in packages:
        print(f"Package definitions not available for {pkg_manager}")
        return None

    pkg_list = []
    pkg_info = packages[pkg_manager]

    # Add VTK (optional - may not be available on all distros)
    try:
        pkg_list.append(pkg_info["vtk"])
    except KeyError:
        print("VTK package not defined for this distribution")

    # Add Embree (optional - may not be available on all distros)
    try:
        pkg_list.append(pkg_info["embree"])
    except KeyError:
        print("Embree package not defined for this distribution")

    # Add Python venv support
    pkg_list.append(pkg_info["venv"])

    # Add build tools
    pkg_list.extend(pkg_info["build"])

    return pkg_list


def install_system_dependencies(distro_id, pkg_manager):
    """Install system dependencies using the appropriate package manager."""
    print(f"Installing system dependencies using {pkg_manager}...")

    packages = get_system_packages(distro_id, pkg_manager)
    if not packages:
        print("Could not determine packages for this distribution.")
        sys.exit(1)

    # Build install command based on package manager
    if pkg_manager == "apt":
        # Update package list first
        print("Updating package lists...")
        if not run_command(["sudo", "apt", "update"]):
            print("Failed to update package lists.")
            sys.exit(1)
        cmd = ["sudo", "apt", "install", "-y"] + packages
    elif pkg_manager in ["dnf", "yum"]:
        cmd = ["sudo", pkg_manager, "install", "-y"] + packages
    elif pkg_manager == "pacman":
        cmd = ["sudo", "pacman", "-S", "--noconfirm"] + packages
    elif pkg_manager == "zypper":
        cmd = ["sudo", "zypper", "install", "-y"] + packages
    else:
        print(f"Unsupported package manager: {pkg_manager}")
        sys.exit(1)

    print(f"Installing packages: {', '.join(packages)}")
    if not run_command(cmd):
        print("Failed to install system dependencies.")
        print("Some packages may not be available in your distribution's repositories.")
        print("You may need to install VTK and Embree development packages manually.")
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != "y":
            sys.exit(1)
    else:
        print("System dependencies installed successfully.")


def setup_viennatools_directory(viennals_version=None, viennaps_version=None):
    """Set up ViennaTools directory and repositories."""
    viennatools_dir = Path("ViennaTools")

    if viennatools_dir.exists():
        print("ViennaTools directory exists. Attempting to reinstall.")
        os.chdir(viennatools_dir)

        # Update ViennaLS
        viennals_dir = Path("ViennaLS")
        if viennals_dir.exists():
            os.chdir(viennals_dir)
            if viennals_version and viennals_version != "master":
                print(f"Checking out ViennaLS version: {viennals_version}")
                if not run_command(["git", "fetch", "--tags"]):
                    print("Failed to fetch tags for ViennaLS repository.")
                    sys.exit(1)
                if not run_command(["git", "checkout", viennals_version]):
                    print(f"Failed to checkout ViennaLS version {viennals_version}.")
                    sys.exit(1)
            else:
                print("Updating ViennaLS to latest master...")
                if not run_command(["git", "checkout", "master"]):
                    print("Failed to checkout master branch for ViennaLS.")
                    sys.exit(1)
                if not run_command(["git", "pull"]):
                    print("Failed to update ViennaLS repository.")
                    sys.exit(1)
            os.chdir("..")

        # Update ViennaPS
        viennaps_dir = Path("ViennaPS")
        if viennaps_dir.exists():
            os.chdir(viennaps_dir)
            if viennaps_version and viennaps_version != "master":
                print(f"Checking out ViennaPS version: {viennaps_version}")
                if not run_command(["git", "fetch", "--tags"]):
                    print("Failed to fetch tags for ViennaPS repository.")
                    sys.exit(1)
                if not run_command(["git", "checkout", viennaps_version]):
                    print(f"Failed to checkout ViennaPS version {viennaps_version}.")
                    sys.exit(1)
            else:
                print("Updating ViennaPS to latest master...")
                if not run_command(["git", "checkout", "master"]):
                    print("Failed to checkout master branch for ViennaPS.")
                    sys.exit(1)
                if not run_command(["git", "pull"]):
                    print("Failed to update ViennaPS repository.")
                    sys.exit(1)
            os.chdir("..")

    else:
        # Create directory and clone repositories
        viennatools_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(viennatools_dir)

        # Clone ViennaLS
        print("Cloning ViennaLS repository...")
        if not run_command(
            ["git", "clone", "https://github.com/ViennaTools/ViennaLS.git"]
        ):
            print("Failed to clone ViennaLS repository.")
            sys.exit(1)

        # Checkout specific version if requested
        if viennals_version and viennals_version != "master":
            os.chdir("ViennaLS")
            print(f"Checking out ViennaLS version: {viennals_version}")
            if not run_command(["git", "fetch", "--tags"]):
                print("Failed to fetch tags for ViennaLS repository.")
                sys.exit(1)
            if not run_command(["git", "checkout", viennals_version]):
                print(f"Failed to checkout ViennaLS version {viennals_version}.")
                sys.exit(1)
            os.chdir("..")

        # Clone ViennaPS
        print("Cloning ViennaPS repository...")
        if not run_command(
            ["git", "clone", "https://github.com/ViennaTools/ViennaPS.git"]
        ):
            print("Failed to clone ViennaPS repository.")
            sys.exit(1)

        # Checkout specific version if requested
        if viennaps_version and viennaps_version != "master":
            os.chdir("ViennaPS")
            print(f"Checking out ViennaPS version: {viennaps_version}")
            if not run_command(["git", "fetch", "--tags"]):
                print("Failed to fetch tags for ViennaPS repository.")
                sys.exit(1)
            if not run_command(["git", "checkout", viennaps_version]):
                print(f"Failed to checkout ViennaPS version {viennaps_version}.")
                sys.exit(1)
            os.chdir("..")


def setup_virtual_environment(venv_dir):
    """Set up Python virtual environment."""
    venv_path = Path(venv_dir)

    if venv_path.exists():
        print(f"{venv_dir} already exists. Reusing the existing virtual environment.")
    else:
        print(f"Creating Python virtual environment: {venv_dir}")
        if not run_command([sys.executable, "-m", "venv", venv_dir]):
            print("Failed to create virtual environment.")
            sys.exit(1)

    return venv_path


def activate_and_install_packages(
    venv_path, optix_dir, verbose, gpu_enabled, distro_id
):
    """Install ViennaLS and ViennaPS packages."""
    # Set up environment variables
    env = os.environ.copy()

    # Set compiler based on distribution
    if distro_id == "ubuntu" and shutil.which("gcc-12"):
        env["CC"] = "gcc-12"
        env["CXX"] = "g++-12"
    else:
        env["CC"] = "gcc"
        env["CXX"] = "g++"

    # Python executable in virtual environment
    python_exe = venv_path / "bin" / "python"
    pip_exe = venv_path / "bin" / "pip"

    # Install ViennaLS
    print("Installing ViennaLS...")
    os.chdir("ViennaLS")
    install_cmd = [str(pip_exe), "install", "."]
    if verbose:
        install_cmd.append("-v")

    if not run_command(install_cmd, shell=False):
        print("Failed to install ViennaLS.")
        sys.exit(1)
    os.chdir("..")

    # Install ViennaPS
    if gpu_enabled:
        print("Installing ViennaPS with GPU support...")
        env["OptiX_INSTALL_DIR"] = optix_dir
        env["CMAKE_ARGS"] = "-DVIENNAPS_USE_GPU=ON"
    else:
        print("Installing ViennaPS (CPU-only)...")
        # Ensure GPU is disabled
        env["CMAKE_ARGS"] = "-DVIENNAPS_USE_GPU=OFF"

    os.chdir("ViennaPS")

    install_cmd = [str(pip_exe), "install", "."]
    if verbose:
        install_cmd.append("-v")

    # Run with modified environment
    try:
        subprocess.run(install_cmd, env=env, check=True)
    except subprocess.CalledProcessError:
        print("Failed to install ViennaPS.")
        sys.exit(1)

    os.chdir("..")


def main():
    """Main installation function."""
    parser = argparse.ArgumentParser(
        description="Install ViennaTools package on various Linux distributions"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose mode"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=True,
        help="Enable GPU support (default: True)",
    )
    parser.add_argument(
        "--no-gpu", dest="gpu", action="store_false", help="Disable GPU support"
    )
    parser.add_argument(
        "--viennals-version",
        type=str,
        default="master",
        help="ViennaLS version to install (tag name or 'master' for latest, default: master)",
    )
    parser.add_argument(
        "--viennaps-version",
        type=str,
        default="master",
        help="ViennaPS version to install (tag name or 'master' for latest, default: master)",
    )

    args = parser.parse_args()

    if args.verbose:
        print("Verbose mode is enabled.")
    else:
        print("Verbose mode is disabled.")

    if args.gpu:
        print("GPU support is enabled.")
    else:
        print("GPU support is disabled.")

    print(f"ViennaLS version: {args.viennals_version}")
    print(f"ViennaPS version: {args.viennaps_version}")

    # Check prerequisites
    distro_id, version, distro_name = detect_linux_distribution()
    check_distribution_support(distro_id, version)

    # Get package manager
    pkg_manager = get_package_manager(distro_id)
    if not pkg_manager:
        sys.exit(1)

    check_gcc_compilers(distro_id, args.gpu)
    check_cuda_version(args.gpu)

    # Get virtual environment directory
    venv_dir = input(
        "Enter the path to the virtual environment directory (default: .venv): "
    ).strip()
    if not venv_dir:
        venv_dir = ".venv"
        print(f"No virtual environment directory specified. Using default: {venv_dir}")

    # Get OptiX directory if GPU is enabled
    optix_dir = get_optix_directory(args.gpu)

    # Install system dependencies
    install_system_dependencies(distro_id, pkg_manager)

    # Set up ViennaTools directory and repositories
    original_dir = os.getcwd()
    setup_viennatools_directory(args.viennals_version, args.viennaps_version)

    # Set up virtual environment
    venv_path = setup_virtual_environment(venv_dir)

    # Install packages
    activate_and_install_packages(
        venv_path, optix_dir, args.verbose, args.gpu, distro_id
    )

    # Return to original directory
    os.chdir(original_dir)

    build_type = "GPU-enabled" if args.gpu else "CPU-only"
    print(
        f"Installation complete ({build_type}). To activate the virtual environment, run:"
    )
    print(f"source ViennaTools/{venv_dir}/bin/activate")
    print("To deactivate the virtual environment, run:")
    print("deactivate")


if __name__ == "__main__":
    main()
