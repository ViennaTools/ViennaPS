#!/usr/bin/env python3

"""
This script installs the ViennaTools package on various Linux distributions and macOS with optional GPU support.
It attempts to install the required dependencies on the system, therefore sudo privileges are required on Linux
and Homebrew is required on macOS.

Supported platforms:
- Ubuntu 20.04+, Debian 11+
- Fedora 35+, Rocky Linux 8+, AlmaLinux 8+
- Arch Linux, Manjaro
- openSUSE Leap 15.3+, openSUSE Tumbleweed
- macOS 12+ (with Homebrew)

Features:
- Optional GPU support (enabled by default, can be disabled with --no-gpu)
- Version selection for both ViennaLS and ViennaPS (specific tags or master branch)
- Automatic dependency management for multiple distributions and macOS
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
        if capture_output:
            return None
        return False


def detect_os_platform():
    """Detect the operating system platform and version."""
    system = platform.system()

    if system == "Darwin":  # macOS
        version = platform.mac_ver()[0]
        print(f"Detected platform: macOS {version}")
        return "macos", version, f"macOS {version}"

    elif system == "Linux":
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
                distro_id_result = run_command(
                    ["lsb_release", "-si"], capture_output=True
                )
                version_result = run_command(
                    ["lsb_release", "-sr"], capture_output=True
                )
                distro_name_result = run_command(
                    ["lsb_release", "-sd"], capture_output=True
                )

                if (
                    isinstance(distro_id_result, str)
                    and isinstance(version_result, str)
                    and isinstance(distro_name_result, str)
                ):
                    distro_id = distro_id_result.lower()
                    version = version_result
                    distro_name = distro_name_result
                    print(f"Detected distribution: {distro_name}")
                    return distro_id, version, distro_name
                else:
                    raise Exception("Failed to get distribution info from lsb_release")

            else:
                print(
                    "Could not detect Linux distribution. Please ensure /etc/os-release exists or lsb_release is installed."
                )
                sys.exit(1)

        except Exception as e:
            print(f"Failed to detect Linux distribution: {e}")
            sys.exit(1)

    else:
        print(f"Unsupported operating system: {system}")
        print("This script supports Linux distributions and macOS only.")
        sys.exit(1)


def check_platform_support(platform_id, version):
    """Check if the detected platform is supported."""
    supported_platforms = {
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
        "macos": {"min_version": "12.0"},  # macOS Monterey and later
    }

    if platform_id not in supported_platforms:
        print(f"Platform '{platform_id}' is not officially supported.")
        print("Supported platforms:", ", ".join(supported_platforms.keys()))
        response = input("Do you want to continue anyway? (y/N): ").strip().lower()
        if response != "y":
            sys.exit(1)
        return True

    min_version = supported_platforms[platform_id]["min_version"]
    if min_version and version:
        try:
            # Simple version comparison (works for most cases)
            if float(version.split(".")[0]) < float(min_version.split(".")[0]):
                print(
                    f"Minimum supported version for {platform_id} is {min_version}, found {version}"
                )
                sys.exit(1)
        except (ValueError, IndexError):
            print(f"Could not parse version {version} for {platform_id}")

    print(f"Platform {platform_id} {version} is supported.")
    return True


def get_package_manager(platform_id):
    """Determine the package manager for the platform."""
    if platform_id in ["ubuntu", "debian"]:
        return "apt"
    elif platform_id in ["fedora", "rocky", "almalinux", "rhel", "centos"]:
        return "dnf" if shutil.which("dnf") else "yum"
    elif platform_id in ["arch", "manjaro"]:
        return "pacman"
    elif platform_id in ["opensuse-leap", "opensuse-tumbleweed", "sles"]:
        return "zypper"
    elif platform_id == "macos":
        if shutil.which("brew"):
            return "brew"
        else:
            print("Homebrew is required for macOS but not installed.")
            print("Please install Homebrew from https://brew.sh/ and try again.")
            sys.exit(1)
    else:
        print(f"Unknown package manager for platform: {platform_id}")
        return None


def check_gcc_compilers(platform_id, gpu_enabled=True):
    """Check if compilers are installed."""
    # only necessary for GPU builds
    if not gpu_enabled:
        print("GPU build is disabled. Skipping compiler check.")
        return True

    # For macOS, check for clang (from Xcode command line tools)
    if platform_id == "macos":
        compilers = ["clang", "clang++"]
    # For Ubuntu, prefer gcc-12/g++-12, for others use default gcc/g++
    elif platform_id == "ubuntu":
        compilers = ["gcc-12", "g++-12"]
    else:
        compilers = ["gcc", "g++"]

    missing = [comp for comp in compilers if not shutil.which(comp)]
    if missing:
        print(f"Required compilers not found: {', '.join(missing)}")
        if platform_id == "macos":
            print("They will be installed with Xcode command line tools.")
        else:
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
        if not isinstance(nvcc_output, str):
            print("Failed to get CUDA version information.")
            sys.exit(1)

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


def get_system_packages(platform_id, pkg_manager):
    """Get system package names for the platform."""
    packages = {
        "apt": {
            "vtk": "libvtk9-dev",
            "embree": "libembree-dev",
            "venv": "python3-venv",
            "build": ["build-essential", "cmake"],
            "compilers": (
                ["gcc-12", "g++-12"] if platform_id == "ubuntu" else ["gcc", "g++"]
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
        "brew": {
            "vtk": "vtk",
            "embree": "embree",
            "venv": None,  # macOS Python includes venv by default
            "build": ["cmake"],
            "compilers": [],  # Xcode command line tools provide compilers
        },
    }

    if pkg_manager not in packages:
        print(f"Package definitions not available for {pkg_manager}")
        return None

    pkg_list = []
    pkg_info = packages[pkg_manager]

    # Add VTK (optional - may not be available on all platforms)
    if "vtk" in pkg_info and pkg_info["vtk"]:
        pkg_list.append(pkg_info["vtk"])
    else:
        print("VTK package not defined for this platform")

    # Add Embree (optional - may not be available on all platforms)
    if "embree" in pkg_info and pkg_info["embree"]:
        pkg_list.append(pkg_info["embree"])
    else:
        print("Embree package not defined for this platform")

    # Add Python venv support (if needed)
    if "venv" in pkg_info and pkg_info["venv"]:
        pkg_list.append(pkg_info["venv"])

    # Add build tools
    if "build" in pkg_info:
        pkg_list.extend(pkg_info["build"])

    return pkg_list


def install_system_dependencies(platform_id, pkg_manager):
    """Install system dependencies using the appropriate package manager."""
    print(f"Installing system dependencies using {pkg_manager}...")

    packages = get_system_packages(platform_id, pkg_manager)
    if not packages:
        print("Could not determine packages for this platform.")
        sys.exit(1)

    # Special handling for macOS - check for Xcode command line tools first
    if pkg_manager == "brew":
        print("Checking for Xcode command line tools...")
        try:
            # Check if Xcode command line tools are installed
            run_command(["xcode-select", "-p"], capture_output=True, check=True)
            print("Xcode command line tools are already installed.")
        except subprocess.CalledProcessError:
            print("Installing Xcode command line tools...")
            if not run_command(["xcode-select", "--install"]):
                print("Failed to install Xcode command line tools.")
                print("Please install them manually and try again.")
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
    elif pkg_manager == "brew":
        # Set the environment var to avoid auto-upgrade
        os.environ["HOMEBREW_NO_INSTALL_UPGRADE"] = "1"

        # Homebrew doesn't need sudo and installs packages individually
        for package in packages:
            print(f"Installing {package}...")
            if not run_command(["brew", "install", package]):
                print(
                    f"Failed to install {package}. It may already be installed or unavailable."
                )
        return
    else:
        print(f"Unsupported package manager: {pkg_manager}")
        sys.exit(1)

    print(f"Installing packages: {', '.join(packages)}")
    if not run_command(cmd):
        print("Failed to install system dependencies.")
        print("Some packages may not be available in your platform's repositories.")
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


def activate_and_install_packages(venv_path, verbose, gpu_enabled, platform_id):
    """Install ViennaLS and ViennaPS packages."""
    # Set up environment variables
    env = os.environ.copy()

    # Set compiler based on platform
    if platform_id == "macos":
        env["CC"] = "clang"
        env["CXX"] = "clang++"
    elif platform_id == "ubuntu" and shutil.which("gcc-12"):
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
    install_cmd = ["../" + str(pip_exe), "install", "."]
    if verbose:
        install_cmd.append("-v")

    if not run_command(install_cmd, shell=False):
        print("Failed to install ViennaLS.")
        sys.exit(1)
    os.chdir("..")

    # Install ViennaPS
    if gpu_enabled:
        print("Installing ViennaPS with GPU support...")
        env["CMAKE_ARGS"] = "-DVIENNAPS_USE_GPU=ON"
    else:
        print("Installing ViennaPS (CPU-only)...")
        # Ensure GPU is disabled
        env["CMAKE_ARGS"] = "-DVIENNAPS_USE_GPU=OFF"

    os.chdir("ViennaPS")

    install_cmd = ["../" + str(pip_exe), "install", "."]
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
        description="Install ViennaTools package on Linux distributions and macOS"
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
    platform_id, version, platform_name = detect_os_platform()
    check_platform_support(platform_id, version)

    # Get package manager
    pkg_manager = get_package_manager(platform_id)
    if not pkg_manager:
        sys.exit(1)

    if platform_id == "macos" and args.gpu:
        print("Warning: GPU support on macOS is not supported. Disabling GPU support.")
        args.gpu = False

    check_gcc_compilers(platform_id, args.gpu)
    check_cuda_version(args.gpu)

    # Install system dependencies
    install_system_dependencies(platform_id, pkg_manager)

    # Set up ViennaTools directory and repositories
    original_dir = os.getcwd()
    setup_viennatools_directory(args.viennals_version, args.viennaps_version)

    # Set up virtual environment
    venv_path = setup_virtual_environment(".venv")

    # Install packages
    activate_and_install_packages(venv_path, args.verbose, args.gpu, platform_id)

    # Return to original directory
    os.chdir(original_dir)

    build_type = "GPU-enabled" if args.gpu else "CPU-only"
    print(
        f"Installation complete ({build_type}). To activate the virtual environment, run:"
    )

    print(f"source ViennaTools/.venv/bin/activate")
    print("To deactivate the virtual environment, run:")
    print("deactivate")


if __name__ == "__main__":
    main()
