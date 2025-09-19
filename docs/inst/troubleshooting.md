---
layout: default
title: Troubleshooting
parent: Installing the Library
nav_order: 2
---

# Troubleshooting
{: .fs-9 .fw-500 }

---

## Failed Python package build

The following error can occur while building the Python package using pip:

```
Building wheels for collected packages: ViennaPS_Python
  Building wheel for ViennaPS_Python (pyproject.toml) ... error
  error: subprocess-exited-with-error

  × Building wheel for ViennaPS_Python (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> [21 lines of output]
      *** scikit-build-core 0.9.3 using CMake 3.27.4 (wheel)
      *** Configuring CMake...
      loading initial cache file build/CMakeInit.txt
      -- CPM: Adding package PackageProject@1.11.1 (v1.11.1)
      ninja: error: Makefile:5: expected '=', got ':'
      default_target: all
                    ^ near here

      CMake Error at /usr/share/cmake-3.27/Modules/FetchContent.cmake:1662 (message):
        Build step for packageproject failed: 1
      Call Stack (most recent call first):
        /usr/share/cmake-3.27/Modules/FetchContent.cmake:1802:EVAL:2 (__FetchContent_directPopulate)
        /usr/share/cmake-3.27/Modules/FetchContent.cmake:1802 (cmake_language)
        build/cmake/CPM_0.38.6.cmake:1004 (FetchContent_Populate)
        build/cmake/CPM_0.38.6.cmake:798 (cpm_fetch_package)
        CMakeLists.txt:98 (CPMAddPackage)


      -- Configuring incomplete, errors occurred!

      *** CMake configuration failed
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for ViennaPS_Python
Failed to build ViennaPS_Python
ERROR: Could not build wheels for ViennaPS_Python, which is required to install pyproject.toml-based projects
```

This error is due to a conflict with the _Ninja_ build system and _Unix Makefiles_. To resolve this error, you can remove the build folder and then rerun pip. However, please note that this action will also remove all dependencies if they were installed alongside ViennaPS.

## Python ImportError
The following error can occur when trying to import the ViennaPS Python package:

```
ImportError: arg(): could not convert default argument 'boundary: viennahrle::BoundaryType' in method '<class 'viennaps2d.viennaps2d.Domain'>.__init__' into a Python object (type not registered yet?)
```

This error indicates the your ViennaPS Python package is not compatible with the installed ViennaLS Python package. This can happen if you have installed the ViennaLS Python package from PyPI and then built the ViennaPS Python package from source. To resolve this issue, you can either uninstall the ViennaPS Python package or build the ViennaLS Python package from source as well.
Alternatively, you can install the ViennaPS Python package from PyPI, which will ensure compatibility with the installed ViennaLS Python package. To do this, run the following command:

```bash
pip install ViennaPS
```
This will install the latest version of the ViennaPS Python package from PyPI, which should be compatible with the installed ViennaLS Python package.

Table of compatibility between ViennaPS and ViennaLS versions:

| ViennaPS Package | ViennaLS Package  | Compatible |
|:------------------|:-------------------| :--------|
| Local Build      | Local Build       | Yes &#9989;    |
| Local Build      | PyPI              | No &#10060;    |
| PyPI             | Local Build       | No &#10060;    |
| PyPI             | PyPI              | Yes &#9989;    |


## Windows Python DLL ImportError

If you see the following error when importing the ViennaPS Python package:

```
ImportError: DLL load failed while importing viennaps2d: The specified module could not be found.
```

This usually means that a required shared library (`.dll`) is missing. The most common cause on Windows is a missing OpenMP runtime.

### Solution

Make sure the **OpenMP runtime** is available on your system. Specifically, the file `libomp140.x86_64.dll` must be accessible through your system `PATH`.

You can get it from:

* Visual Studio (in `debug_nonredist` folder)
* Prebuilt LLVM distributions for Windows

After downloading or locating the DLL:

* Either copy it into the folder containing `_core.cp*.pyd`
* Or add the folder containing the DLL to your system `PATH`

Once the DLL is accessible, the import should work correctly.
