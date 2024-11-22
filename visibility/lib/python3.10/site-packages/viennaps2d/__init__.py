"""
ViennaPS
========

ViennaPS is a header-only C++ process simulation library,
which includes surface and volume representations, 
a ray tracer, and physical models for the simulation of 
microelectronic fabrication processes.
"""

import sys

def _windows_dll_path():
    import os

    additional_paths = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'viennaps.libs')
    ]

    for path in additional_paths:
        if not os.path.exists(path):
            continue

        os.add_dll_directory(path)
        os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]

if sys.platform == "win32":
    _windows_dll_path()

from .viennaps2d import *
import viennals2d as ls

# Config file reader helper function
def ReadConfigFile(fileName: str):
    """Read a config file in the ViennaPS standard config file format.

    Parameters
    ----------
    fileName: str
              Name of the config file.
    
    Returns
    -------
    dict
        A dictionary containing the parameters from the config file.
    """
    par_dict = {}

    with open(fileName, "r") as file:
        lines = file.readlines()
        for line in lines:
            
            line = line[:line.find('#')] # remove comments

            if len(line) > 0:
                par_name = line[:line.find('=')].strip(' ')
                par_value = line[line.find('=')+1:]

                try:
                    val = float(par_value) 
                except: val = par_value

                par_dict[par_name] = val

    return par_dict
