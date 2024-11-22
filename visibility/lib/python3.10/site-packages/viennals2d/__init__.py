"""
ViennaLS
========

ViennaLS is a level set library developed for high performance
topography simulations. The main design goals are simplicity and efficiency,
tailored towards scientific simulations. ViennaLS can also be used for
visualisation applications, although this is not the main design target.
"""

import sys

def _windows_dll_path():
    
    import os
    # import vtk

    # vtk_path = vtk.__path__[0]

    additional_paths = [
        # os.path.join(os.path.dirname(vtk_path), 'vtk.libs'),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'viennals.libs')
    ]

    for path in additional_paths:
        os.add_dll_directory(path)
        os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]

if sys.platform == "win32":
    _windows_dll_path()

from .viennals2d import *
