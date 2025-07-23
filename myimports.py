# myimports.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy import array, reshape, hstack
from numpy.linalg import eigh, inv
import sympy as sp
from IPython.display import Math, display

import inspect

def show_mat(M, name=None):
    if name is None:
        # look in the caller's locals for an object with the same id
        frame_locals = inspect.currentframe().f_back.f_locals
        for k, v in frame_locals.items():
            if v is M:
                name = k
                break
        if name is None:
            name = "M"  # fallback
    display(Math(rf"{name} = {sp.latex(sp.Matrix(M))}"))
