import numpy as np
import os


def matrix(matrix: np.matrix, latexMatrixType: str) -> str:
    """ 
    returns a latex bmatrix formatted string
    """
    lines = str(matrix).replace('[', '').replace(']', '').splitlines()
    rv = [os.linesep + r'\begin{' + latexMatrixType + '}']
    rv += ['  ' + ' & '.join(line.split()) + r'\\' for line in lines]
    rv += [r'\end{' + latexMatrixType + '}']
    return '\n'.join(rv)


def bmatrix(mat: np.matrix) -> str:
    return matrix(mat, "bmatrix")


def system(system: tuple[np.matrix]) -> str:
    A, B, C, D = system
    result = ""
    result += r"\begin{equation}" + os.linesep
    result += r"\dot x = " + bmatrix(A) + "x + "
    result += bmatrix(B) + "u" + os.linesep
    result += r"\end{equation}" + os.linesep
    result += r"\begin{equation}" + os.linesep
    result += r"\dot x = " + bmatrix(C) + "x + "
    result += bmatrix(D) + "u" + os.linesep
    result += r"\end{equation}" + os.linesep
    return result
