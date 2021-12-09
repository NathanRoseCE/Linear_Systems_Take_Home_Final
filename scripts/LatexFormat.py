import numpy as np
import os
from control import tf
from typing import List, Tuple, Dict
import Utilities as utils

ROUND_TO = 3


def round_float(val: float) -> str:
    """
    rounds the float to ROUND_TO decimal places and casts to a string
    """
    return str(round(val, ROUND_TO))


utils.registerTemplateFilter("round_float", round_float)


def imaginary(val: Dict[str, float]) -> str:
    """
    rounds an imaginary number to ROUND_TO decimal places and
    casts to a string
    """
    return (round_float(val["real"]) +
            r" \pm " +
            round_float(val["imaginary"]) +
            "j")


utils.registerTemplateFilter("imaginary", imaginary)


def matrix_List(vals: List, latexMatrixType: str) -> str:
    latexMatrixString = r"\begin{" + latexMatrixType + r"}" + os.linesep
    for i, row in enumerate(vals):
        for j, column in enumerate(row):
            if type(column) == float:
                column = round_float(column)
            latexMatrixString += str(column)
            if not j == len(row)-1:
                latexMatrixString += r"&"
        if not i == len(vals) - 1:
            latexMatrixString += r"\\"
        latexMatrixString += os.linesep
    latexMatrixString += r"\end{" + latexMatrixType + r"}"
    return latexMatrixString


def bmatrix(mat: List[List[float]]) -> str:
    if type(mat).__module__ == np.__name__:
        mat = mat.tolist()
    return matrix_List(mat, "bmatrix")


utils.registerTemplateFilter("bmatrix", bmatrix)


def system(system: Tuple[np.matrix]) -> str:
    A, B, C, D = system
    result = ""
    result += r"\begin{equation}" + os.linesep
    result += r"\dot x = " + bmatrix(A) + "x + "
    result += bmatrix(B) + "u" + os.linesep
    result += r"\end{equation}" + os.linesep
    result += r"\begin{equation}" + os.linesep
    result += r"y = " + bmatrix(C) + "x + "
    result += bmatrix(D) + "u" + os.linesep
    result += r"\end{equation}" + os.linesep
    return result


utils.registerTemplateFilter("formatSystem", system)


def transferFunction(transfer: tf) -> str:
    scipyLTI = transfer.returnScipySignalLTI()
    lti_response = []
    for output in scipyLTI:
        out_responses = []
        for input in output:
            out_responses.append(frac("s", input.num, input.den))
        lti_response.append(out_responses)
    return bmatrix(lti_response)


utils.registerTemplateFilter("transferFunction", transferFunction)


def frac(var: str, numerator: List[float], denominator: List[float]) -> str:
    frac_str = r"\frac{"
    frac_str += polynomial(var, numerator)
    frac_str += r"}{"
    frac_str += polynomial(var, denominator)
    frac_str += r"}"
    return frac_str


utils.registerTemplateFilter("fraction", frac)


def polynomial(var: str, coefficents: List[float]) -> str:
    poly_str = ""
    for i, num in enumerate(coefficents):
        rounded = float(round_float(num))
        if rounded != 0:
            sign = ""
            if not i == 0:
                sign = (" + " if rounded > 0 else " - ")
            poly_str += sign + str(abs(rounded))
            exponent = len(coefficents)-i-1
            if not exponent == 0:
                poly_str += "s"
                if not exponent == 1:
                    poly_str += r"^{" + str(exponent) + r"}"
    return poly_str


utils.registerTemplateFilter("polynomial", polynomial)
