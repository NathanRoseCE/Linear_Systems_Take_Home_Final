import numpy as np
import LatexFormat
import control
import json
import os
from typing import Tuple

ONE_CONFIG_FILE = "resources/one.json"


def main() -> bool:
    config = {}
    with open(ONE_CONFIG_FILE, "r") as read_file:
        config = json.load(read_file)
    return one_a(config) and one_b(config)


def createSystem(config: json, latex_string: bool = False):
    neg_sigma_1 = -config["sigma_1"] if not latex_string else r"-\sigma_1"
    neg_sigma_2 = -config["sigma_2"] if not latex_string else r"-\sigma_2"
    neg_alpha_1 = -config["alpha_1"] if not latex_string else r"-\alpha_1"
    neg_alpha_2 = -config["alpha_2"] if not latex_string else r"-\alpha_1"
    g = config["g"] if not latex_string else r"g"
    n = config["n"] if not latex_string else r"n"
    A = [
        [0, 1, 0, 0],
        [0, neg_sigma_2, g, neg_alpha_2],
        [0, 0, 0, 1],
        [0, neg_sigma_1, 0, neg_alpha_1]
    ]
    B = [
        [0],
        [g],
        [0],
        [n]
    ]
    C = [
        [1, 0, 0, 0],
        [0, 0, 1, 0]
    ]
    D = [
        [0],
        [0]
    ]
    if not latex_string:
        return np.matrix(A), np.matrix(B), np.matrix(C), np.matrix(D)
    else:
        return A, B, C, D


def one_a(config: json) -> bool:
    system = createSystem(config, True)
    output_results_a(config["tex_a_fragment"], system)
    return True


def one_b(config: json) -> bool:
    system = createSystem(config)
    A, B, C, D = system
    transfer_function = control.ss2tf(A, B, C, D)
    output_results_b(config["tex_b_fragment"], system, transfer_function)
    return True


def one_c(config: json) -> bool:
    system = createSystem(config)
    print("Dont know how to solve C")
    return True

def one_c(config: json) -> bool:
    system = createSystem(config)
    
    return True

def output_results_a(outfile: str,
                     system: Tuple[np.matrix]):
    A, B, C, D = system
    with open(outfile, 'w') as out:
        out.writelines([
            "For the system with state variables:",
            r"\begin{equation}" + os.linesep,
            r"x = " + os.linesep,
            r"\begin{bmatrix}" + os.linesep,
            r"x\\" + os.linesep,
            r"\dot x\\" + os.linesep,
            r"\theta\\" + os.linesep,
            r"\dot \theta\\" + os.linesep,
            r"\end{bmatrix}",
            r"\end{equation}",
            "The state space representation is: ",
            LatexFormat.system(system)
        ])


def output_results_b(outfile: str,
                     system: Tuple[np.matrix],
                     transfer: control.tf):
    with open(outfile, 'w') as out:
        out.writelines([
            "For the system described by: ",
            LatexFormat.system(system),
            "",
            "A realization was determined by taking the realization ",
            "of the state space representation. The result of this is: ",
            r"\begin{equation}",
            "g(s) = " + LatexFormat.transferFunction(transfer),
            r"\end{equation}"
        ])


if __name__ == '__main__':
    if not main():
        exit(1)
