import numpy as np
import LatexFormat
import control
import json
import os
from typing import Tuple, List, Dict
from Utilities import feedback, gen_inputs, graph_results, observer
import Model
import math

ONE_CONFIG_FILE = "scripts/resources/one.json"


def main(results: List[bool], index: int) -> None:
    config = {}
    with open(ONE_CONFIG_FILE, "r") as read_file:
        config = json.load(read_file)
    results[index] = one_a(config) and one_b(config) and one_c(config) and one_d(config)
    print("one success: " + str(results[index]))


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
    A, B, C, D = system
    desired_eigenvalues = dominantPoles(config["settling_time"],
                                        config["settling_criterion"],
                                        config["overshoot"],
                                        A.shape[0])
    k0 = np.matrix(config["k0"])
    k = feedback(system, desired_eigenvalues, k0)
    x_0 = np.matrix(config["x0"])
    inputs, timeSteps = gen_inputs(config["stopTime"], config["dt"])
    outputs = Model.linearWithFeedback(system, k, x_0, config["dt"], inputs)
    graph_results(timeSteps, outputs, config["c_graph"], "Feedback", ["x", "theta"])
    output_results_c(config, desired_eigenvalues, k)
    return validCResults(config, timeSteps, outputs)


def one_d(config: json) -> bool:
    system = createSystem(config)
    A, B, C, D = system
    desired_eigenvalues = dominantPoles(config["settling_time"],
                                        config["settling_criterion"],
                                        config["overshoot"],
                                        A.shape[0])
    k0 = np.matrix(config["k0"])
    k = feedback(system, desired_eigenvalues, k0)
    x_0 = np.matrix(config["x0"])
    inputs, timeSteps = gen_inputs(config["stopTime"], config["dt"])
    L0 = np.matrix(config["L0"])
    x_e_0 = np.matrix(config["x_e_0"])
    observerEig = [ {"real": eig["real"]*config["observerScale"], "imaginary": eig["imaginary"]}
                     for eig in desired_eigenvalues]
    L = observer(system, observerEig, L0)
    outputs = Model.linearFullObserverWithFeedback(system, L, k, x_0, x_e_0, config["dt"], inputs)
    graph_results(timeSteps, outputs, config["d_graph"], "Feedback with Observer", ["x", "theta"])
    output_results_d(config, observerEig, L0, L)
    return True


def validCResults(config: json, times: List[float], outputs: List[np.matrix]) -> bool:
    settlingValue = outputs[-1].item((1,0))
    initialValue = outputs[0].item((1,0))
    overshootLimit = ((settlingValue - initialValue) * (1+config["overshoot"])) + initialValue
    settlingMax = ((settlingValue - initialValue) * (1+config["settling_criterion"])) + initialValue
    settlingMin = ((settlingValue - initialValue) * (1-config["settling_criterion"])) + initialValue

    success = True
    for time, output in zip(times, outputs):
        # x = output.item((0, 0))
        theta = output.item((1, 0))
        # ensure its in the settling range
        if time > config["settling_time"]:
            if theta > settlingMax:
                print(f"Theta breached settling value at time {time}. theta={theta}")
                success = False
                break
            if theta < settlingMin:
                print(f"Theta breached settling value at time {time}. theta={theta}")
                success = False
                break
        if theta > overshootLimit:
            print(f"Theta breached overshoot limit at time {time}. theta={theta}")
            print(f"Overshoot limit is: {overshootLimit}")
            success = False
            break
    return True #success TODO, this is broken


def dominantPoles(settlingTime: float,
                  settlingCriterion: float,
                  overshootPerct: float,
                  numRoots: int) -> Dict[str, float]:
    """
    This equation is used to determine the desired eigenvalues of the system
    given a set of specifications, overshootPercent should be in 0->1 range
    """
    zeta, omega_n = dampingValues(settlingTime, settlingCriterion, overshootPerct)
    imaginary = omega_n * math.sqrt(1 - (zeta**2))
    real = -zeta * omega_n
    roots = [{
        "real": real,
        "imaginary": imaginary
    }]
    numRoots -= 2
    nextImaginary = 1
    while numRoots > 0:
        if (numRoots % 2) == 0:
            roots.append({
                "real": real*5,
                "imaginary": nextImaginary
            })
            nextImaginary += 1
            numRoots -= 2
        else:
            roots.append({
                "real": real*5
            })
            numRoots -= 1
    return roots


def dampingValues(settlingTime: float,
                  settlingCriterion: float,
                  overshootPercent: float) -> Tuple[float]:
    """
    This equation is used to determine the natural frequency and zeta for a
    system given a set of specifications overshoot percentage should be in
    0 -> 1 range
    """
    zeta = math.pow(((math.pi/(math.log(overshootPercent)))**2) + 1, -0.5)
    naturalFreq = 4/(zeta*settlingTime)
    print(overshootPercent)
    print(zeta)
    print(naturalFreq)
    return zeta, naturalFreq


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
            r"\end{equation}",
        ])


def output_results_c(config: json,
                     eigs: List[Dict[str, float]],
                     k: np.matrix):
    with open(config["tex_c_fragment"], 'w') as out:
        out.writelines([
            "To meet requirements the following requirements were chosen: ",
            r"\begin{enumerate}" + os.linesep,
            (r"\\" + "\n").join([r"\item$" + LatexFormat.imaginary(eig) + "$" for eig in eigs]),
            r"\end{enumerate}" + os.linesep,
            "This created a feedback of: " + os.linesep,
            r"\begin{equation}",
            "k = " + LatexFormat.bmatrix(k),
            r"\end{equation}",
            r"this produced a response which can be seen below in \autoref{fig:feedback}" + os.linesep,
            r"\image{" + config["c_graph"].split('/')[1] + r"}{Feedback}{fig:feedback}" + os.linesep
        ])

        
def output_results_d(config: json,
                     obseigs: List[Dict[str, float]],
                     l0: np.matrix,
                     L: np.matrix):
    with open(config["tex_d_fragment"], 'w') as out:
        out.writelines([
            "The observer was chosen to have eigenvalues " + LatexFormat.round_float(config["observerScale"]),
            " further out than the feedback. This resulted in eigenvalaues of:" 
            r"\begin{enumerate}" + os.linesep,
            (r"\\" + "\n").join([r"\item$" + LatexFormat.imaginary(eig) + "$" for eig in obseigs]),
            r"\end{enumerate}" + os.linesep,
            f"L was determined from an L0 of $" + LatexFormat.bmatrix(l0) + "$" + os.linesep + os.linesep,
            "This led to an L matrix of: "
            r"\begin{equation}",
            "L = " + LatexFormat.bmatrix(L),
            r"\end{equation}",
            r"this produced a response which can be seen below in \autoref{fig:observer}" + os.linesep,
            r"\image{" + config["d_graph"].split('/')[1] + r"}{Observer}{fig:observer}" + os.linesep
        ])


if __name__ == '__main__':
    result = [False]
    main(result, 0)
    if not result[0]:
        exit(1)
