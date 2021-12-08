import json
import numpy as np
import LatexFormat
import os
import Model
from typing import Tuple, List, Dict
from Utilities import F, gen_inputs, graph_results, feedback
from control import lyap
from numpy.linalg import inv
from math import sin, cos, pi, degrees
from threading import Thread
from NonLinearFragment import nonLinearUpdate, nonLinearOutput

THREE_CONFIG_FILE = "scripts/resources/three.json"


def main(results: List[bool], index: int) -> None:
    config = {}
    with open(THREE_CONFIG_FILE, "r") as read_file:
        config = json.load(read_file)
    results[index] = (three_one(config) and
                      three_two(config) and
                      three_three(config) and
                      three_four(config))
    print("three success: " + str(results[index]))


def three_one(config: json) -> bool:
    system = createSystem(config, True)
    print_results_one(config, system)
    return True


def three_two(config: json) -> bool:
    system = createSystem(config)
    A, B, C, D = system
    f = F(config["desired_eig"], A.shape)
    K0 = np.matrix(config["K0"])
    T = lyap(A, -f, -B*K0)
    k = K0 * inv(T)
    x_0 = config["two_initial"]
    inputs, timeSteps = gen_inputs(config["stopTime"], config["dt"])
    outputs = Model.linearWithFeedback(system, k, x_0, config["dt"], inputs)
    graph_results(timeSteps, outputs, config["two_graph"], "3-2", ["Y", "Theta"])
    print_results_two(config, system, f, K0, k)
    return True


def three_three(config: json) -> bool:
    system = createSystem(config)
    A, B, C, D = system
    f = F(config["desired_eig"], A.shape)
    K0 = np.matrix(config["K0"])
    T = lyap(A, -f, -B*K0)
    k = K0 * inv(T)
    x_0 = np.matrix(config["three_sample_x0"])
    inputs, timeSteps = gen_inputs(config["stopTime"], config["dt"])
    outputs = Model.ModelSystem(nonLinearUpdate,
                                nonLinearOutput,
                                x_0,
                                inputs,
                                k,
                                config,
                                config["dt"])
    graph_results(timeSteps, outputs, config["three_graph"], "Nonlinear feedback sample", ["theta"])
    thetaLimit = theta_limit(config, k)
    print_results_three(config, x_0, thetaLimit)
    results = [True, False]
    thetaStable(0, config, k, results, 0) and not thetaStable(3, config, k, results, 1)
    return results[0] and not results[1]


def three_four(config: json) -> bool:
    system = createSystem(config, onlyTheta=True)
    A, B, C, D = system
    k = feedback(system, config["desired_eig"], np.matrix(config["K0"]))
    x_0 = np.matrix(config["four_x0"])
    inputs, timeSteps = gen_inputs(config["stopTime"], config["dt"])
    nonlin_out = Model.ModelSystem(nonLinearUpdate,
                                   nonLinearOutput,
                                   x_0,
                                   inputs,
                                   k,
                                   config,
                                   config["dt"])
    lin_out = Model.linearWithFeedback(system, k, x_0, config["dt"], inputs)
    outs = []
    zipped = zip(nonlin_out, lin_out)
    for nonLinear, linear in zipped:
        outs.append(
            np.matrix([
                [nonLinear.item((0,0))],
                [linear.item((0,0))]
            ])
        )
    graph_results(timeSteps, outs, config["four_graph"], "Comparison", ["non-linear", "linear"])
    print_results_four(config)
    return True


def theta_limit(config: json, k: np.matrix, minTheta: float = 0, maxTheta: float = pi) -> float:
    """ 
    This is a function that will find the stable limit for theta, assumes the min is stable
    and the max is not
    """
    if abs(maxTheta - minTheta) < 10**-(LatexFormat.ROUND_TO):
        return maxTheta
    numThreads = int(config["threads"])

    thetas = np.arange(minTheta, maxTheta, (maxTheta-minTheta)/numThreads)
    threads = [None] * numThreads
    results = [None] * numThreads
    for i in range(numThreads):
        threads[i] = Thread(target=thetaStable, args=(thetas[i], config, k, results, i))
        threads[i].start()

    for i in range(numThreads):
        threads[i].join()

    i = 0
    while results[i]:
        i = i + 1
    return theta_limit(config, k, thetas[i-1], thetas[i])


def thetaStable(theta: float, config: json, k: np.matrix, result: List[bool], index: int) -> None:
    """ 
    This function is used to determine if the system is stable at a given theta,
    stores the result in bool
    """
    x_0 = np.matrix([
        [0],
        [0],
        [theta],
        [0]
    ])
    inputs, timeSteps = gen_inputs(config["stopTime"], config["dt"])
    try:
        outputs = Model.ModelSystem(nonLinearUpdate,
                                    nonLinearOutput,
                                    x_0,
                                    inputs,
                                    k,
                                    config,
                                    config["dt"])
        if outputs[-1].item((0, 0)) < config["threshold"]:
            result[index] = True
        else:
            result[index] = False
    except AssertionError:
        result[index] = False


def createSystem(config: json, latex_string: bool = False, onlyTheta: bool=False):
    if not latex_string:
        M = config["M"]
        m = config["m"]
        L = config["l"]
        g = config["g"]
        A = np.matrix([
            [0, 1, 0, 0],
            [-(m*g)/M, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, (g/L) + (g*m/(M*L)), 0]
        ])
        B = np.matrix([
            [0],
            [1/M],
            [0],
            [-1/(M*L)]
        ])
        C = []
        D = []
        if onlyTheta:
            C = np.matrix([
                [0, 0, 1, 0]
            ])
            D = np.matrix([
                [0]
            ])
        else:
            C = np.matrix([
                [1, 0, 0, 0],
                [0, 0, 1, 0]
            ])
            D = np.matrix([
                [0],
                [0]
            ])
    else:
        A = [
            [0, 1, 0, 0],
            [r"\frac{-mg}{M}", 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, r"\frac g l + \frac {gm} {Ml}", 0]
        ]
        B = [
            [0],
            [r"\frac 1 M"],
            [0],
            [r"-\frac 1 {m*l}"]
        ]
        if onlyTheta:
            C = [
                [0, 0, 1, 0]
            ]
            D = [
                [0]
            ]
        else:
            C = [
                [1, 0, 0, 0],
                [0, 0, 1, 0]
            ]
            D = [
                [0],
                [0]
            ]
    return A, B, C, D


def print_results_one(config: json, system: Tuple[str]):
    A, B, C, D = system
    with open(config["tex_one_fragment"], 'w') as out:
        out.writelines([
            "For the system with state matrix:" + os.linesep,
            r"\begin{equation}" + os.linesep,
            r"x = ",
            LatexFormat.bmatrix([["y"],
                                 [r"\dot y"],
                                 [r"\theta"],
                                 [r"\dot \theta"]
                                 ]),
            r"\end{equation}" + os.linesep,
            "The following system is described:" + os.linesep,
            LatexFormat.system(system)
        ])


def print_results_two(config: json,
                      system: Tuple[np.matrix],
                      f: np.matrix,
                      k0: np.matrix,
                      k: np.matrix):
    A, B, C, D = system
    with open(config["tex_two_fragment"], 'w') as out:
        out.writelines([
            "The input values were: " + os.linesep,
            r"\begin{tabular}{c|c}" + os.linesep,
            r"$K_0$ & $" + LatexFormat.bmatrix(k0) + r" $ \\",
            r"$F$ & $" + LatexFormat.bmatrix(f) + r" $ \\",
            r"\end{tabular}" + os.linesep,
            r"this produced a k of: " + os.linesep,
            r"\begin{equation}" + os.linesep,
            r"  k = " + LatexFormat.bmatrix(k),
            r"\end{equation}" + os.linesep + os.linesep,
            r"With this feedback value the following outputs were made:" + os.linesep,
            r"\image{" + config["two_graph"].split('/')[1] + r"}{3-2 system}{fig:3-2}" + os.linesep
        ])


def print_results_three(config: json,
                        x0Sample: np.matrix,
                        theta_limit: float):
    code_str = ""
    with open("scripts/NonLinearFragment.py") as file:
        code_str = file.read()
    with open(config["tex_three_fragment"], 'w') as out:
        out.writelines([
            "The non-linear system was implimented with the following code" + os.linesep,
            r"\begin{minted}{python3}" + os.linesep,
            code_str,
            r"\end{minted}" + os.linesep,
            "A sample with starting inputs of: $" +
            LatexFormat.bmatrix(x0Sample) + "$ produces the followung outputs" + os.linesep,
            r"\image{" + config["three_graph"].split('/')[1] + r"}{3-3 system}{fig:3-3}" + os.linesep,
            "For the limit, the system was considered stabalized if after " + str(config["stopTime"]),
            r" seconds the system state variable $\theta$ was within $\pm" + str(config["threshold"]),
            "$ of 0, and it had not fallen over it yet",
            ", the result of this is a theta limit of: " + LatexFormat.round_float(theta_limit),
            " radians which is: " + LatexFormat.round_float(degrees(theta_limit)) + " degrees" + os.linesep
        ])


def print_results_four(config: json):
    with open(config["tex_four_fragment"], 'w') as out:
        out.writelines([
            "The comparison betweeen the linear and nonlinear system can be seen" + os.linesep,
            r"in the system below in \autoref{fig:comparison}" + os.linesep,
            r"\image{" + config["four_graph"].split('/')[1] + r"}{Comparison}{fig:comparison}" + os.linesep
        ])


if __name__ == '__main__':
    result = [False]
    main(result, 0)
    if not result[0]:
        exit(1)
