import json
import numpy as np
import LatexFormat
import os
import Model
from typing import Tuple, List
from Utilities import F, gen_inputs, graph_results, feedback, renderTemplate
from control import lyap
from numpy.linalg import inv
from math import pi, degrees
from multiprocessing import Process, Array
from NonLinearFragment import nonLinearUpdate, nonLinearOutput

THREE_CONFIG_FILE = "resources/three.json"


def main(results: List[bool], index: int) -> None:
    config = {}
    with open(THREE_CONFIG_FILE, "r") as read_file:
        config = json.load(read_file)
    results[index] = (three_one(config) and
                      three_two(config) and
                      three_three(config) and
                      three_four(config))
    output_overall_results(config)
    print("three success: " + str(results[index]))


def three_one(config: json) -> bool:
    system = createSystem(config, True)
    print_results_one(config, system)
    return True


def three_two(config: json) -> bool:
    system = createSystem(config)
    A, B, C, D = system

    # redundant code to get intermediate values
    f = F(config["desired_eig"])
    K0 = np.matrix(config["K0"])
    T = lyap(A, -f, -B*K0)
    k = K0 * inv(T)

    # should be the same but I need the intermediate variables
    assert np.all(k == feedback(system, config["desired_eig"], K0))

    x_0 = config["two_initial"]
    inputs, timeSteps = gen_inputs(config["stopTime"], config["dt"])
    outputs = Model.linearWithFeedback(system, k, x_0, config["dt"], inputs)
    graph_results(timeSteps, outputs, config["two_graph"], "3-2", ["Y", "Theta"])
    print_results_two(config, system, f, K0, k)
    return True


def three_three(config: json) -> bool:
    system = createSystem(config)
    A, B, C, D = system
    K0 = np.matrix(config["K0"])
    k = feedback(system, config["desired_eig"], K0)
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
    return True


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
                [nonLinear.item((0, 0))],
                [linear.item((0, 0))]
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
    results = Array('b', [False for i in range(numThreads)])
    for i in range(numThreads):
        threads[i] = Process(target=thetaStable, args=(thetas[i], config, k, results, i))
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
    templateFile = f'{config["templatedir"]}/{config["tex_one_fragment"]}.j2'
    renderTemplate(templateFile,
                   f'{config["outdir"]}/{config["tex_one_fragment"]}',
                   system=system,
                   x = [["y"],
                        [r"\dot y"],
                        [r"\theta"],
                        [r"\dot \theta"]])


def print_results_two(config: json,
                      system: Tuple[np.matrix],
                      f: np.matrix,
                      k0: np.matrix,
                      k: np.matrix):
    templateFile = f'{config["templatedir"]}/{config["tex_two_fragment"]}.j2'
    renderTemplate(templateFile,
                   f'{config["outdir"]}/{config["tex_two_fragment"]}',
                   system=system,
                   k0=k0,
                   f=f,
                   k=k,
                   graph=config["two_graph"])


def print_results_three(config: json,
                        x0Sample: np.matrix,
                        theta_limit: float):
    code_str = ""
    with open("scripts/NonLinearFragment.py") as file:
        code_str = file.read()
    templateFile = f'{config["templatedir"]}/{config["tex_three_fragment"]}.j2'
    renderTemplate(templateFile,
                   f'{config["outdir"]}/{config["tex_three_fragment"]}',
                   code_str=code_str,
                   graph=config["three_graph"],
                   stopTime=config["stopTime"],
                   threshold=config["threshold"],
                   tolerence=(1*(10**-LatexFormat.ROUND_TO)),
                   theta_limit_rad=theta_limit,
                   theta_limit_deg=degrees(theta_limit))


def print_results_four(config: json):
    templateFile = f'{config["templatedir"]}/{config["tex_four_fragment"]}.j2'
    renderTemplate(templateFile,
                   f'{config["outdir"]}/{config["tex_four_fragment"]}',
                   graph=config["four_graph"])


def output_overall_results(config: json):
    templateFile = f'{config["templatedir"]}/{config["tex_fragment"]}.j2'
    renderTemplate(templateFile,
                   f'{config["outdir"]}/{config["tex_fragment"]}',
                   three_one=config["tex_one_fragment"],
                   three_two=config["tex_two_fragment"],
                   three_three=config["tex_three_fragment"],
                   three_four=config["tex_four_fragment"])


if __name__ == '__main__':
    result = [False]
    main(result, 0)
    if not result[0]:
        exit(1)
