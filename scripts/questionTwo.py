import numpy as np
import LatexFormat
import os
import Model
from control import lqr
from control import lyap
import json
from typing import List, Tuple
from Utilities import F, gen_inputs, graph_results, observer, renderTemplate

TWO_CONFIG_FILE = "resources/two.json"


def main(results: List[bool], index: int) -> None:
    config = {}
    with open(TWO_CONFIG_FILE, "r") as read_file:
        config = json.load(read_file)
    system = createSystem(config["system"])
    if not os.path.isdir("results"):
        os.mkdir("results")
    success = part_one(system, config)
    success &= part_two(system, config)
    results[index] = success
    output_overall_results(config)
    print("two success: " + str(results[index]))


def createSystem(config: json) -> Tuple[np.matrix]:
    """
    Creates a system from a given configuration
    """
    A = np.matrix(config["A"])
    B = np.matrix(config["B"])
    C = np.matrix(config["C"])
    D = np.matrix(config["D"])
    return A, B, C, D


def part_one(system: Tuple[np.matrix], config: json) -> bool:
    A, B, C, D = system
    q = np.matrix(config["Q"])
    r = np.matrix(config["R"])
    k = feedback(system, q, r)
    initialState = np.matrix(config["x_0"])
    dt = np.matrix(config["dt"])
    stopTime = np.matrix(config["stopTime"])
    inputs = [np.matrix([[0.0]]) for i in np.arange(0, stopTime, dt)]
    timeSteps = [float(timeStep) for timeStep in np.arange(0, stopTime, dt)]
    system_dynamics = Model.linearWithFeedback(system, k, initialState, 0.01, inputs)
    success = validResultsOne(config, k, system_dynamics, timeSteps)
    graph_results(timeSteps,
                  system_dynamics,
                  f'{config["outdir"]}/{config["one_graph"]}',
                  'Question 2-1')
    if success:
        output_results_one(config, system, q, r, k)
    return success


def part_two(system: Tuple[np.matrix], config: json) -> bool:
    A, B, C, D = system
    k = feedback(system,
                 np.matrix(np.matrix(config["Q"])),
                 np.matrix(config["R"]))
    x_0 = np.matrix(config["x_0"])
    x_e_0 = np.matrix(config["x_e_0"])
    inputs, timeSteps = gen_inputs(config["stopTime"], config["dt"])

    # redundant code to get intermediate values
    L0 = np.matrix(config["L"])
    f = F(config["desired_eigenvalues"])
    T = lyap(-f, A, -L0*C)
    L = np.linalg.inv(T)*L0
    assert np.all(L == observer(system, config["desired_eigenvalues"], L0))

    system_dynamics = Model.linearFullObserverWithFeedback(system, L, k, x_0, x_e_0, 0.01, inputs)
    success = validResultsTwo(config, L, system_dynamics, timeSteps)
    graph_results(timeSteps,
                  system_dynamics,
                  f'{config["outdir"]}/{config["two_graph"]}',
                  'Question 2-2')
    if success:
        output_results_two(config, system, f, T, L0, L, x_e_0)
    return success


def feedback(system: Tuple[np.matrix], q: np.matrix, r: np.matrix) -> np.matrix:
    """
    Gets the desired LQR feedback matrix(k) for a given system
    """
    A, B, C, D = system
    k, s, e = lqr(A, B, C.T*q*C, r)
    return k


def validResultsOne(config: json, k: np.matrix, outputs: np.matrix, times: List[float]) -> bool:
    """
    Checks to ensure that the results are valid for question 2-1
    """
    if np.any(abs(k) > config["kMax"]):
        print("K matrix has values that are to large:")
        print("k = " + str(k))
        return False
    timesToCheck = np.where(np.array(times) > config["out_limit"]["start"])
    for time in timesToCheck[0]:
        output = outputs[time]
        if np.any(np.matrix(output) > config["out_limit"]["start"]):
            print("Value outside acceptable range")
            return False
    return True


def validResultsTwo(config: json, L: np.matrix, outputs: np.matrix, times: List[float]) -> bool:
    """
    Checks to ensure that the results are valid for question 2-2
    """
    if np.any(abs(L) > config["lMax"]):
        print("L matrix has values that are to large:")
        print("L = " + str(L))
        return False
    return True


def output_results_one(config: json,
                       system: Tuple[np.matrix],
                       Q: np.matrix,
                       R: np.matrix,
                       k: np.matrix):
    templateFile = f'{config["templatedir"]}/{config["tex_one_fragment"]}.j2'
    renderTemplate(templateFile,
                   f'{config["outdir"]}/{config["tex_one_fragment"]}',
                   system=system,
                   R=R,
                   Q=Q,
                   k=k,
                   image_path=config["one_graph"])


def output_results_two(config: json,
                       system: Tuple[np.matrix],
                       F: np.matrix,
                       T: np.matrix,
                       L0: np.matrix,
                       L: np.matrix,
                       x_e_0: np.matrix):
    templateFile = f'{config["templatedir"]}/{config["tex_two_fragment"]}.j2'
    renderTemplate(templateFile,
                   f'{config["outdir"]}/{config["tex_two_fragment"]}',
                   system=system,
                   F=F,
                   T=T,
                   L0=L0,
                   L=L,
                   x_e_0=x_e_0,
                   image_path=config["two_graph"])


def output_overall_results(config: json):
    templateFile = f'{config["templatedir"]}/{config["tex_fragment"]}.j2'
    renderTemplate(templateFile,
                   f'{config["outdir"]}/{config["tex_fragment"]}',
                   two_one=config["tex_one_fragment"],
                   two_two=config["tex_two_fragment"])


if __name__ == '__main__':
    result = [False]
    main(result, 0)
    if not result[0]:
        exit(1)
