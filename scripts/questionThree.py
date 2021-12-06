import json
import numpy as np
import LatexFormat
import os
import Model
from typing import Tuple
from Utilities import F, gen_inputs, graph_results
from control import lyap
from numpy.linalg import inv
from math import sin, cos, pi

THREE_CONFIG_FILE = "resources/three.json"


def main() -> bool:
    config = {}
    with open(THREE_CONFIG_FILE, "r") as read_file:
        config = json.load(read_file)
    return three_one(config) and three_two(config) and three_three(config)


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
    graph_results(timeSteps, outputs, config["two_graph"], "3-2")
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
    print(k)
    outputs = Model.ModelSystem(nonLinearUpdate,
                                nonLinearOutput,
                                x_0,
                                inputs,
                                k,
                                config,
                                config["dt"])
    print("model complet")
    graph_results(timeSteps, outputs, config["three_graph"], "Nonlinear with feedback sample")
    print("graphed")
    print_results_three(config, x_0, 4.0)
    print("output")
    return True


def thetaStable(theta, config: json, k: np.matrix) -> bool:
    """ 
    This function is used to determine if the system is stable at a given theta
    """
    pass


def nonLinearUpdate(x: np.matrix, r: np.matrix, k: np.matrix, config: json, dt: float) -> np.matrix:
    y = x.item((0, 0))
    dot_y = x.item((1, 0))
    theta = x.item((2, 0))
    dot_theta = x.item((3, 0))
    u = r - (k*x)
    n = u.item((0, 0))
    M = float(config["M"])
    m = float(config["m"])
    L = float(config["l"])
    g = float(config["g"])

    # non-linear model
    accels = inv(np.matrix([
        [M+m, m*L],
        [1, L]
    ])) * np.matrix([
        [n],
        [g*theta]
    ])

    ddot_y = accels.item((0, 0))
    ddot_theta = accels.item((1, 0))
    # ddot_y = (n/M) - (m*g*y/M)
    # temp_one = ((g/L) + (g*m/(M*L)))
    # temp_two = -1.0/(M*L)
    # ddot_theta = theta*temp_one + n*temp_two
    # dx = np.matrix([
    #     [dot_y],
    #     [ddot_y],
    #     [dot_theta],
    #     [ddot_theta]
    # ])
    A, B, C, D = createSystem(config)
    dx = A*x + B*n
    # print("y: 0 vs " + str(A.item((3, 0))))
    # print("doty: 0 vs " + str(A.item((3, 1))))
    # print("theta:" + str(temp_one) + " vs " + str(A.item((3, 2))))
    # print("dottheta: 0 vs " + str(A.item((3, 3))))
    # print("n: " + str(temp_two) + " vs " + str(B.item((3, 0))))
    # print("matrix: " + str(dx.item((3, 0))))
    # print("manual: " + str(ddot_theta))
    dx.itemset((0, 0), dot_y)
    dx.itemset((1, 0), ddot_y)
    dx.itemset((2, 0), dot_theta)
    dx.itemset((3, 0), ddot_theta)
    next_x = x + (dx*dt)

    # ensure theta is in range -pi -> pi
    next_theta = next_x.item((2, 0))
    while (next_theta) > pi:
        raise "collapsed"
        next_theta = next_theta - (2*pi)
    while (next_theta) < -pi:
        raise "collapsed"
        next_theta = next_theta + (2*pi)
    next_x.itemset((2, 0), next_theta)

    return next_x


def nonLinearOutput(x: np.matrix, r: np.matrix, k: np.matrix, config: json, dt: float) -> np.matrix:
    y = x.item((0, 0))
    dot_y = x.item((1, 0))
    theta = x.item((2, 0))
    dot_theta = x.item((3, 0))
    return np.matrix([
        [y],
        [theta]
    ])


def createSystem(config: json, latex_string: bool = False):
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
    with open(config["tex_three_fragment"], 'w') as out:
        out.writelines([
            "The non-linear system was implimented with the following code" + os.linesep,
            r"\TODO{add code fragment}" + os.linesep,
            "A sample with starting inputs of: $" +
            LatexFormat.bmatrix(x0Sample) + "$ produces the followung outputs" + os.linesep,
            r"\image{" + config["three_graph"].split('/')[1] + r"}{3-3 system}{fig:3-3}" + os.linesep,
            "For the limit, the system was considered stabalized if after " + str(config["stopTime"]),
            r" seconds the system state variable $\theta$ was within $\pm" + str(config["threshold"]),
            "$ and the result of this is a theta limit of: " + str(theta_limit) + os.linesep
        ])

if __name__ == '__main__':
    if not main():
        exit(1)
