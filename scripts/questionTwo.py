import numpy as np
import LatexFormat
import os
import Model
from control import lqr
from control import lyap
import matplotlib.pyplot as plt
import json
from typing import List, Tuple

TWO_CONFIG_FILE = "resources/two.json"


def linearSystem(config: json):
    A = np.matrix(config["A"])
    B = np.matrix(config["B"])
    C = np.matrix(config["C"])
    D = np.matrix(config["D"])
    return A, B, C, D


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
    print("all results valid")
    return True


def validResultsTwo(config: json, L: np.matrix, outputs: np.matrix, times: List[float]) -> bool:
    """
    Checks to ensure that the results are valid for question 2-2
    """
    if np.any(abs(L) > config["lMax"]):
        print("L matrix has values that are to large:")
        print("L = " + str(L))
        return False
    print("all results valid")
    return True


def feedback(system: Tuple[np.matrix], q: np.matrix, r: np.matrix) -> np.matrix:
    A, B, C, D = system
    k, s, e = lqr(A, B, C.T*q*C, r)
    return k


def gen_inputs(stopTime: float, dt: float) -> Tuple[List[float]]:
    return ([np.matrix([[1.0]]) for i in np.arange(0, stopTime, dt)],
            [timeStep for timeStep in np.arange(0, stopTime, dt)])


def part_one(system: Tuple[np.matrix], one_config: json) -> bool:
    A, B, C, D = system
    q = np.matrix(one_config["Q"])
    r = np.matrix(one_config["R"])
    k = feedback(system, q, r)
    initialState = np.matrix(one_config["x_0"])
    dt = np.matrix(one_config["dt"])
    stopTime = np.matrix(one_config["stopTime"])
    inputs = [np.matrix([[0.0]]) for i in np.arange(0, stopTime, dt)]
    timeSteps = [float(timeStep) for timeStep in np.arange(0, stopTime, dt)]
    system_dynamics = Model.linearWithFeedback(system, k, initialState, 0.01, inputs)
    success = validResultsOne(one_config, k, system_dynamics, timeSteps)
    graph_results(timeSteps, system_dynamics, success, "results/two_one_output.png", 'Question 2-1')
    if success:
        output_results_one(system, q, r, "results/two_one.tex")
    return success


def part_two(system: Tuple[np.matrix], config: json) -> bool:
    A, B, C, D = system
    k = feedback(system,
                 np.matrix(np.matrix(config["Q"])),
                 np.matrix(config["R"]))
    x_0 = np.matrix(config["x_0"])
    x_e_0 = np.matrix(config["x_e_0"])
    inputs, timeSteps = gen_inputs(config["stopTime"], config["dt"])
    L0 = np.matrix(config["L"])
    f = F(config["desired_eigenvalues"], A.shape)
    T = lyap(-f, A, -L0*C)
    L = np.linalg.inv(T)*L0
    system_dynamics = Model.linearFullObserverWithFeedback(system, L, k, x_0, x_e_0, 0.01, inputs)
    success = validResultsTwo(config, L, system_dynamics, timeSteps)
    graph_results(timeSteps, system_dynamics, success, "results/two_two_output.png",'Question 2-2')
    if success:
        output_results_two(system, f, T, L0, L, x_e_0,  "results/two_two.tex")
    return success


def F(desiredEigens: List[dict], shape):
    F = np.zeros(shape, float)
    i = 0
    for desired_eig in desiredEigens:
        if "imaginary" in desired_eig:
            real = desired_eig["real"]
            imaginary = desired_eig["imaginary"]
            F[i][i] = real
            F[i+1][i] = imaginary
            F[i][i+1] = -imaginary
            i = i + 1
        F[i][i] = real
        i = i + 1
    return F


def main() -> bool:
    config = {}
    with open(TWO_CONFIG_FILE, "r") as read_file:
        config = json.load(read_file)
    system = linearSystem(config["system"])
    if not os.path.isdir("results"):
        os.mkdir("results")
    success = part_one(system, config)
    success &= part_two(system, config)
    return success


def graph_results(timeSteps: List[float], outputs: np.matrix, success:bool, save_file: str, title_name):
    fig = plt.figure()
    axis = fig.add_axes([0.1, 0.1, 0.75, 0.75])
    outputs = np.concatenate(outputs, axis=1)
    for i, output in enumerate(outputs):
        axis.plot(timeSteps, output.T, label=f"output {i}")
    axis.set_title(title_name)
    axis.set_xlabel('Time(s)')
    axis.set_ylabel('System outputs')
    axis.legend()
    if success:
        print("Success")
        fig.savefig(save_file)
    else:
        print("failure")
        fig.show()


def output_results_one(system: Tuple[np.matrix],
                       Q: np.matrix,
                       R: np.matrix,
                       outputFile: str):
    A, B, C, D = system
    with open(outputFile, "w") as out:
        out.write("For the system described by: " + os.linesep)
        out.write(LatexFormat.system(system))
        out.write(r" and $Q = " + LatexFormat.bmatrix(Q) + r"$,")
        out.write(r"$R = " + LatexFormat.bmatrix(R) + r"$" + os.linesep)
        out.write(os.linesep)
        out.write(r"The following is the outputs of the LQR system " +
                  r"assuming the inputs are 0" + os.linesep + os.linesep)
        out.write(r"\image{two_one_output.png}{LQR system}" +
                  r"{fig:two_one}")


def output_results_two(system: Tuple[np.matrix],
                       F: np.matrix,
                       T: np.matrix,
                       L0: np.matrix,
                       L: np.matrix,
                       x_e_0: np.matrix,
                       outputFile: str):
    A, B, C, D = system
    with open(outputFile, "w") as out:
        out.write("For the system described by: " + os.linesep)
        out.write(LatexFormat.system(system))
        out.write(r"The following variables were chosen for the observer:" + os.linesep + os.linesep)
        out.write(r"\begin{tabular}{r|l}" + os.linesep)
        out.write(r"$L_0$ & $" + LatexFormat.bmatrix(L0) + r"$\\" + os.linesep)
        out.write(r"$F$ & $" + LatexFormat.bmatrix(F) + r"$\\" + os.linesep)
        out.write(r"\end{tabular}" + os.linesep + os.linesep)
        out.write(r"These were used to calculate: " + os.linesep + os.linesep)
        out.write(r"\begin{tabular}{r|l}" + os.linesep)
        out.write(r"$T$ & $" + LatexFormat.bmatrix(T) + r"$\\" + os.linesep)
        out.write(r"$L$ & $" + LatexFormat.bmatrix(L) + r"$\\" + os.linesep)
        out.write(r"\end{tabular}" + os.linesep)
        out.write(os.linesep)
        out.write(r"The following is the outputs of the LQR system " +
                  r"assuming the inputs are 0 and an initial estimate of "
                  r"state of $" + LatexFormat.bmatrix(x_e_0) + "$" + os.linesep + os.linesep)
        out.write(r"\image{two_two_output.png}{LQR system}" +
                  r"{fig:two_two}")


if __name__ == '__main__':
    if not main():
        exit(1)
