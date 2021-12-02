import numpy as np
import LatexFormat
import os
import Model
from control import lqr
import matplotlib.pyplot as plt
import json

TWO_CONFIG_FILE = "resources/two.json"


def linearSystem(config: json):
    A = np.matrix(config["A"])
    B = np.matrix(config["B"])
    C = np.matrix(config["C"])
    D = np.matrix(config["D"])
    return A, B, C, D


def validResults(config: json, k: np.matrix, outputs: np.matrix, times: list[float]) -> bool:
    """
    Checks to ensure that the results are valid for question 2
    """
    if np.any(k > config["kMax"]):
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


def part_one(system: tuple[np.matrix], one_config: json) -> bool:
    A, B, C, D = system
    q = np.matrix(one_config["Q"])
    r = np.matrix(one_config["R"])
    initialState = np.matrix(one_config["initialState"])
    dt = np.matrix(one_config["dt"])
    stopTime = np.matrix(one_config["stopTime"])
    k, s, e = lqr(A, B, C.T*q*C, r)
    inputs = [np.matrix([[0.0]]) for i in np.arange(0, stopTime, dt)]
    timeSteps = [float(timeStep) for timeStep in np.arange(0, stopTime, dt)]
    system_dynamics = Model.linearWithFeedback(system, k, initialState, 0.01, inputs)
    success = validResults(one_config, k, system_dynamics, timeSteps)
    graph_results(timeSteps, system_dynamics, success)
    if success:
        output_results(system, q, r, "results/two_one.tex")
    return success

def main():
    config = {}
    with open(TWO_CONFIG_FILE, "r") as read_file:
        config = json.load(read_file)
    system = linearSystem(config["system"])
    if not os.path.isdir("results"):
        os.mkdir("results")
    part_one(system, config["one"])


def graph_results(timeSteps: list[float], outputs: np.matrix, success:bool):
    outputs = np.concatenate(outputs, axis=1)
    for i, output in enumerate(outputs):
        plt.plot(timeSteps, output.T, label=f"output {i}")
    plt.title('Question 2-1')
    plt.xlabel('Time(s)')
    plt.ylabel('System outputs')
    plt.legend()
    if success:
        print("Success")
        plt.savefig("results/two_one_output.png")
    else:
        print("failure")
        plt.show()


def output_results(system: tuple[np.matrix],
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


if __name__ == '__main__':
    main()
