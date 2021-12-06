from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt


def F(desiredEigens: List[dict], shape):
    """
    Generates a proper F matrix that follows imaginary number rules
    """
    F = np.zeros(shape, float)
    i = 0
    for desired_eig in desiredEigens:
        real = desired_eig["real"]
        if "imaginary" in desired_eig:
            imaginary = desired_eig["imaginary"]
            F[i][i] = real
            F[i+1][i] = imaginary
            F[i][i+1] = -imaginary
            i = i + 1
        F[i][i] = real
        i = i + 1
    return F


def gen_inputs(stopTime: float, dt: float) -> Tuple[List[float]]:
    """
    Generated the inputs for the system
    """
    return ([np.matrix([[1.0]]) for i in np.arange(0, stopTime, dt)],
            [timeStep for timeStep in np.arange(0, stopTime, dt)])


def graph_results(timeSteps: List[float], outputs: np.matrix, save_file: str, title_name: str):
    fig = plt.figure()
    axis = fig.add_axes([0.1, 0.1, 0.75, 0.75])
    outputs = np.concatenate(outputs, axis=1)
    for i, output in enumerate(outputs):
        axis.plot(timeSteps, output.T, label=f"output {i}")
    axis.set_title(title_name)
    axis.set_xlabel('Time(s)')
    axis.set_ylabel('System outputs')
    axis.legend()
    fig.savefig(save_file)
