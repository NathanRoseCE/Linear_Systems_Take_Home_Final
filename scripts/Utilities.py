from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
from control import lyap
from numpy.linalg import inv


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


def graph_results(timeSteps: List[float],
                  outputs: np.matrix,
                  save_file: str,
                  title_name: str,
                  output_names: List[str] = None):
    fig = plt.figure()
    axis = fig.add_axes([0.1, 0.1, 0.75, 0.75])
    outputs = np.concatenate(outputs, axis=1)
    if output_names is None:
        output_names = []
        for i in range(len(outputs)):
            output_names.append(f"output {i}")

    for output, name in zip(outputs, output_names):
        axis.plot(timeSteps, output.T, label=name)
    axis.set_title(title_name)
    axis.set_xlabel('Time(s)')
    axis.set_ylabel('System outputs')
    axis.legend()
    fig.savefig(save_file)


def feedback(system: Tuple[np.matrix], desiredEig: Dict[str, float], K0: np.matrix) -> np.matrix:
    A, B, C, D = system
    f = F(desiredEig, A.shape)
    T = lyap(A, -f, -B*K0)
    K0 = np.matrix(K0)
    T = lyap(A, -f, -B*K0)
    return K0 * inv(T)


def observer(system: Tuple[np.matrix], desiredEig: Dict[str, float], L0: np.matrix) -> np.matrix:
    A, B, C, D = system
    f = F(desiredEig, A.shape)
    T = lyap(-f, A, -L0*C)
    return np.linalg.inv(T)*L0
    
