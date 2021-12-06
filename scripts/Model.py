import numpy as np
from copy import deepcopy
from typing import Tuple, List


def linearWithFeedback(system: Tuple[np.matrix],
                       k: np.matrix,
                       initialState: np.matrix,
                       dt: float,
                       inputs: List[np.matrix]) -> List[np.matrix]:
    """
    Models a linear system with state feedback, assumes system.A is without 
    feedback,
    \\dot x = Ax + Bu
    y = Cx + Du
    u = r + kx
    output is of the observer
    """
    outputs = []
    x = deepcopy(initialState)
    A, B, C, D = deepcopy(system)
    k = deepcopy(k)
    dt = deepcopy(dt)
    inputs = deepcopy(inputs)

    # Adjust A based off feedback
    A = A - (B*k)
    for r in inputs:
        dot_x = (A*x) + (B*r)
        x += dot_x*dt
        outputs.append((C*x) + D)
    return outputs


def linearFullObserverWithFeedback(system: Tuple[np.matrix],
                                   L: np.matrix,
                                   k: np.matrix,
                                   x_0: np.matrix,
                                   x_e_0: np.matrix,
                                   dt: float,
                                   inputs: List[np.matrix]) -> List[np.matrix]:
    """
    Models a linear system with state feedback, assumes system.A is without
    feedback,
    \\dot x = Ax + Bu
    y = Cx + Du
    u = r + kx
    """
    L = deepcopy(L)
    k = deepcopy(k)
    dt = deepcopy(dt)
    inputs = deepcopy(inputs)
    outputs = []
    A, B, C, D = deepcopy(system)
    x = deepcopy(x_0)
    x_e = deepcopy(x_e_0)
    y_e = np.zeros((C.shape[0], 1))
    k = np.matrix(k)
    # A_est = A - L*C
    for r in inputs:
        # feedback step
        u = r - (k*x_e)

        # system step
        x, y = linearStep(x, u, dt, (A, B, C, D))

        # estimator step
        dot_x_e = (A*x_e) + (B*u) + L*(y-y_e)
        x_e += dot_x_e*dt
        y_e = (C*x_e) + D
        outputs.append(y)
    return outputs


def linearStep(x: np.matrix, u: np.matrix, dt: float, system: Tuple[np.matrix]) -> Tuple[np.matrix]:
    A, B, C, D = system
    dot_x = A*x + B*u
    x += dt * dot_x
    y = C*x + D*u
    return x, y
