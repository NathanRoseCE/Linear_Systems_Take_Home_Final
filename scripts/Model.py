import numpy as np
from copy import deepcopy
from typing import Tuple, List
from typing import Callable


def ModelSystem(update: Callable[[np.matrix, np.matrix], np.matrix],
                output: Callable[[np.matrix, np.matrix], np.matrix],
                x_0: np.matrix,
                inputs: List[np.matrix],
                *args) -> List[Tuple]:
    """
    A generic model for any system can work with linear and nonlinear
    systems. update functions returns the next state, and output outputs
    the output matrix. Both of these functions take in state, system input,
    then *args.
    This function will return list of outputs

    This is an underlying function see other models for examples
    """
    outputs = []
    x = deepcopy(x_0)
    inputs = deepcopy(inputs)
    for r in inputs:
        x = update(x, r, *args)
        outputs.append(output(x, r, *args))
    return outputs


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
    x_0 = deepcopy(initialState)
    A, B, C, D = deepcopy(system)
    k = deepcopy(k)
    dt = deepcopy(dt)
    inputs = deepcopy(inputs)

    # Adjust A based off feedback
    A = A - (B*k)
    return ModelSystem(_linearUpdate,
                       _linearOutput,
                       x_0,
                       inputs,
                       (A, B, C, D),
                       dt)


def linearFullObserverWithFeedback(system: Tuple[np.matrix],
                                   L: np.matrix,
                                   k: np.matrix,
                                   x_0: np.matrix,
                                   x_e_0: np.matrix,
                                   dt: float,
                                   inputs: List[np.matrix]) -> List[np.matrix]:
    """
    Models a linear system with state feedback and a full state observer, this is only valid if
    D is a zero matrix
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
    if not np.all(D == 0):
        raise ValueError("D matrix should be all 0's")
    x = deepcopy(x_0)
    k = np.matrix(k)

    x_e = deepcopy(x_e_0)
    A_est = A - L*C
    B_est = np.identity(A.shape[0])
    C_est = np.identity(A.shape[0])
    D_est = np.identity(A.shape[0])
    sys_est = (A_est, B_est, C_est, D_est)
    for r in inputs:
        # feedback step
        u = r - (k*x_e)

        # system step
        x = _linearUpdate(x, u, (A, B, C, D), dt)
        y = _linearOutput(x, u, (A, B, C, D), dt)

        # estimator step
        u_est = (B*u) + (L*y)
        x_e = _linearUpdate(x_e, u_est, sys_est, dt)
        outputs.append(y)
    return outputs


def _linearUpdate(x: np.matrix,
                  r: np.matrix,
                  system: Tuple[np.matrix],
                  dt: float) -> np.matrix:
    """
    Linear update step, determines the next state for a linera system
    """
    A, B, C, D = system
    dx = (A*x) + (B*r)
    return x + (dx*dt)


def _linearOutput(x: np.matrix,
                  r: np.matrix,
                  system: Tuple[np.matrix],
                  dt: float) -> np.matrix:
    """
    Linear output step, determine the output for a linear system
    """
    A, B, C, D = system
    return (C*x) + (D*r)

