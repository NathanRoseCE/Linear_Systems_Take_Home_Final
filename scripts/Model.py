import numpy as np


def linearWithFeedback(system: tuple[np.matrix],
                       k: np.matrix,
                       initialState: np.matrix,
                       dt: float,
                       inputs: list[np.matrix]) -> list[np.matrix]:
    """
    Models a linear system with state feedback, assumes system.A is without 
    feedback,
    \\dot x = Ax + Bu
    y = Cx + Du
    u = r + kx
    """
    outputs = []
    x = initialState
    A, B, C, D = system

    # Adjust A based off feedback
    A = A - (B*k)
    for r in inputs:
        dot_x = (A*x) + (B*r)
        x += dot_x*dt
        outputs.append((C*x) + D)
    return outputs
