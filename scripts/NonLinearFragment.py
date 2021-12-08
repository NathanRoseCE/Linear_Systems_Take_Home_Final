import numpy as np
from numpy.linalg import inv
from math import cos, sin, pi
import json


def nonLinearUpdate(x: np.matrix,
                    r: np.matrix,
                    k: np.matrix,
                    config: json,
                    dt: float) -> np.matrix:
    """
    This function is used to model the non-linear system, this will throw an
    exception if there is a rollover
    """
    # a list of variables that make my life easier
    dot_y = x.item((1, 0))
    theta = x.item((2, 0))
    dot_theta = x.item((3, 0))
    u = r - (k*x)
    n = u.item((0, 0))
    M = float(config["M"])
    m = float(config["m"])
    L = float(config["l"])
    g = float(config["g"])

    # non linear update code
    accels = inv(np.matrix([
        [M+m, m*L*cos(theta)],
        [cos(theta), L]
    ])) * np.matrix([
        [n + m*L*theta*theta*sin(theta)],
        [g*sin(theta)]
    ])
    ddot_y = accels.item((0, 0))
    ddot_theta = accels.item((1, 0))
    dx = np.matrix([
        [dot_y],
        [ddot_y],
        [dot_theta],
        [ddot_theta]
    ])
    next_x = x + (dx*dt)

    # rage quit if it falls over lol
    next_theta = next_x.item((2, 0))
    while (next_theta) > pi:
        raise AssertionError("collapsed")
        next_theta = next_theta - (2*pi)
    while (next_theta) < -pi:
        raise AssertionError("collapsed")
        next_theta = next_theta + (2*pi)
    next_x.itemset((2, 0), next_theta)

    return next_x


def nonLinearOutput(x: np.matrix, r: np.matrix, k: np.matrix, config: json, dt: float) -> np.matrix:
    """
    A simple output function, just grabs the theta stat variable
    """
    theta = x.item((2, 0))
    return np.matrix([
        [theta]
    ])
