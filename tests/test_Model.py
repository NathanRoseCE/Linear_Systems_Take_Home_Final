from typing import Tuple
import pytest
import numpy as np
import Model
import Utilities as util
from copy import deepcopy


@pytest.fixture
def system() -> Tuple[np.matrix]:
    """
    Gives a fully controllable and fully observable linear
    system
    """
    A = np.matrix([
        [1, 2, 5],
        [9, 4, 91],
        [1, 32, 89]
    ])
    B = np.matrix([
        [4, 3],
        [1, 4],
        [10, 2]
    ])
    C = np.matrix([
        [1.2, 32, 34],
        [12, 34, 31]
    ])
    D = np.matrix([
        [3, 19]
    ])
    return A, B, C, D


def test_linearOutput(system: Tuple[np.matrix]):
    A, B, C, D = system
    x_0 = np.matrix([
        [3],
        [4],
        [5]])
    dt = 0.01
    r = np.matrix([
        [-1],
        [1]
    ])
    y_n = Model._linearOutput(x_0, r, system, dt)
    assert np.all(y_n == (C*x_0) + (D*r))


def test_ModelSystem(system: Tuple[np.matrix]):
    x_0 = np.matrix([
        [1],
        [2],
        [3]
    ])
    inputs = [
        np.matrix([
            [1],
            [1]]),
        np.matrix([
            [2],
            [2]])
    ]
    dt = 0.1
    x = deepcopy(x_0)
    exp_outputs = []
    for u in inputs:
        x = Model._linearUpdate(x, u, system, dt)
        exp_outputs.append(Model._linearOutput(x, u, system, dt))

    act_outs = Model.ModelSystem(Model._linearUpdate,
                                 Model._linearOutput,
                                 x_0,
                                 inputs,
                                 system,
                                 dt)
    for exp, act in zip(exp_outputs, act_outs):
        assert np.all(exp == act)

