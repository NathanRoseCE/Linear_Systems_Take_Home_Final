from scripts import questionOne
import numpy as np


def test_createSystem_values():
    config = {}
    config["sigma_1"] = 2
    config["sigma_2"] = 3
    config["alpha_1"] = 5
    config["alpha_2"] = 7
    config["g"] = 11
    config["n"] = 13
    A_exp = np.matrix([
        [0, 1, 0, 0],
        [0, -3, 11, -7],
        [0, 0, 0, 1],
        [0, -2, 0, -5]
    ])
    B_exp = np.matrix([
        [0],
        [11],
        [0],
        [13]
    ])
    C_exp = np.matrix([
        [1, 0, 0, 0],
        [0, 0, 1, 0]
    ])
    D_exp = np.matrix([
        [0],
        [0]
    ])
    A, B, C, D = questionOne.createSystem(config)
    assert np.all(A_exp == A)
    assert np.all(B_exp == B)
    assert np.all(C_exp == C)
    assert np.all(D_exp == D)
