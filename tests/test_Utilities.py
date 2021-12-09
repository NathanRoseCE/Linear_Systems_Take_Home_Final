import numpy as np
import pytest
from typing import Tuple
import scripts.Utilities as util


def test_F_real():
    desired_eigs = [
        {
            "real": 1
        },
        {
            "real": 2
        }
    ]
    f = util.F(desired_eigs)
    f_exp = np.array([
        [1, 0],
        [0, 2]])
    assert np.all(f == f_exp)


def test_F_complex():
    desired_eigs = [
        {
            "real": 2,
            "imaginary": 3,
        },
        {
            "real": 10
        },
        {
            "real": 4,
            "imaginary": 5,
        }
    ]
    f = util.F(desired_eigs)
    f_exp = np.array([
        [2, 3, 0, 0, 0],
        [-3, 2, 0, 0, 0],
        [0, 0, 10, 0, 0],
        [0, 0, 0, 4, 5],
        [0, 0, 0, -5, 4]
    ])
    assert np.all(f == f_exp)


def test_gen_inputs():
    stop_time = 20
    dt = 0.01
    stepMagn = 4
    inputs, timesteps = util.gen_inputs(stop_time, dt, stepMagn)
    assert len(inputs) == (stop_time/dt)
    assert len(timesteps) == (stop_time/dt)
    for input in inputs:
        assert input.item((0, 0)) == stepMagn


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
        []
    ])
    return A, B, C, D


def test_feedback(system):
    A, B, C, D = system
    k0 = np.matrix([
        [1, 2, 10],
        [2, 3, 39]])
    desiredEigs = [
        {
            "real": -10
        },
        {
            "real": -2,
            "imaginary": 2
        }
    ]
    k = util.feedback(system, desiredEigs, k0)
    bar_A = A - B*k
    actual_eigs = np.linalg.eigvals(bar_A)

    assert np.any(np.isclose(-2+2j, actual_eigs, atol=1e-8))
    assert np.any(np.isclose(-2-2j, actual_eigs, atol=1e-8))
    assert np.any(np.isclose(-10, actual_eigs, atol=1e-8))


def test_observer(system):
    A, B, C, D = system
    l0 = np.matrix([
        [1, 2],
        [2, 5],
        [3, 4]])
    desiredEigs = [
        {
            "real": -20
        },
        {
            "real": -3,
            "imaginary": 2
        }
    ]
    L = util.observer(system, desiredEigs, l0)
    bar_A = A - L*C
    actual_eigs = np.linalg.eigvals(bar_A)

    assert np.any(np.isclose(-3+2j, actual_eigs, atol=1e-8))
    assert np.any(np.isclose(-3-2j, actual_eigs, atol=1e-8))
    assert np.any(np.isclose(-20, actual_eigs, atol=1e-8))


def test_eigs():
    eigs = [
        {
            "real": 2,
            "imaginary": 3
        },
        {
            "real": -4
        }
    ]
    actual_eigs = util.eigs(eigs)
    assert np.any(np.isclose(2+3j, actual_eigs, atol=1e-8))
    assert np.any(np.isclose(2-3j, actual_eigs, atol=1e-8))
    assert np.any(np.isclose(-4, actual_eigs, atol=1e-8))
