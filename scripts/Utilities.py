from typing import List, Tuple, Dict, Callable
import numpy as np
import matplotlib.pyplot as plt
import control
import itertools
from numpy.linalg import inv
from jinja2 import Environment, FileSystemLoader
import os


def F(desiredEigens: List[dict]):
    """
    Generates a proper F matrix that follows imaginary number rules
    """
    complex = [complex for complex in desiredEigens
               if ("imaginary" in complex) and (complex["imaginary"] != 0)]
    # complex roots are double counted intentionally
    size = len(desiredEigens) + len(complex)
    F = np.zeros((size, size), float)
    i = 0
    for desired_eig in desiredEigens:
        real = desired_eig["real"]
        if "imaginary" in desired_eig:
            imaginary = desired_eig["imaginary"]
            F[i][i] = real
            imaginary = abs(imaginary)
            F[i+1][i] = -imaginary
            F[i][i+1] = imaginary
            i = i + 1
        F[i][i] = real
        i = i + 1
    return F


def gen_inputs(stopTime: float, dt: float, stepMagn: float = 1.0) -> Tuple[List[float]]:
    """
    Generated the inputs for the system
    """
    return ([np.matrix([[stepMagn]]) for i in np.arange(0, stopTime, dt)],
            [timeStep for timeStep in np.arange(0, stopTime, dt)])


def graph_results(timeSteps: List[float],
                  outputs: np.matrix,
                  save_file: str,
                  title_name: str,
                  output_names: List[str] = None):
    """
    Graphs the results over time
    """
    fig = plt.figure()
    axis = fig.add_axes([0.15, 0.15, 0.75, 0.75])
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


def feedback(system: Tuple[np.matrix],
             desiredEig: Dict[str, float],
             K0: np.matrix) -> np.matrix:
    """
    Gets the feedback matrix K for a system
    will satisfy eigvals(A - BK) == desiredEigs
    """
    A, B, C, D = system
    if not controllable(A, B):
        raise ValueError("{A,B} must be controllable")
    f = F(desiredEig)
    T = control.lyap(A, -f, -B*K0)
    K = K0 * inv(T)
    desiredEigs = eigs(desiredEig)
    actualEigs = np.linalg.eigvals(A-B*K)
    assert np.all(
        [np.any(desiredEig in actualEigs for desiredEig in desiredEigs)]
    )
    return K


def observer(system: Tuple[np.matrix],
             desiredEig: Dict[str, float],
             L0: np.matrix) -> np.matrix:
    """
    Gets the feedback matrix L for a system
    will satisfy eigvals(A - LC) == desiredEigs
    """
    A, B, C, D = system
    if not observable(A, C):
        print(f"A = {A}")
        print(f"C = {C}")
        raise ValueError("{A,C} must be observable")
    f = F(desiredEig)
    T = control.lyap(-f, A, -L0*C)
    L = np.linalg.inv(T)*L0
    desiredEigs = eigs(desiredEig)
    actualEigs = np.linalg.eigvals(A-L*C)
    assert np.all(
        [np.any(desiredEig in actualEigs for desiredEig in desiredEigs)]
    )
    return L


"""
This is the private jinja environment that is used to render
the templates
"""
_latex_jinja_env = Environment(
    block_start_string=r'\PYTHON{',
    block_end_string='}',
    variable_start_string=r'\PY{',
    variable_end_string='}',
    comment_start_string='#{',
    comment_end_string='}',
    line_statement_prefix='%-',
    line_comment_prefix='%#',
    trim_blocks=True,
    autoescape=False,
    loader=FileSystemLoader(os.path.abspath('.'))
)


def registerTemplateFilter(name: str,
                           filter: Callable[[any], str]):
    """
    This function is used to registser a jinja filter
    for the templates
    """
    _latex_jinja_env.filters[name] = filter


def renderTemplate(templateFile: str,
                   outFile: str,
                   **args):
    """
    renders a jinja file located at template file and writes it to
    out file. The args are passed to the template engine
    """
    template = _latex_jinja_env.get_template(templateFile)
    with open(outFile, "w") as out:
        out.write(template.render(**args))


def eigs(eigs: Dict[str, float]) -> List[complex]:
    """
    This function is used to turn the dictonary complex numbers into the
    native python version
    """
    return_eigs = []
    for eig in eigs:
        eigVal = eig["real"]
        if ("imaginary" in eig) and (eig["imaginary"] != 0):
            return_eigs.append(eigVal + (1j*eig["imaginary"]))
            return_eigs.append(eigVal - (1j*eig["imaginary"]))
            continue
        return_eigs.append(eigVal)
    return return_eigs


def controllable(A: np.matrix, B: np.matrix) -> bool:
    """
    Determines if a given system is controllable
    """
    g_c = control.ctrb(A, B)
    vectors = [np.array(g_c[:, i])[np.newaxis].T for i in range(g_c.shape[1])]
    permutations = itertools.combinations(vectors, A.shape[1])
    return not np.any(0 == np.array([np.linalg.det(np.concatenate(permutation, axis=1))
                                     for permutation in permutations]))


def observable(A: np.matrix, C: np.matrix) -> bool:
    """
    Determines if a given system is observable
    """
    g_o = control.obsv(A, C)
    vectors = [np.array(g_o[i][np.newaxis]) for i in range(g_o.shape[0])]
    permutations = itertools.combinations(vectors, A.shape[0])
    perms = [perm for perm in permutations]
    return np.any(0 != np.array([np.linalg.det(np.concatenate(perm, axis=0))
                                 for perm in perms]))
