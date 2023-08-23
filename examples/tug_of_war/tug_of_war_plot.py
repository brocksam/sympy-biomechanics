from collections import OrderedDict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from sympy.core.function import AppliedUndef
from sympy.physics._biomechanics import (
    MusculotendonDeGroote2016,
)


@dataclass
class TugOfWarData:
    x: AppliedUndef
    v: AppliedUndef
    musc_1: MusculotendonDeGroote2016
    musc_2: MusculotendonDeGroote2016
    replacements: dict | OrderedDict
    NUM_NODES: int | None = None
    DURATION: float | None = None


def plot_solution_opty(sol, ocp_data):
    data = _collect_data_opty(sol, ocp_data)
    data = _populate_data_other(data, ocp_data)
    _plot_solution(data)


def plot_solution_pycollo(sol, ocp_data):
    data = _collect_data_pycollo(sol, ocp_data)
    data = _populate_data_other(data, ocp_data)
    _plot_solution(data)


def _collect_data_opty(sol, ocp_data):
    data = {}
    x_slice = slice(0, ocp_data.NUM_NODES)
    v_slice = slice(ocp_data.NUM_NODES, 2*ocp_data.NUM_NODES)
    musc_1_a_slice = slice(2*ocp_data.NUM_NODES, 3*ocp_data.NUM_NODES)
    musc_2_a_slice = slice(3*ocp_data.NUM_NODES, 4*ocp_data.NUM_NODES)
    musc_1_e_slice = slice(4*ocp_data.NUM_NODES, 5*ocp_data.NUM_NODES)
    musc_2_e_slice = slice(5*ocp_data.NUM_NODES, 6*ocp_data.NUM_NODES)
    data['t'] = np.concatenate([np.linspace(0.0, ocp_data.DURATION, ocp_data.NUM_NODES), np.linspace(ocp_data.DURATION, 2*ocp_data.DURATION, ocp_data.NUM_NODES)])
    data['x'] = np.concatenate([sol[x_slice], -sol[x_slice]])
    data['v'] = np.concatenate([sol[v_slice], -sol[v_slice]])
    data['musc_1_a'] = np.concatenate([sol[musc_1_a_slice], sol[musc_2_a_slice]])
    data['musc_2_a'] = np.concatenate([sol[musc_2_a_slice], sol[musc_1_a_slice]])
    data['musc_1_e'] = np.concatenate([sol[musc_1_e_slice], sol[musc_2_e_slice]])
    data['musc_2_e'] = np.concatenate([sol[musc_2_e_slice], sol[musc_1_e_slice]])
    return data


def _collect_data_pycollo(sol, ocp_data):
    data = {}
    data['t'] = np.concatenate([sol._time_[0][:-1], np.array([t + 0.5 for t in sol._time_[0]])])
    data['x'] = np.concatenate([sol.state[0][0][:-1], -sol.state[0][0]])
    data['v'] = np.concatenate([sol.state[0][1][:-1], -sol.state[0][1]])
    data['musc_1_a'] = np.concatenate([sol.state[0][2][:-1], sol.state[0][3]])
    data['musc_2_a'] = np.concatenate([sol.state[0][3][:-1], sol.state[0][2]])
    data['musc_1_e'] = np.concatenate([sol.control[0][0][:-1], sol.control[0][1]])
    data['musc_2_e'] = np.concatenate([sol.control[0][1][:-1], sol.control[0][0]])
    return data


def _populate_data_other(data, ocp_data):
    kdes = {ocp_data.x.diff(me.dynamicsymbols._t): ocp_data.v}
    inputs = [ocp_data.x, ocp_data.v, ocp_data.musc_1.a, ocp_data.musc_2.a, ocp_data.musc_1.e, ocp_data.musc_2.e]
    outputs = [
        ocp_data.musc_1._F_T_tilde.doit().subs(kdes).xreplace(ocp_data.replacements),
        ocp_data.musc_2._F_T_tilde.doit().subs(kdes).xreplace(ocp_data.replacements),
        ocp_data.musc_1._l_M_tilde.doit().subs(kdes).xreplace(ocp_data.replacements),
        ocp_data.musc_2._l_M_tilde.doit().subs(kdes).xreplace(ocp_data.replacements),
    ]
    eval_other = sm.lambdify(inputs, outputs, cse=True)
    musc_1_F_T_tilde, musc_2_F_T_tilde, musc_1_l_M_tilde, musc_2_l_M_tilde = eval_other(
        data['x'],
        data['v'],
        data['musc_1_a'],
        data['musc_2_a'],
        data['musc_1_e'],
        data['musc_2_e'],
    )
    data['musc_1_F_T_tilde'] = musc_1_F_T_tilde
    data['musc_2_F_T_tilde'] = musc_2_F_T_tilde
    data['musc_1_l_M_tilde'] = musc_1_l_M_tilde
    data['musc_2_l_M_tilde'] = musc_2_l_M_tilde
    return data


def _plot_solution(data):
    plt.figure()
    plt.plot(data['t'], data['x'], color="tab:blue", label="Block Position")
    plt.plot(data['t'], data['v'], color="tab:olive", label="Block Velocity")
    plt.title("Dynamics States")
    plt.legend()

    plt.figure()
    plt.plot(data['t'], data['musc_1_F_T_tilde'], color="tab:brown", label="Muscle 1")
    plt.plot(data['t'], data['musc_2_F_T_tilde'], color="tab:grey", label="Muscle 2")
    plt.title("Normalised Tendon Forces")
    plt.legend()

    plt.figure()
    plt.plot(data['t'], data['musc_1_l_M_tilde'], color="tab:red", label="Muscle 1")
    plt.plot(data['t'], data['musc_2_l_M_tilde'], color="tab:pink", label="Muscle 2")
    plt.title("Normalised Fibre Lengths")
    plt.legend()

    plt.figure()
    plt.plot(data['t'], data['musc_1_a'], color="tab:green", label="Muscle 1")
    plt.plot(data['t'], data['musc_2_a'], color="tab:cyan", label="Muscle 2")
    plt.title("Muscle Activations")
    plt.legend()

    plt.figure()
    plt.plot(data['t'], data['musc_1_e'], color="tab:purple", label="Muscle 1")
    plt.plot(data['t'], data['musc_2_e'], color="tab:pink", label="Muscle 2")
    plt.title("Muscle Excitations")
    plt.legend()
