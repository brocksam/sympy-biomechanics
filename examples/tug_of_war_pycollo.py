"""Solve the tug of war OCP.

Notes
=====

- Needs Pycollo installed (including dependencies like CasADi and Pyproprop)

"""

import matplotlib.pyplot as plt
import numpy as np
import pycollo
import sympy as sm
import sympy.physics.mechanics as me
from sympy.physics._biomechanics import (
    FirstOrderActivationDeGroote2016,
    MusculotendonDeGroote2016,
)


WALL_OFFSET = 0.35
BLOCK_SIZE = 0.05

MASS = 20
DURATION = 1.0
DISTANCE = 0.08

OPTIMAL_FIBER_LENGTH = 0.25
MAXIMAL_FIBER_VELOCITY = 2.5
PEAK_ISOMETRIC_FORCE = 1000
TENDON_SLACK_LENGTH = 0.05
OPTIMAL_PENNATION_ANGLE = 0.0
FIBER_DAMPING_COEFFICIENT = 0.1
ACTIVATION_TIME_CONSTANT = 0.055
DEACTIVATION_TIME_CONSTANT = 0.065
SMOOTHING_RATE = 10

x, v = me.dynamicsymbols("x, v")
m, g = sm.symbols("m, g")
t = sm.Symbol("t")
T = sm.symbols("T")

kdes = {x.diff(me.dynamicsymbols._t): v}

tau_a, tau_d, b = sm.symbols('tau_a, tau_d, b')
l_T_slack, F_M_max, l_M_opt, v_M_max, alpha_opt, beta = sm.symbols('l_T_slack, F_M_max, l_M_opt, v_M_max, alpha_opt, beta')

P_origin = me.Point('pO')
N_global = me.ReferenceFrame('N')

musc_1_origin = P_origin.locatenew("musc_1_origin", WALL_OFFSET * -N_global.x)
musc_1_origin.set_vel(N_global, musc_1_origin.pos_from(P_origin).dt(N_global).subs(kdes))
musc_1_insertion = P_origin.locatenew("musc_1_insertion", (x - BLOCK_SIZE) * N_global.x)
musc_1_insertion.set_vel(N_global, musc_1_insertion.pos_from(P_origin).dt(N_global).subs(kdes))
musc_1_pathway = me.LinearPathway(musc_1_origin, musc_1_insertion)
musc_1_activation = FirstOrderActivationDeGroote2016(
    '1',
    activation_time_constant=tau_a,
    deactivation_time_constant=tau_d,
    smoothing_rate=b,
)
musc_1 = MusculotendonDeGroote2016(
    '1',
    musc_1_pathway,
    musc_1_activation,
    tendon_slack_length=l_T_slack,
    peak_isometric_force=F_M_max,
    optimal_fiber_length=l_M_opt,
    maximal_fiber_velocity=v_M_max,
    optimal_pennation_angle=alpha_opt,
    fiber_damping_coefficient=beta,
)
musc_2_origin = P_origin.locatenew("musc_2_origin", WALL_OFFSET * N_global.x)
musc_2_origin.set_vel(N_global, musc_2_origin.pos_from(P_origin).dt(N_global).subs(kdes))
musc_2_insertion = P_origin.locatenew("musc_2_insertion", (x + BLOCK_SIZE) * N_global.x)
musc_2_insertion.set_vel(N_global, musc_2_insertion.pos_from(P_origin).dt(N_global).subs(kdes))
musc_2_pathway = me.LinearPathway(musc_2_origin, musc_2_insertion)
musc_2_activation = FirstOrderActivationDeGroote2016(
    '2',
    activation_time_constant=tau_a,
    deactivation_time_constant=tau_d,
    smoothing_rate=b,
)
musc_2 = MusculotendonDeGroote2016(
    '2',
    musc_2_pathway,
    musc_2_activation,
    tendon_slack_length=l_T_slack,
    peak_isometric_force=F_M_max,
    optimal_fiber_length=l_M_opt,
    maximal_fiber_velocity=v_M_max,
    optimal_pennation_angle=alpha_opt,
    fiber_damping_coefficient=beta,
)

problem = pycollo.OptimalControlProblem("Tug of War")
phase_A = problem.new_phase("A")

phase_A.state_variables = (x, v) + (musc_1.x[0], ) + (musc_2.x[0], )
phase_A.control_variables = (musc_1.r[0], ) + (musc_2.r[0], )
phase_A.state_equations = {
    x: v,
    v: (musc_2.force.doit().subs(kdes) - musc_1.force.doit().subs(kdes)) / m,
    musc_1.x[0]: musc_1.rhs()[0],
    musc_2.x[0]: musc_2.rhs()[0],
}

phase_A.integrand_functions = [
    musc_1.a**2 + musc_2.a**2,
]

phase_A.bounds.initial_time = 0
phase_A.bounds.final_time = T / 2
phase_A.bounds.state_variables = {
    x: [-WALL_OFFSET, WALL_OFFSET],
    v: [-2, 2],
    musc_1.a: [0, 1],
    musc_2.a: [0, 1]
}
phase_A.bounds.control_variables = {
    musc_1.e: [0, 1],
    musc_2.e: [0, 1],
}
phase_A.bounds.integral_variables = [[0, 1]]
phase_A.bounds.initial_state_constraints = {
    x: -DISTANCE,
    v: 0,
}
phase_A.bounds.final_state_constraints = {
    x: DISTANCE,
    v: 0,
}

phase_A.guess.time = [0, DURATION / 2]
phase_A.guess.state_variables = [
    [-DISTANCE, DISTANCE],
    [0, 0],
    [0, 0],
    [0, 0],
]
phase_A.guess.control_variables = [
    [0, 0],
    [0, 0],
]
phase_A.guess.integral_variables = [0]

problem.objective_function = 2 * phase_A.integral_variables[0]

problem.endpoint_constraints = [
    phase_A.initial_state_variables.a_1 - phase_A.final_state_variables.a_2,
    phase_A.initial_state_variables.a_2 - phase_A.final_state_variables.a_1,
]
problem.bounds.endpoint_constraints = [[0, 0], [0, 0]]

problem.auxiliary_data = {
    m: MASS,
    T: DURATION,
    tau_a: ACTIVATION_TIME_CONSTANT,
    tau_d: DEACTIVATION_TIME_CONSTANT,
    b: SMOOTHING_RATE,
    l_T_slack: TENDON_SLACK_LENGTH,
    F_M_max: PEAK_ISOMETRIC_FORCE,
    l_M_opt: OPTIMAL_FIBER_LENGTH,
    v_M_max: MAXIMAL_FIBER_VELOCITY,
    alpha_opt: OPTIMAL_PENNATION_ANGLE,
    beta: FIBER_DAMPING_COEFFICIENT,
}

problem.settings.quadrature_method = "lobatto"
problem.settings.max_mesh_iterations = 1
problem.settings.scaling_method = "bounds"

phase_A.mesh.number_mesh_section_nodes = 4
phase_A.mesh.number_mesh_sections = 10

problem.solve()

data = {}
sol = problem.solution
data[t] = np.concatenate([sol._time_[0][:-1], np.array([t + 0.5 for t in sol._time_[0]])])
data[x] = np.concatenate([sol.state[0][0][:-1], -sol.state[0][0]])
data[v] = np.concatenate([sol.state[0][1][:-1], -sol.state[0][1]])
data[musc_1.a] = np.concatenate([sol.state[0][2][:-1], sol.state[0][3]])
data[musc_2.a] = np.concatenate([sol.state[0][3][:-1], sol.state[0][2]])
data[musc_1.e] = np.concatenate([sol.control[0][0][:-1], sol.control[0][1]])
data[musc_2.e] = np.concatenate([sol.control[0][1][:-1], sol.control[0][0]])

inputs = [x, v, musc_1.a, musc_2.a, musc_1.e, musc_2.e]
outputs = [
    musc_1._F_T_tilde.doit().subs(kdes).xreplace(problem.auxiliary_data),
    musc_2._F_T_tilde.doit().subs(kdes).xreplace(problem.auxiliary_data),
    musc_1._l_M_tilde.doit().subs(kdes).xreplace(problem.auxiliary_data),
    musc_2._l_M_tilde.doit().subs(kdes).xreplace(problem.auxiliary_data),
]
eval_other = sm.lambdify(inputs, outputs, cse=True)
musc_1_F_T_tilde, musc_2_F_T_tilde, musc_1_l_M_tilde, musc_2_l_M_tilde = eval_other(
    data[x],
    data[v],
    data[musc_1.a],
    data[musc_2.a],
    data[musc_1.e],
    data[musc_2.e],
)

plt.figure()
plt.plot(data[t], data[x], color="tab:blue", label="Block Position")
plt.plot(data[t], data[v], color="tab:olive", label="Block Velocity")
plt.title("Dynamics States")
plt.legend()

plt.figure()
plt.plot(data[t], musc_1_F_T_tilde, color="tab:brown", label="Muscle 1")
plt.plot(data[t], musc_2_F_T_tilde, color="tab:grey", label="Muscle 2")
plt.title("Normalised Tendon Forces")
plt.legend()

plt.figure()
plt.plot(data[t], musc_1_l_M_tilde, color="tab:red", label="Muscle 1")
plt.plot(data[t], musc_2_l_M_tilde, color="tab:pink", label="Muscle 2")
plt.title("Normalised Fibre Lengths")
plt.legend()

plt.figure()
plt.plot(data[t], data[musc_1.a], color="tab:green", label="Muscle 1")
plt.plot(data[t], data[musc_2.a], color="tab:cyan", label="Muscle 2")
plt.title("Muscle Activations")
plt.legend()

plt.figure()
plt.plot(data[t], data[musc_1.e], color="tab:purple", label="Muscle 1")
plt.plot(data[t], data[musc_2.e], color="tab:pink", label="Muscle 2")
plt.title("Muscle Excitations")
plt.legend()

plt.show()
