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
    MusculotendonFormulation,
)

from tug_of_war_plot import TugOfWarData, plot_solution_pycollo


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
# MUSCULOTENDON_FORMULATION = MusculotendonFormulation.RIGID_TENDON
MUSCULOTENDON_FORMULATION = MusculotendonFormulation.FIBER_LENGTH_EXPLICIT

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
    musculotendon_dynamics=MUSCULOTENDON_FORMULATION,
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
    musculotendon_dynamics=MUSCULOTENDON_FORMULATION,
    tendon_slack_length=l_T_slack,
    peak_isometric_force=F_M_max,
    optimal_fiber_length=l_M_opt,
    maximal_fiber_velocity=v_M_max,
    optimal_pennation_angle=alpha_opt,
    fiber_damping_coefficient=beta,
)

problem = pycollo.OptimalControlProblem("Tug of War")
phase_A = problem.new_phase("A")

phase_A.state_variables = (x, v) + tuple(musc_1.x) + tuple(musc_2.x)
phase_A.control_variables = tuple(musc_1.r) + tuple(musc_2.r)

phase_A.state_equations = {
    x: v,
    v: (musc_2.force.doit().subs(kdes) - musc_1.force.doit().subs(kdes)) / m,
    **dict(zip(musc_1.x, musc_1.rhs())),
    **dict(zip(musc_2.x, musc_2.rhs())),
}

phase_A.integrand_functions = [
    musc_1.a**2 + musc_2.a**2,
]

phase_A.bounds.initial_time = 0
phase_A.bounds.final_time = T / 2
musculotendon_bounds_state_variables = {}
for musc in (musc_1, musc_2):
    for state_var in musc.state_vars:
        if state_var.name[0] == 'a':
            musculotendon_bounds_state_variables[state_var] = [0, 1]
        else:
            musculotendon_bounds_state_variables[state_var] = [0.01, 1.99]
phase_A.bounds.state_variables = {
    x: [-WALL_OFFSET, WALL_OFFSET],
    v: [-2, 2],
    **musculotendon_bounds_state_variables,
}
musculotendon_bounds_control_variables = {}
for musc in (musc_1, musc_2):
    for control_var in musc.input_vars:
        if control_var.name[0] == 'e':
            musculotendon_bounds_control_variables[control_var] = [0, 1]
phase_A.bounds.control_variables = {
    **musculotendon_bounds_control_variables,
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
phase_A_guess_state_variables = []
for musc in (musc_1, musc_2):
    for state_var in musc.state_vars:
        if state_var.name[0] == 'l' and musc == musc_1:
            phase_A_guess_state_variables.append([0.68, 1.28])
        elif state_var.name[0] == 'l' and musc == musc_2:
            phase_A_guess_state_variables.append([1.28, 0.68])
        else:
            phase_A_guess_state_variables.append([0, 0])
phase_A.guess.state_variables = [
    [-DISTANCE, DISTANCE],
    [0, 0],
    *phase_A_guess_state_variables,
]
phase_A_guess_control_variables = []
for musc in (musc_1, musc_2):
    for control_var in musc.input_vars:
        phase_A_guess_control_variables.append([0, 0])
phase_A.guess.control_variables = [
    *phase_A_guess_control_variables,
]
phase_A.guess.integral_variables = [0]

problem.objective_function = 2 * phase_A.integral_variables[0]

problem_endpoint_constraints = []
problem_bounds_endpoint_constraints = []
for musc_a, musc_b in ((musc_1, musc_2), (musc_2, musc_1)):
    for state_var_a, state_var_b in zip(musc_a.x, musc_b.x):
        state_var_t0 = phase_A.initial_state_variables[state_var_a]
        state_var_tF = phase_A.final_state_variables[state_var_b]
        problem_endpoint_constraints.append(state_var_t0 - state_var_tF)
        problem_bounds_endpoint_constraints.append([0, 0])

problem.endpoint_constraints = problem_endpoint_constraints
problem.bounds.endpoint_constraints = problem_bounds_endpoint_constraints

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

# Plot the solution
data = TugOfWarData(x, v, musc_1, musc_2, problem.auxiliary_data)
plot_solution_pycollo(problem.solution, data)
plt.show()
