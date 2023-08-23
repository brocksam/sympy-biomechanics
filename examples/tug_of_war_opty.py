"""Solve the tug of war OCP using opty."""

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty.direct_collocation import Problem
from sympy.physics._biomechanics import (
    FirstOrderActivationDeGroote2016,
    MusculotendonDeGroote2016,
)


WALL_OFFSET = 0.35
BLOCK_SIZE = 0.05

MASS = 20
DURATION = 0.5
DISTANCE = 0.08

NUM_NODES = 500
INTERVAL_VALUE = DURATION / (NUM_NODES - 1)

OPTIMAL_FIBER_LENGTH = 0.25
MAXIMAL_FIBER_VELOCITY = 2.5
PEAK_ISOMETRIC_FORCE = 1000
TENDON_SLACK_LENGTH = 0.05
OPTIMAL_PENNATION_ANGLE = 0.0
FIBER_DAMPING_COEFFICIENT = 0.1
ACTIVATION_TIME_CONSTANT = 0.055
DEACTIVATION_TIME_CONSTANT = 0.065
SMOOTHING_RATE = 10

x, v = me.dynamicsymbols('x, v')
m, g = sm.symbols('m, g')
t, t0, tF = sm.symbols('t, t0, tF')

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


def obj(free):
    """Minimize the sum of the squares of the muscle activations."""
    musc_1_a = free[2*NUM_NODES:3*NUM_NODES]
    musc_2_a = free[3*NUM_NODES:]
    return INTERVAL_VALUE*(np.sum(musc_1_a**2) + np.sum(musc_2_a**2))


def obj_grad(free):
    grad = np.zeros_like(free)
    grad[2*NUM_NODES:3*NUM_NODES] = 2.0*INTERVAL_VALUE*free[2*NUM_NODES:3*NUM_NODES]
    grad[3*NUM_NODES:] = 2.0*INTERVAL_VALUE*free[3*NUM_NODES:]
    return grad


state_symbols = (x, v, musc_1.a, musc_2.a)
specified_symbols = (musc_1.e, musc_2.e)
eom = sm.Matrix([
    x.diff() - v,
    v.diff() - (musc_2.force.doit() - musc_1.force.doit()) / m,
    musc_1.a.diff() - musc_1.rhs()[0],
    musc_2.a.diff() - musc_2.rhs()[0],
])

instance_constraints = (
    x.replace(t, 0.0) + 0.08,
    x.replace(t, DURATION) - 0.08,
    v.replace(t, 0.0),
    v.replace(t, DURATION),
    musc_1.a.replace(t, 0.0) - musc_2.a.replace(t, DURATION),
    musc_2.a.replace(t, 0.0) - musc_1.a.replace(t, DURATION),
)

par_map = OrderedDict()
par_map[m] = MASS
par_map[t0] = 0.0
par_map[tF] = DURATION
par_map[tau_a] = ACTIVATION_TIME_CONSTANT
par_map[tau_d] = DEACTIVATION_TIME_CONSTANT
par_map[b] = SMOOTHING_RATE
par_map[l_T_slack] = TENDON_SLACK_LENGTH
par_map[F_M_max] = PEAK_ISOMETRIC_FORCE
par_map[l_M_opt] = OPTIMAL_FIBER_LENGTH
par_map[v_M_max] = MAXIMAL_FIBER_VELOCITY
par_map[alpha_opt] = OPTIMAL_PENNATION_ANGLE
par_map[beta] = FIBER_DAMPING_COEFFICIENT

bounds = {
    x: (-0.1, 0.1),
    v: (-10.0, 10.0),
    musc_1.a: (0.0, 1.0),
    musc_2.a: (0.0, 1.0),
    musc_1.e: (0.0, 1.0),
    musc_2.e: (0.0, 1.0),
}

problem = Problem(
    obj,
    obj_grad,
    eom,
    state_symbols,
    NUM_NODES,
    INTERVAL_VALUE,
    known_parameter_map=par_map,
    instance_constraints=instance_constraints,
    bounds=bounds)

# Use a random positive initial guess.
initial_guess = np.random.randn(problem.num_free)

# Find the optimal solution.
sol, info = problem.solve(initial_guess)

# Make some plots
data = {}
x_slice = slice(0, NUM_NODES)
v_slice = slice(NUM_NODES, 2*NUM_NODES)
musc_1_a_slice = slice(2*NUM_NODES, 3*NUM_NODES)
musc_2_a_slice = slice(3*NUM_NODES, 4*NUM_NODES)
musc_1_e_slice = slice(4*NUM_NODES, 5*NUM_NODES)
musc_2_e_slice = slice(5*NUM_NODES, 6*NUM_NODES)
data[t] = np.concatenate([np.linspace(0.0, DURATION, NUM_NODES), np.linspace(DURATION, 2*DURATION, NUM_NODES)])
data[x] = np.concatenate([sol[x_slice], -sol[x_slice]])
data[v] = np.concatenate([sol[v_slice], -sol[v_slice]])
data[musc_1.a] = np.concatenate([sol[musc_1_a_slice], sol[musc_2_a_slice]])
data[musc_2.a] = np.concatenate([sol[musc_2_a_slice], sol[musc_1_a_slice]])
data[musc_1.e] = np.concatenate([sol[musc_1_e_slice], sol[musc_2_e_slice]])
data[musc_2.e] = np.concatenate([sol[musc_2_e_slice], sol[musc_1_e_slice]])

kdes = {x.diff(me.dynamicsymbols._t): v}
inputs = [x, v, musc_1.a, musc_2.a, musc_1.e, musc_2.e]
outputs = [
    musc_1._F_T_tilde.doit().subs(kdes).xreplace(par_map),
    musc_2._F_T_tilde.doit().subs(kdes).xreplace(par_map),
    musc_1._l_M_tilde.doit().subs(kdes).xreplace(par_map),
    musc_2._l_M_tilde.doit().subs(kdes).xreplace(par_map),
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

plt.show()
