import numpy as np
import matplotlib.pyplot as plt
import sympy as sm
import sympy.physics.mechanics as me
from sympy.physics.mechanics.pathway import LinearPathway

from biomechanics import (FirstOrderActivationDeGroote2016,
                          MusculotendonDeGroote2016)
from biomechanics.plot import plot_config, plot_traj

q, u = me.dynamicsymbols('q, u')
m, g = sm.symbols('m, g')
F_M_max, l_M_opt, l_T_slack = sm.symbols('F_M_max, l_M_opt, l_T_slack')
v_M_max, alpha_opt, beta = sm.symbols('v_M_max, alpha_opt, beta')

N = me.ReferenceFrame('N')
O, P = sm.symbols('O, P', cls=me.Point)

P.set_pos(O, q*N.x)
O.set_vel(N, 0)
P.set_vel(N, u*N.x)

gravity = me.Force(P, m*g*N.x)

muscle_pathway = LinearPathway(O, P)
muscle_activation = FirstOrderActivationDeGroote2016.with_default_constants('muscle')
muscle = MusculotendonDeGroote2016(
    'muscle',
    muscle_pathway,
    activation_dynamics=muscle_activation,
    tendon_slack_length=l_T_slack,
    peak_isometric_force=F_M_max,
    optimal_fiber_length=l_M_opt,
    maximal_fiber_velocity=v_M_max,
    optimal_pennation_angle=alpha_opt,
    fiber_damping_coefficient=beta,
)

block = me.Particle('block', P, m)

kane = me.KanesMethod(N, (q,), (u,), kd_eqs=(u - q.diff(),))
kane.kanes_equations((block,), (muscle.to_loads() + [gravity]))

e = muscle.activation_dynamics.control_variables[0]
a = muscle.activation_dynamics.state_variables[0]

dqdt = u
dudt = kane.forcing[0]/m
dadt = list(muscle.activation_dynamics.state_equations.values())[0]

state = [q, u, a]
inputs = [e]
constants = [m, g, F_M_max, l_M_opt, l_T_slack, v_M_max, alpha_opt, beta]

eval_eom = sm.lambdify((state, inputs, constants), (dqdt, dudt, dadt))

p_vals = np.array([
    1.0,  # m [kg]
    9.81,  # g [m/s/s]
    10.0,  # F_M_max
    0.18,  # l_M_opt
    0.17,  # l_T_slack
    10.0,  # v_M_max
    0.0,  # alpha_opt
    0.1,  # beta
])

x_vals = np.array([
    p_vals[3] + p_vals[4],  # q [m]
    0.0,  # u [m/s]
    0.0,  # a [?]
])

r_vals = np.array([
    0.0,  # e
])

print(dudt.doit().xreplace({
    q: 0.17 + 0.18,
    u: 0.0,
    a: 0.0,
    e: 0.0,
    m: 1.0,
    g: 9.81,
    F_M_max: 500.0,
    l_M_opt: 0.18,
    l_T_slack: 0.17,
    v_M_max: 10.0,
    alpha_opt: 0.0,
    beta: 0.1,
}))
print(eval_eom(x_vals, r_vals, p_vals))


def eval_rhs(t, x):

    r = np.array([0.5*t])

    return eval_eom(x, r, p_vals)

from scipy.integrate import solve_ivp

t0, tf = 0.0, 2.0
times = np.linspace(t0, tf, num=1001)
sol = solve_ivp(eval_rhs,
                (t0, tf),
                x_vals, t_eval=times)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1)
axes[0].plot(sol.t, sol.y[0], label=state[0])
axes[1].plot(sol.t, sol.y[1], label=state[1])
axes[2].plot(sol.t, sol.y[2], label=state[2])
axes[0].legend(), axes[1].legend(), axes[2].legend()
fig.savefig('muscle-vs-gravity.png')
