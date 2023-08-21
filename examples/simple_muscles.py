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
    500.0,  # F_M_max
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

print(dudt.doit().evalf(subs={
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
print(dudt.doit().subs({
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
eval_dudt = sm.lambdify((state, inputs, constants), dudt)
# TODO : The following give incorrect results (as compared to the prior to
# xreplace() and subs() calls.
print(eval_dudt(x_vals, r_vals, p_vals))
print(eval_eom(x_vals, r_vals, p_vals))

from scipy.integrate import solve_ivp

sol = solve_ivp(lambda t, x: eval_eom(x, r_vals, p_vals), (0.0, 1.0), x_vals,
                method='LSODA')

import matplotlib.pyplot as plt

plt.plot(sol.t, sol.y.T)
plt.show()
