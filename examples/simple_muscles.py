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
muscle_activation = FirstOrderActivationDeGroote2016('big_dog').with_default_constants('muscle')
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

a = muscle.activation_dynamics.state_variables[0]
e = muscle.activation_dynamics.control_variables[0]

dqdt = u
dudt = kane.forcing[0]/m
dadt = list(muscle.activation_dynamics.state_equations.values())[0]

state = [q, u, a]
inputs = [e]
constants = [m, g, F_M_max, l_M_opt, l_T_slack, v_M_max, alpha_opt, beta]

eval_eom = sm.lambdify((state, inputs, constants), (dqdt, dudt, dadt))

x_vals = np.array([
    0.0,
    0.0,
    0.0,
])

r_vals = np.array([
    0.0,
])

p_vals = np.array([
    1.0,  # m [kg]
    9.81,  # g
    500.0,
    0.18,
    0.17,
    10.0,
    0.0,
    0.1,
])

print(eval_eom(x_vals, r_vals, p_vals))

from scipy.integrate import solve_ivp

sol = solve_ivp(lambda t, x: eval_eom(x, r_vals, p_vals), (0.0, 5.0), x_vals)

import matplotlib.pyplot as plt

plt.plot(sol.t, sol.y.T)
plt.show()
