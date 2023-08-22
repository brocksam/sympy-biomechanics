import numpy as np
import matplotlib.pyplot as plt
import sympy as sm
import sympy.physics.mechanics as me
from sympy.physics.mechanics.pathway import LinearPathway

from sympy.physics._biomechanics import (FirstOrderActivationDeGroote2016,
                                         MusculotendonDeGroote2016)

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

a = muscle.x[0]
e = muscle.r[0]

force = muscle.force.xreplace({q.diff(): u})

dqdt = u
dudt = kane.forcing[0]/m
dadt = muscle.rhs()[0]

state = [q, u, a]
inputs = [e]
constants = [m, g, F_M_max, l_M_opt, l_T_slack, v_M_max, alpha_opt, beta]

eval_eom = sm.lambdify((state, inputs, constants), (dqdt, dudt, dadt))
eval_force = sm.lambdify((state, constants), force)

# q-l_T_slack is the length of the muscle

p_vals = np.array([
    0.5,  # m [kg]
    9.81,  # g [m/s/s]
    10.0,  # F_M_max
    0.18,  # l_M_opt, length of muscle at which max force is produced
    0.17,  # l_T_slack, always fixed (rigid tendon)
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

print(eval_eom(x_vals, r_vals, p_vals))


def eval_rhs(t, x):

    #if 0.5*t > 1.0:
        #r = np.array([0.0])
    #else:
        #r = np.array([0.5*t])

    r = np.array([1.0])

    return eval_eom(x, r, p_vals)

from scipy.integrate import solve_ivp

t0, tf = 0.0, 10.0
times = np.linspace(t0, tf, num=1001)
sol = solve_ivp(eval_rhs,
                (t0, tf),
                x_vals, t_eval=times)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 1, sharex=True)
axes[0].plot(sol.t, sol.y[0] - p_vals[4], label='length of muscle')
axes[1].plot(sol.t, sol.y[1], label=state[1])
axes[2].plot(sol.t, sol.y[2], label=state[2])
axes[3].plot(sol.t, eval_force(sol.y, p_vals).T, label='force')
axes[0].legend(), axes[1].legend(), axes[2].legend(), axes[3].legend()
fig.savefig('muscle-vs-gravity.png')
