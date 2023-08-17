"""Basic example of a simple planar arm driven by a bicep and tricep.

Similar to the Opensim example:

https://github.com/opensim-org/opensim-core/tree/v4.0.0_beta#simple-example

"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sympy as sm
import sympy.physics.mechanics as me
from sympy.physics.mechanics.actuator import LinearSpring, LinearDamper
from sympy.physics.mechanics.pathway import LinearPathway, PathwayBase

from biomechanics import (
    ExtensorPathway,
    FirstOrderActivationDeGroote2016,
    MusculotendonDeGroote2016,
)


# q : elbow angle
# q' = u
q1 = me.dynamicsymbols('q1')
u1 = me.dynamicsymbols('u1')


# lA : length of humerus (upper arm)
# lB : length of radius (lower arm)
# mA : mass of upper arm
# mB : mass of lower arm
# g : acceleration due to gravity
# iAz : central moment of inertia of upper arm
# iBz : central moment of inertia of lower arm
lA, lB, mA, mB, g, iAz, iBz = sm.symbols('lA, lB, mA, mB, g, iAz, iBz')
r = sm.symbols('r')

# pack things up
q = sm.Matrix([q1])
u = sm.Matrix([u1])
ud = u.diff(me.dynamicsymbols._t)
ud_zerod = {udi: 0 for udi in ud}
p = sm.Matrix([lA, lB, mA, mB, g, iAz, iBz, r])

# N : inertial
# A : humerous
# B : radius
N, A, B = sm.symbols('N, A, B', cls=me.ReferenceFrame)
# O : shoulder
# Am : muscle attachment on humerous
# Ao : upper arm mass center
# P : elbow
# Bm : muscle attachment on radius
# Bo : lower arm mass center
# Q : hand
O, Am, Ao, P, Bm, Bo, Q = sm.symbols('O, Am, Ao, P, Bm, Bo, Q', cls=me.Point)

A.orient_axis(N, 0, N.z)
B.orient_axis(A, q1, A.z)

A.set_ang_vel(N, 0)
B.set_ang_vel(A, u1*A.z)

Am.set_pos(O, -2*lA/10*A.y)
Ao.set_pos(O, -lA/2*A.y)
P.set_pos(O, -lA*A.y)
Bm.set_pos(P, -3*lB/10*B.y)
Bo.set_pos(P, -lB/2*B.y)
Q.set_pos(P, -lB*B.y)

O.set_vel(N, 0)
Am.v2pt_theory(O, N, A)
Ao.v2pt_theory(O, N, A)
P.v2pt_theory(O, N, A)
Bm.v2pt_theory(P, N, B)
Bo.v2pt_theory(P, N, B)
Q.v2pt_theory(P, N, B)

IA = me.inertia(A, 0, 0, iAz)
IB = me.inertia(A, 0, 0, iBz)

humerous = me.RigidBody('humerus', masscenter=Ao, frame=A, mass=mA,
                        inertia=(IA, Ao))
radius = me.RigidBody('radius', masscenter=Bo, frame=B, mass=mB,
                      inertia=(IB, Bo))

# TODO : should be able to sum actuators that have the same pathway
# TODO : no easy way to set generalized speeds
bicep_pathway = LinearPathway(Am, Bm)
bicep_activation = FirstOrderActivationDeGroote2016.with_default_constants('bicep')
bicep = MusculotendonDeGroote2016('bicep', bicep_pathway, activation_dynamics=bicep_activation)
bicep_constants = {
    bicep._F_M_max: 200.0,
    bicep._l_M_opt: 0.6,
    bicep._l_T_slack: 0.55,
    bicep._v_M_max: 10.0,
    bicep._alpha_opt: 0,
    bicep._beta: 0.1,
}

tricep_pathway = ExtensorPathway(A.z, P, A.y, -B.y, Am, Bm, r, q1)
tricep_activation = FirstOrderActivationDeGroote2016.with_default_constants('tricep')
tricep = MusculotendonDeGroote2016('tricep', tricep_pathway, activation_dynamics=tricep_activation)
tricep_constants = {
    tricep._F_M_max: 150.0,
    tricep._l_M_opt: 0.6,
    tricep._l_T_slack: 0.65,
    tricep._v_M_max: 10.0,
    tricep._alpha_opt: 0,
    tricep._beta: 0.1,
}
musculotendon_constants = {**bicep_constants, **tricep_constants}
mt = sm.Matrix(list(musculotendon_constants.keys()))

a = list(bicep.activation_dynamics.state_variables) + list(tricep.activation_dynamics.state_variables)
e = list(bicep.activation_dynamics.control_variables) + list(tricep.activation_dynamics.control_variables)
da = list(bicep.activation_dynamics.state_equations.values()) + list(tricep.activation_dynamics.state_equations.values())
eval_da = sm.lambdify((a, e), da, cse=True)

gravA = me.Force(humerous, -mA*g*N.y)
gravB = me.Force(radius, -mB*g*N.y)

loads = (
    bicep.to_loads() +
    tricep.to_loads() +
    [gravA, gravB]
)

kane = me.KanesMethod(
    N,
    (q1, ),
    (u1, ),
    kd_eqs=(u1 - q1.diff(), ),
    bodies=(humerous, radius),
    forcelist=loads,
)

Fr, Frs = kane.kanes_equations()

Md = Frs.jacobian(ud)
gd = Frs.xreplace(ud_zerod) + Fr
eval_Mdgd = sm.lambdify((q, u, a, p, mt), [Md, gd], cse=True)


def eval_excitation(t):
    """Return the excitation of the bicep and tricep at a given time.

    Bicep and tricep excitations are step functions. The bicep excites at a
    level of 0.3 until time t=0.5s, then increases to 1.0. The tricep excites
    at a level of 0.1 until time t=2.0s, then increases to 0.5.

    Parameters
    ==========
    t : float
        Time in seconds

    Returns
    =======
    e : ndarray, shape(2,)
        Excitation of the bicep and tricep at time t.

    """
    if t < 2.0:
        e_bicep = 0.3
        e_tricep = 0.3
    elif t < 4.0:
        e_bicep = 1.0
        e_tricep = 0.1
    elif t < 6.0:
        e_bicep = 1.0
        e_tricep = 1.0
    elif t < 8.0:
        e_bicep = 0.1
        e_tricep = 1.0
    else:
        e_bicep = 0.5
        e_tricep = 0.5
    e = np.array([e_bicep, e_tricep])
    return e


def eval_rhs(t, x, p, mt):
    """Return the right hand side of the explicit ordinary differential
    equations which evaluates the time derivative of the state ``x`` at time
    ``t``.

    Parameters
    ==========
    t : float
       Time in seconds.
    x : array_like, shape(6,)
       State at time t: [q1, q2, u1, u2, a_bicep, a_tricep]
    p : array_like, shape(10,)
       Constant parameters: [lA, lB, mA, mB, g, iAz, iBz, k, c, r]
    mt : array_like, shape(12,)
        Musculotendon constant parameters: [F_M_max_bicep, l_M_opt_bicep,
        l_T_slack_bicep, v_M_max_bicep, alpha_opt_bicep, beta_bicep,
        F_M_max_tricep, l_M_opt_tricep, l_T_slack_tricep, v_M_max_tricep,
        alpha_opt_tricep, beta_tricep]

    Returns
    =======
    xd : ndarray, shape(6,)
        Derivative of the state with respect to time at time ``t``.

    """

    # unpack the q, u, and a vectors from x
    q = x[:1]
    qd = x[1:2]
    a = x[2:]

    # evaluate the equations of motion matrices with the values of q, u, p, mt
    Md, gd = eval_Mdgd(q, qd, a, p, mt)

    # evaluate the activation dynamics with the values of a, e
    e = eval_excitation(t)
    da = eval_da(a, e)

    # solve for u'
    # ud = np.linalg.solve(-Md, np.squeeze(gd))
    ud = gd[0][0] / Md[0][0]

    # pack dq/dt and du/dt into a new state time derivative vector dx/dt
    xd = np.empty_like(x)
    xd[:1] = qd
    xd[1:2] = ud
    xd[2:] = da

    return xd


def plot_results(ts, xs):
    """Returns the array of axes of a 4 panel plot of the state trajectory
    versus time.

    Parameters
    ==========
    ts : array_like, shape(m,)
       Values of time.
    xs : array_like, shape(m, 6)
       Values of the state trajectories corresponding to ``ts`` in order
       [q1, q2, u1, u2, a_bicep, a_tricep].

    Returns
    =======
    axes : ndarray, shape(4,)
       Matplotlib axes for each panel.

    """

    fig, axes = plt.subplots(3, 1, sharex=True)

    fig.set_size_inches((10.0, 6.0))

    axes[0].plot(ts, np.rad2deg(xs[:, :1]))
    axes[1].plot(ts, xs[:, 1:2])
    axes[2].plot(ts, xs[:, 2:])

    axes[0].legend([me.vlatex(q[0], mode='inline')])
    axes[1].legend([me.vlatex(u[0], mode='inline')])
    axes[2].legend([me.vlatex(a[0], mode='inline'), me.vlatex(a[1], mode='inline')])

    axes[0].set_ylabel('Angle [deg]')
    axes[1].set_ylabel('Angular Rate [deg/s]')
    axes[2].set_ylabel('Activation [.]')

    axes[-1].set_xlabel('Time [s]')

    fig.tight_layout()

    return axes


q_vals = np.array([
    np.deg2rad(90.0),  # q2, rad
])

u_vals = np.array([
    0.0,  # u1, rad/s
])

a_vals = np.array([
    0.1,  # a_bicep, nondimensional
    0.1,  # a_tricep, nondimensional
])

#p = sm.Matrix([lA, lB, mA, mB, g, iAz, iBz, k, c, r])
p_vals = np.array([
    1.0,  # lA, m
    1.0,  # lB, m
    1.0,  # mA, kg
    1.0,  # mB, kg
    9.81,  # g, m/s**2
    1.0/12.0*1.0**2,  # iAz, kg*m**2
    1.0/12.0*1.0**2,  # iAz, kg*m**2
    0.1,  # r, m
])

#mt = sm.Matrix([F_M_max_bicep, l_M_opt_bicep, l_T_slack_bicep,
#                v_M_max_bicep, alpha_opt_bicep, beta_bicep,
#                F_M_max_tricep, l_M_opt_tricep, l_T_slack_tricep,
#                v_M_max_tricep, alpha_opt_tricep, beta_tricep])
mt_vals = np.array(list(musculotendon_constants.values()))

t0, tf, fps = 0.0, 10.0, 30
ts = np.linspace(t0, tf, num=int(fps*(tf - t0)))
x0 = np.hstack((q_vals, u_vals, a_vals))

result = solve_ivp(eval_rhs, (t0, tf), x0, args=(p_vals, mt_vals), t_eval=ts)
plot_results(result.t, np.transpose(result.y))

plt.show()
