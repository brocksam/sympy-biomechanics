"""Basic example of a bicep driven planar arm model.

Similar to the Opensim example:

https://github.com/opensim-org/opensim-core/tree/v4.0.0_beta#simple-example

"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sympy as sm
import sympy.physics.mechanics as me
from sympy.physics.mechanics._actuator import LinearSpring, LinearDamper
from sympy.physics.mechanics._pathway import LinearPathway, PathwayBase

from biomechanics import (
    FirstOrderActivationDeGroote2016,
    MusculotendonDeGroote2016,
)


class TricepPathway(PathwayBase):

    def __init__(self, A, B, Am, P, Bm, r, q):
        """

        Parameters
        ==========
        A : ReferenceFrame
            Attached to upper arm. A.y points from elbow to shoulder.
        B : ReferenceFrame
            Attached to lower arm. B.y points from hand to elbow.
        Am : Point
            Muscle insertion point on upper arm (lies on line O to P). Fixed in
            A.
        P : Point
            Elbow pin joint location, fixed in A and B.
        Bm : Point
            Muscle insertion point on lower arm (lies on line P to Q). Fixed in
            B.
        r : sympyfiable
            Radius of the elbow cylinder that the muscle wraps around.
        q : sympfiable function of time
            Elbow angle, zero when A and B align. Positive rotation about A.z.

        Notes
        =====

        Only valid for q >= 0.

        """
        self.A = A
        self.B = B
        self.Am = Am
        self.P = P
        self.Bm = Bm
        self.r = r
        self.q = q

        self.yA = P.pos_from(Am).magnitude()
        self.yB = P.pos_from(Bm).magnitude()
        self.alpha = sm.asin(self.r/self.yA)
        self.beta = sm.asin(self.r/self.yB)

        super().__init__(Am, Bm)

    @property
    def length(self):
        """Length of two fixed length line segments and a changing arc length
        of a circle."""

        arc_len = self.r*(self.alpha + self.q + self.beta)

        lmA = self.yA*sm.cos(self.alpha)
        lmB = self.yB*sm.cos(self.beta)

        return lmA + arc_len + lmB

    @property
    def extension_velocity(self):
        """Arc length of circle is the only thing that changes when the elbow
        flexes and extends."""
        return self.r*self.q.diff(me.dynamicsymbols._t)

    def compute_loads(self, force_magnitude):
        """Forces applied to Am, Bm, and P from the muscle wrapped over
        cylinder of radius r."""

        Aw = me.Point('Aw')  # fixed in A
        Bw = me.Point('Bw')  # fixed in B

        Aw.set_pos(self.P, -self.r*sm.cos(self.alpha)*A.x +
                   self.r*sm.sin(self.alpha)*A.y)

        Bw.set_pos(self.P, -self.r*sm.cos(self.beta)*B.x -
                   self.r*sm.sin(self.beta)*B.y)
        force_on_Aw = force_magnitude*self.Am.pos_from(Aw).normalize()
        force_on_Bw = force_magnitude*self.Bm.pos_from(Bw).normalize()
        loads = [
            me.Force(self.Am, force_on_Aw),
            me.Force(self.P, -(force_on_Aw + force_on_Bw)),
            me.Force(self.Bm, force_on_Bw),
        ]
        return loads


# q1 : shoulder angle
# q2 : elbow angle
# q1' = u1, q2' = u2
q1, q2 = me.dynamicsymbols('q1, q2')
u1, u2 = me.dynamicsymbols('u1, u2')


# lA : length of humerus (upper arm)
# lB : length of radius (lower arm)
# mA : mass of upper arm
# mB : mass of lower arm
# g : acceleration due to gravity
# iAz : central moment of inertia of upper arm
# iBz : central moment of inertia of lower arm
# k : linear spring coefficient
# c : linear damper coefficient
lA, lB, mA, mB, g, iAz, iBz = sm.symbols('lA, lB, mA, mB, g, iAz, iBz')
k, c, r = sm.symbols('k, c, r')

# pack things up
q = sm.Matrix([q1, q2])
u = sm.Matrix([u1, u2])
ud = u.diff(me.dynamicsymbols._t)
ud_zerod = {udi: 0 for udi in ud}
p = sm.Matrix([lA, lB, mA, mB, g, iAz, iBz, k, c, r])

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

A.orient_axis(N, q1, N.z)
B.orient_axis(A, q2, A.z)

A.set_ang_vel(N, u1*N.z)
B.set_ang_vel(A, u2*A.z)

Am.set_pos(O, -lA/10*A.y)
Ao.set_pos(O, -lA/2*A.y)
P.set_pos(O, -lA*A.y)
Bm.set_pos(P, -lB/10*B.y)
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
    bicep._F_M_max: 200,
    bicep._l_M_opt: 0.6,
    bicep._l_T_slack: 0.55,
    bicep._v_M_max: 10.0,
    bicep._alpha_opt: 0,
    bicep._beta: 0.1,
}

tricep_pathway = TricepPathway(A, B, Am, P, Bm, r, q2)
tricep_activation = FirstOrderActivationDeGroote2016.with_default_constants('tricep')
tricep = MusculotendonDeGroote2016('tricep', tricep_pathway, activation_dynamics=tricep_activation)
tricep_constants = {
    tricep._F_M_max: 150,
    tricep._l_M_opt: 0.6,
    tricep._l_T_slack: 0.95,
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
    (q1, q2),
    (u1, u2),
    kd_eqs=(u1 - q1.diff(), u2 - q2.diff()),
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
    e_bicep = 0.3 if t < 0.5 else 1.0
    e_tricep = 0.1 if t < 2.0 else 0.5
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
    q = x[:2]
    qd = x[2:4]
    a = x[4:]

    # evaluate the equations of motion matrices with the values of q, u, p, mt
    Md, gd = eval_Mdgd(q, qd, a, p, mt)

    # evaluate the activation dynamics with the values of a, e
    e = eval_excitation(t)
    da = eval_da(a, e)

    # solve for u'
    ud = np.linalg.solve(-Md, np.squeeze(gd))

    # pack dq/dt and du/dt into a new state time derivative vector dx/dt
    xd = np.empty_like(x)
    xd[:2] = qd
    xd[2:4] = ud
    xd[4:] = da

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

    axes[0].plot(ts, np.rad2deg(xs[:, :2]))
    axes[1].plot(ts, xs[:, 2:4])
    axes[2].plot(ts, xs[:, 4:])

    axes[0].legend([me.vlatex(q[0], mode='inline')])
    axes[1].legend([me.vlatex(q[1], mode='inline')])

    axes[0].set_ylabel('Angle [deg]')
    axes[1].set_ylabel('Angular Rate [deg/s]')
    axes[2].set_ylabel('Activation [.]')

    axes[-1].set_xlabel('Time [s]')

    fig.tight_layout()

    return axes


q_vals = np.array([
    np.deg2rad(0.0),  # q1, rad
    np.deg2rad(4.0),  # q2, rad
])

u_vals = np.array([
    0.0,  # u1, rad/s
    0.0,  # u2, rad/s
])

a_vals = np.array([
    0.5,  # a_bicep, nondimensional
    0.5,  # a_tricep, nondimensional
])

#p = sm.Matrix([lA, lB, mA, mB, g, iAz, iBz, k, c, r])
p_vals = np.array([
    30.0,  # lA, m
    30.0,  # lB, m
    2.0,  # mA, kg
    1.0,  # mB, kg
    9.81,  # g, m/s**2
    2.0/12.0*30.0**2,  # iAz, kg*m**2
    1.0/12.0*30.0**2,  # iAz, kg*m**2
    20.0,  # k, N/m
    0.1,  # c, Nms
    2.0,  # r, m
])

#mt = sm.Matrix([F_M_max_bicep, l_M_opt_bicep, l_T_slack_bicep,
#                v_M_max_bicep, alpha_opt_bicep, beta_bicep,
#                F_M_max_tricep, l_M_opt_tricep, l_T_slack_tricep,
#                v_M_max_tricep, alpha_opt_tricep, beta_tricep])
mt_vals = np.array(list(musculotendon_constants.values()))

t0, tf, fps = 0.0, 3.0, 30
ts = np.linspace(t0, tf, num=int(fps*(tf - t0)))
x0 = np.hstack((q_vals, u_vals, a_vals))

result = solve_ivp(eval_rhs, (t0, tf), x0, args=(p_vals, mt_vals), t_eval=ts)
plot_results(result.t, np.transpose(result.y))

plt.show()
