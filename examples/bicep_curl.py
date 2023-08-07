"""Basic example of a bicep driven planar arm model.

Similar to the Opensim example:

https://github.com/opensim-org/opensim-core/tree/v4.0.0_beta#simple-example

"""
import sympy as sm
import sympy.physics.mechanics as me
from sympy.physics.mechanics._actuator import LinearSpring, LinearDamper
from sympy.physics.mechanics._pathway import LinearPathway

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
k, c = sm.symbols('k, c')

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

muscle_pathway = LinearPathway(Am, Bm)
# TODO : should be able to sum actuators that have the same pathway
muscle_act1 = LinearSpring(k, muscle_pathway)
# TODO : no easy way to set generalized speeds
muscle_act2 = LinearDamper(c, muscle_pathway)

gravA = me.Force(humerous, -mA*g*N.y)
gravB = me.Force(radius, -mB*g*N.y)

# TODO : should gravA and gravB have a to_loads() method?
loads = muscle_act1.to_loads() + muscle_act2.to_loads() + [gravA, gravB]

kane = me.KanesMethod(
    N,
    (q1, q2),
    (u1, u2),
    kd_eqs=(u1 - q1.diff(), u2 - q2.diff()),
    bodies=(humerous, radius),
    forcelist=loads,
)

Fr, Frs = kane.kanes_equations()
