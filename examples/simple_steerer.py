import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sympy as sm
import sympy.physics.mechanics as me
#from sympy.physics.mechanics._actuator import LinearSpring, LinearDamper
#from sympy.physics.mechanics._pathway import LinearPathway, PathwayBase

# q1 : steer angle
# q2 : shoulder extension
# q3 : shoulder rotation
# q4 : elbow extension
# q1' = u1, q2' = u2, q3' = u3, q4' = u4
q1, q2, q3, q4 = me.dynamicsymbols('q1, q2, q3, q4')
u1, u2, u3, u4 = me.dynamicsymbols('u1, u2, u3, u4')

# dx, dy, dz: locates P2 from O along the N unit vector directions
# lA : handlebar halfwidth
# lC : length of humerus (upper arm)
# lD : length of radius (lower arm)
# mA : mass of handlebar
# mC : mass of upper arm
# mD : mass of lower arm
# g : acceleration due to gravity
# kA : linear rotational spring coefficient
# cA : linear rotational damper coefficient
dx, dy, dz, lA, lC, lD = sm.symbols('dx, dy, dz, lA, lC, lD')
mA, mC, mD = sm.symbols('mA, mC, mD')
g, kA, cA, r = sm.symbols('g, kA, cA, r')

# pack things up
q = sm.Matrix([q1, q2, q3, q4])
u = sm.Matrix([u1, u2, u3, u4])
ud = u.diff(me.dynamicsymbols._t)
ud_zerod = {udi: 0 for udi in ud}
p = sm.Matrix([
    dx,
    dy,
    dz,
    lA,
    lC,
    lD,
    mA,
    mC,
    mD,
    g,
    kA,
    cA,
])

# N : inertial
# A : handlebar
# C : humerous
# D : radius
N, A, B, C, D = sm.symbols('N, A, B, C, D', cls=me.ReferenceFrame)
# O : handlebar center
# P1 : right handgrip
# P2 : shoulder
# Co : humerous mass center
# Cm : humerous muscle attachment
# P3 : elbow
# Dm : muscle attachment on radius
# Do : lower arm mass center
# P4 : hand
O, P1, P2, P3, P4 = sm.symbols('O, P1, P2, P3, P4 ', cls=me.Point)
Co, Cm, Dm, Do = sm.symbols('Co, Cm, Dm, Do', cls=me.Point)

A.orient_axis(N, q1, N.z)
B.orient_axis(N, q2, N.y)
C.orient_axis(B, q3, B.z)
D.orient_axis(C, q4, C.y)

A.set_ang_vel(N, u1*N.z)
B.set_ang_vel(N, u2*N.z)
C.set_ang_vel(B, u3*B.z)
D.set_ang_vel(C, u4*C.y)

P1.set_pos(O, lA*A.y)
P2.set_pos(O, dx*N.x + dy*N.y + dz*N.z)
Co.set_pos(P2, lC/2*C.z)
Cm.set_pos(P2, 2*lC/3*C.z)
P3.set_pos(P2, lC*C.z)
Dm.set_pos(P3, 1*lD/3*D.z)
Do.set_pos(P3, lD/2*D.z)
P4.set_pos(P3, lD*D.z)

con_vec = P4.pos_from(O) - P1.pos_from(O)
holonomic = con_vec.to_matrix(N)

O.set_vel(N, 0)
P1.v2pt_theory(O, N, A)
P2.set_vel(N, 0)
Co.v2pt_theory(P2, N, C)
Cm.v2pt_theory(P2, N, C)
P3.v2pt_theory(P2, N, C)
Dm.v2pt_theory(P3, N, D)
Do.v2pt_theory(P3, N, D)
P4.v2pt_theory(P3, N, D)

# assume thin cylinders for now
IA = me.inertia(A, mA/12*lA**2, mA/2*lA**2, mA/12*lA**2)
IC = me.inertia(C, mC/12*lC**2, mC/12*lC**2, mC/2*lC**2)
ID = me.inertia(D, mD/12*lD**2, mD/12*lD**2, mD/2*lD**2)

steer = me.RigidBody('humerus',
                     masscenter=O,
                     frame=A,
                     mass=mA,
                     inertia=(IA, O))
humerous = me.RigidBody('humerus',
                        masscenter=Co,
                        frame=C,
                        mass=mC,
                        inertia=(IC, Co))
radius = me.RigidBody('radius',
                      masscenter=Do,
                      frame=D,
                      mass=mD,
                      inertia=(ID, Do))

kane = me.KanesMethod(
    N,
    (q1,),
    (u1, u2, u3, u4),
    kd_eqs=(u1 - q1.diff(), u2 - q2.diff(), u3 - q3.diff(), u4 - q4.diff()),
    q_dependent=(q2, q3, q4),
    configuration_constraints=holonomic,
    bodies=(steer, humerous, radius),
)

Fr, Frs = kane.kanes_equations()
