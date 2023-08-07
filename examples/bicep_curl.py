"""Basic example of a bicep driven planar arm model.

Similar to the Opensim example:

https://github.com/opensim-org/opensim-core/tree/v4.0.0_beta#simple-example

"""
import sympy as sm
import sympy.physics.mechanics as me
from sympy.physics.mechanics._actuator import LinearSpring, LinearDamper
from sympy.physics.mechanics._pathway import LinearPathway, PathwayBase


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
        force_on_Bw = force_magnitude*self.Bm.pos_from(Bw).normalize()),
        loads = [
            me.Force(self.Am, force_on_Aw),
            me.Force(self.P, -(force_on_Aw + force_on_Bw),
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
k, c, r= sm.symbols('k, c, r')

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

tricep_path = TricepPathway(A, B, Am, P, Bm, r, q2)

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
