import matplotlib.pyplot as plt
import sympy as sm
import sympy.physics.mechanics as me
from sympy.physics.mechanics._pathway import PathwayBase


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

        Aw.set_pos(self.P, -self.r*sm.cos(self.alpha)*self.A.x +
                   self.r*sm.sin(self.alpha)*self.A.y)

        Bw.set_pos(self.P, -self.r*sm.cos(self.beta)*self.B.x -
                   self.r*sm.sin(self.beta)*self.B.y)
        force_on_Aw = force_magnitude*self.Am.pos_from(Aw).normalize()
        force_on_Bw = force_magnitude*self.Bm.pos_from(Bw).normalize()
        loads = [
            me.Force(self.Am, force_on_Aw),
            me.Force(self.P, -(force_on_Aw + force_on_Bw)),
            me.Force(self.Bm, force_on_Bw),
        ]
        return loads


def plot_config(x, y, z):

    # create a figure
    fig = plt.figure()
    fig.set_size_inches((10.0, 10.0))

    # setup the subplots
    ax_top = fig.add_subplot(2, 2, 1)
    ax_3d = fig.add_subplot(2, 2, 2, projection='3d')
    ax_front = fig.add_subplot(2, 2, 3)
    ax_right = fig.add_subplot(2, 2, 4)

    # common line and marker properties for each panel
    line_prop = {
        'color': 'black',
        'marker': 'o',
        'markerfacecolor': 'blue',
        'markersize': 10,
    }

    # top view
    lines_top, = ax_top.plot(x, z, **line_prop)
    ax_top.set_xlim((-1.0, 1.0))
    ax_top.set_ylim((1.0, -1.0))
    ax_top.set_title('Top View')
    ax_top.set_xlabel('x')
    ax_top.set_ylabel('z')
    ax_top.set_aspect('equal')

    # 3d view
    lines_3d, = ax_3d.plot(x, z, y, **line_prop)
    ax_3d.set_xlim((-1.0, 1.0))
    ax_3d.set_ylim((1.0, -1.0))
    ax_3d.set_zlim((-1.0, 1.0))
    ax_3d.set_xlabel('x')
    ax_3d.set_ylabel('z')
    ax_3d.set_zlabel('y')

    # front view
    lines_front, = ax_front.plot(x, y, **line_prop)
    ax_front.set_xlim((-1.0, 1.0))
    ax_front.set_ylim((-1.0, 1.0))
    ax_front.set_title('Front View')
    ax_front.set_xlabel('x')
    ax_front.set_ylabel('y')
    ax_front.set_aspect('equal')

    # right view
    lines_right, = ax_right.plot(z, y, **line_prop)
    ax_right.set_xlim((1.0, -1.0))
    ax_right.set_ylim((-1.0, 1.0))
    ax_right.set_title('Right View')
    ax_right.set_xlabel('z')
    ax_right.set_ylabel('y')
    ax_right.set_aspect('equal')

    fig.tight_layout()
