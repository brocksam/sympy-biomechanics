from sympy import asin, cos, sin
from sympy.physics.mechanics import Force, Point, dynamicsymbols
from sympy.physics.mechanics._pathway import PathwayBase


class ExtensorPathway(PathwayBase):

    def __init__(
        self,
        axis,
        axis_point,
        parent_axis,
        child_axis,
        origin,
        insertion,
        radius,
        coordinate,
    ):
        """A custom pathway that wraps a cylinder around a pin joint.

        This is intended to be used for extensor muscles. For example, a tricep
        wrapping around the elbow joint to extend the upper arm at the elbow.

        Parameters
        ==========
        axis : Vector
            Pin joint axis
        axis_point : Point
            Pin joint location, fixed in both the parent and child.
        parent_axis : Vector
            Axis of the parent frame (A) aligned with the parent body.
        child_axis : Vector
            Axis of the child frame (B) aligned with the child body.
        origin : Point
            Muscle origin point on the parent body. Lies on the line from the
            axis point in the direction of the parent axis. Fixed in the parent.
        insertion : Point
            Muscle insertion point on the child body. Lies on the line from the
            axis point in the direction of the child axis. Fixed in the child.
        radius : sympyfiable
            Radius of the cylinder that the muscle wraps around.
        coordinate : sympfiable function of time
            Joint angle, zero when parent and child frames align. Positive
            rotation about the pin joint axis.

        Notes
        =====

        Only valid for coordinate >= 0.

        """
        super().__init__(origin, insertion)

        self.axis = axis.normalize()
        self.axis_point = axis_point
        self.parent_axis = parent_axis.normalize()
        self.child_axis = child_axis.normalize()
        self.origin = origin
        self.insertion = insertion
        self.radius = radius
        self.coordinate = coordinate

        self.origin_distance = axis_point.pos_from(origin).magnitude()
        self.insertion_distance = axis_point.pos_from(insertion).magnitude()
        self.origin_angle = asin(self.radius / self.origin_distance)
        self.insertion_angle = asin(self.radius / self.insertion_distance)

    @property
    def length(self):
        """Length of the pathway.

        Length of two fixed length line segments and a changing arc length
        of a circle.

        """

        angle = self.origin_angle + self.coordinate + self.insertion_angle
        arc_length = self.radius * angle

        origin_segment_length = self.origin_distance * cos(self.origin_angle)
        insertion_segment_length = self.insertion_distance * cos(self.insertion_angle)

        return origin_segment_length + arc_length + insertion_segment_length

    @property
    def extension_velocity(self):
        """Extension velocity of the pathway.

        Arc length of circle is the only thing that changes when the elbow
        flexes and extends.

        """
        return self.radius * self.coordinate.diff(dynamicsymbols._t)

    def compute_loads(self, force_magnitude):
        """Loads in the correct format to be supplied to `KanesMethod`.

        Forces applied to origin, insertion, and P from the muscle wrapped over
        cylinder of radius r.

        """

        parent_tangency_point = Point('Aw')  # fixed in parent
        child_tangency_point = Point('Bw')  # fixed in child

        parent_tangency_point.set_pos(
            self.axis_point,
            -self.radius*cos(self.origin_angle)*self.parent_axis.cross(self.axis)
            + self.radius*sin(self.origin_angle)*self.parent_axis,
        )
        child_tangency_point.set_pos(
            self.axis_point,
            self.radius*cos(self.insertion_angle)*self.child_axis.cross(self.axis)
            + self.radius*sin(self.insertion_angle)*self.child_axis),

        parent_force_vector = self.origin.pos_from(parent_tangency_point)
        child_force_vector = self.insertion.pos_from(child_tangency_point)
        force_on_parent = force_magnitude * parent_force_vector.normalize()
        force_on_child = force_magnitude * child_force_vector.normalize()
        loads = [
            Force(self.origin, force_on_parent),
            Force(self.axis_point, -(force_on_parent + force_on_child)),
            Force(self.insertion, force_on_child),
        ]
        return loads
