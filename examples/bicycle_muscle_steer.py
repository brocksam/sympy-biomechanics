import numpy as np
from scikits.odes import dae
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sm
import sympy.physics.mechanics as mec

from biomechanics.plot import plot_config


class ReferenceFrame(mec.ReferenceFrame):
    """Subclass that enforces the desired unit vector indice style."""

    def __init__(self, *args, **kwargs):

        kwargs.pop('indices', None)
        kwargs.pop('latexs', None)

        lab = args[0].lower()
        tex = r'\hat{{{}}}_{}'

        super(ReferenceFrame, self).__init__(*args, indices=('1', '2', '3'),
                                             latexs=(tex.format(lab, '1'),
                                                     tex.format(lab, '2'),
                                                     tex.format(lab, '3')),
                                             **kwargs)


##################
# Reference Frames
##################

print('Defining reference frames.')

# Newtonian Frame
N = ReferenceFrame('N')
# Yaw Frame
A = ReferenceFrame('A')
# Roll Frame
B = ReferenceFrame('B')
# Rear Frame
C = ReferenceFrame('C')
# Rear Wheel Frame
D = ReferenceFrame('D')
# Front Frame
E = ReferenceFrame('E')
# Front Wheel Frame
F = ReferenceFrame('F')
# Right Upper Arm
G = ReferenceFrame('G')
# Right Lower Arm
H = ReferenceFrame('H')
# Left Upper Arm
I = ReferenceFrame('I')
# Left Lower Arm
J = ReferenceFrame('J')

####################################
# Generalized Coordinates and Speeds
####################################

# All the following are a function of time.
t = mec.dynamicsymbols._t

# q1: perpendicular distance from the n2> axis to the rear contact
#     point in the ground plane
# q2: perpendicular distance from the n1> axis to the rear contact
#     point in the ground plane
# q3: frame yaw angle
# q4: frame roll angle
# q5: frame pitch angle
# q6: rear wheel rotation angle
# q7: steering rotation angle
# q8: front wheel rotation angle
# q9: perpendicular distance from the n2> axis to the front contact
#     point in the ground plane
# q10: perpendicular distance from the n1> axis to the front contact
#     point in the ground plane
# q11,q12: right shoulder angles
# q13: right elbow angle
# q14,q15: left shoulder angles
# q16: left elbow angle

print('Defining time varying symbols.')

q1, q2, q3, q4 = mec.dynamicsymbols('q1 q2 q3 q4')
q5, q6, q7, q8 = mec.dynamicsymbols('q5 q6 q7 q8')
q11, q12, q13 = mec.dynamicsymbols('q11, q12, q13')
q14, q15, q16 = mec.dynamicsymbols('q14, q15, q16')

u1, u2, u3, u4 = mec.dynamicsymbols('u1 u2 u3 u4')
u5, u6, u7, u8 = mec.dynamicsymbols('u5 u6 u7 u8')
u11, u12, u13 = mec.dynamicsymbols('u11, u12, u13')
u14, u15, u16 = mec.dynamicsymbols('u14, u15, u16')

#################################
# Orientation of Reference Frames
#################################

print('Orienting frames.')

# The following defines a 3-1-2 Tait-Bryan rotation with yaw (q3), roll
# (q4), pitch (q5) angles to orient the rear frame relative to the ground.
# The front frame is then rotated through the steer angle (q7) about the
# rear frame's 3 axis.

# rear frame yaw
A.orient(N, 'Axis', (q3, N['3']))
# rear frame roll
B.orient(A, 'Axis', (q4, A['1']))
# rear frame pitch
C.orient(B, 'Axis', (q5, B['2']))
# front frame steer
E.orient(C, 'Axis', (q7, C['3']))
# right upper arm
G.orient_body_fixed(C, (q11, q12, 0), '232')
# right lower arm
H.orient_axis(G, q13, G['2'])
# left upper arm
I.orient_body_fixed(C, (q14, q15, 0), '232')
# left lower arm
J.orient_axis(I, q16, I['2'])

###########
# Constants
###########

print('Defining constants.')

# geometry
# rf: radius of front wheel
# rr: radius of rear wheel
# d1: the perpendicular distance from the steer axis to the center
#     of the rear wheel (rear offset)
# d2: the distance between wheels along the steer axis
# d3: the perpendicular distance from the steer axis to the center
#     of the front wheel (fork offset)
# l1: the distance in the c1> direction from the center of the rear
#     wheel to the frame center of mass
# l2: the distance in the c3> direction from the center of the rear
#     wheel to the frame center of mass
# l3: the distance in the e1> direction from the front wheel center to
#     the center of mass of the fork
# l4: the distance in the e3> direction from the front wheel center to
#     the center of mass of the fork
# d4, d5, d6: locates right shoulder from rear wheel center
# d7 : length of upper arm
# d8 : length of lower arm
# d9, d10, d11 : locates right handgrip from front wheel center
rf, rr = sm.symbols('rf, rr')
d1, d2, d3, d4, d5, d6 = sm.symbols('d1, d2, d3, d4, d5, d6')
d7, d8, d9, d10, d11 = sm.symbols('d7, d8, d9, d10, d11')
l1, l2, l3, l4 = sm.symbols('l1, l2, l3, l4')

# acceleration due to gravity
g = sm.symbols('g')

# mass
mc, md, me, mf, mg, mh, mi, mj = sm.symbols('mc, md, me, mf, mg, mh, mi, mj')

# inertia components
ic11, ic22, ic33, ic31 = sm.symbols('ic11, ic22, ic33, ic31')
id11, id22 = sm.symbols('id11, id22')
ie11, ie22, ie33, ie31 = sm.symbols('ie11, ie22, ie33, ie31')
if11, if22 = sm.symbols('if11, if22')

###########
# Specified
###########

# control torques
# T4 : roll torque
# T6 : rear wheel torque
# T7 : steer torque
T4, T6, T7 = mec.dynamicsymbols('T4 T6 T7')

##################
# Position Vectors
##################

print('Defining position vectors.')

# rear wheel contact point
dn = mec.Point('dn')

# newtonian origin to rear wheel center
do = mec.Point('do')
do.set_pos(dn, -rr*B['3'])

# rear wheel center to bicycle frame center
co = mec.Point('co')
co.set_pos(do, l1*C['1'] + l2*C['3'])

# rear wheel center to steer axis point
ce = mec.Point('ce')
ce.set_pos(do, d1*C['1'])

## right arm
# rear wheel center to right shoulder
cgr = mec.Point('cgr')
cgr.set_pos(do, d4*C['1'] + d5*C['2'] + d6*C['3'])

# right shoulder to elbow
gh = mec.Point('gh')
gh.set_pos(cgr, d7*G['3'])

# right shoulder to upper arm mass center
go = mec.Point('go')
go.set_pos(cgr, d7/2*G['3'])

# right shoulder to upper arm muscle attachment
gm = mec.Point('gm')
gm.set_pos(cgr, 2*d7/3*G['3'])

# right elbow to lower arm muscle atachment
hm = mec.Point('hm')
hm.set_pos(gh, d8/3*H['3'])

# right elbow to lower arm mass center
ho = mec.Point('ho')
ho.set_pos(gh, d8/2*H['3'])

# elbow to hand
hc = mec.Point('hc')
hc.set_pos(gh, d8*H['3'])

## left arm
# rear wheel center to left shoulder
cgl = mec.Point('cgl')
cgl.set_pos(do, d4*C['1'] - d5*C['2'] + d6*C['3'])

# left shoulder to elbow
ji = mec.Point('ji')
ji.set_pos(cgl, d7*I['3'])

# left shoulder to upper arm mass center
io = mec.Point('io')
io.set_pos(cgl, d7/2*I['3'])

# left shoulder to upper arm muscle attachment
im = mec.Point('im')
im.set_pos(cgl, 2*d7/3*I['3'])

# left elbow to lower arm muscle atachment
jm = mec.Point('jm')
jm.set_pos(ji, d8/3*J['3'])

# left elbow to lower arm mass center
jo = mec.Point('jo')
jo.set_pos(ji, d8/2*J['3'])

# elbow to hand
jc = mec.Point('jc')
jc.set_pos(ji, d8*J['3'])

# steer axis point to the front wheel center
fo = mec.Point('fo')
fo.set_pos(ce, d2*E['3'] + d3*E['1'])

# front wheel center to right handgrip
ch_r = mec.Point('chr')
ch_r.set_pos(ce, d9*E['1'] + d10*E['2'] + d11*E['3'])

# front wheel center to left handgrip
ch_l = mec.Point('chl')
ch_l.set_pos(ce, d9*E['1'] - d10*E['2'] + d11*E['3'])

# front wheel center to front frame center
eo = mec.Point('eo')
eo.set_pos(fo, l3*E['1'] + l4*E['3'])

# front wheel contact point
fn = mec.Point('fn')
fn.set_pos(fo, rf*E['2'].cross(A['3']).cross(E['2']).normalize())

######################
# Holonomic Constraint
######################

print('Defining holonomic constraints.')

# this constraint is enforced so that the front wheel contacts the ground
holonomic_wheel = sm.Matrix([fn.pos_from(dn).dot(A['3'])])
holonomic_handr = (hc.pos_from(co) - ch_r.pos_from(co)).to_matrix(C)
holonomic_handl = (jc.pos_from(co) - ch_l.pos_from(co)).to_matrix(C)
holonomic = holonomic_wheel.col_join(holonomic_handr).col_join(holonomic_handl)

print('The holonomic constraint is a function of these dynamic variables:')
print(list(sm.ordered(mec.find_dynamicsymbols(holonomic))))

####################################
# Kinematical Differential Equations
####################################

print('Defining kinematical differential equations.')

kinematical = [
    q3.diff(t) - u3,  # yaw
    q4.diff(t) - u4,  # roll
    q5.diff(t) - u5,  # pitch
    q7.diff(t) - u7,  # steer
    q11.diff(t) - u11,  # right shoulder extension
    q12.diff(t) - u12,  # right shoulder rotation
    q13.diff(t) - u13,  # right elbow extension
    q14.diff(t) - u14,  # left shoulder extension
    q15.diff(t) - u15,  # left shoulder rotation
    q16.diff(t) - u16,  # left elbow extension
]

####################
# Angular Velocities
####################

print('Defining angular velocities.')

# Note that the wheel angular velocities are defined relative to the frame
# they are attached to.

A.set_ang_vel(N, u3*N['3'])  # yaw rate
B.set_ang_vel(A, u4*A['1'])  # roll rate
C.set_ang_vel(B, u5*B['2'])  # pitch rate
D.set_ang_vel(C, u6*C['2'])  # rear wheel rate
E.set_ang_vel(C, u7*C['3'])  # steer rate
F.set_ang_vel(E, u8*E['2'])  # front wheel rate
G.set_ang_vel(C, u11*C['2'] + u12*G['3'])
H.set_ang_vel(G, u13*G['2'])
I.set_ang_vel(C, u14*C['2'] + u15*I['3'])
J.set_ang_vel(I, u16*I['2'])

###################
# Linear Velocities
###################

print('Defining linear velocities.')

# rear wheel contact stays in ground plane and does not slip
# TODO : Investigate setting to sm.S(0) and 0.
dn.set_vel(N, 0.0*N['1'])

# mass centers
do.v2pt_theory(dn, N, D)
co.v2pt_theory(do, N, C)
ce.v2pt_theory(do, N, C)
fo.v2pt_theory(ce, N, E)
eo.v2pt_theory(fo, N, E)
go.v2pt_theory(cgr, N, G)
ho.v2pt_theory(gh, N, H)
io.v2pt_theory(cgl, N, I)
jo.v2pt_theory(ji, N, J)

# arm & handlebar joints
cgr.v2pt_theory(co, N, C)
gh.v2pt_theory(cgr, N, G)
hc.v2pt_theory(gh, N, H)
ch_r.v2pt_theory(fo, N, F)

cgl.v2pt_theory(co, N, C)
ji.v2pt_theory(cgl, N, I)
jc.v2pt_theory(ji, N, J)
ch_l.v2pt_theory(fo, N, F)

# front wheel contact velocity
fn.v2pt_theory(fo, N, F)

####################
# Motion Constraints
####################

print('Defining nonholonomic constraints.')

nonholonomic = [
    fn.vel(N).dot(A['1']),
    fn.vel(N).dot(A['3']),
    fn.vel(N).dot(A['2']),
    holonomic_handr[0].diff(t),
    holonomic_handr[1].diff(t),
    holonomic_handr[2].diff(t),
    holonomic_handl[0].diff(t),
    holonomic_handl[1].diff(t),
    holonomic_handl[2].diff(t),
]

print('The nonholonomic constraints are a function of these dynamic variables:')
print(list(sm.ordered(mec.find_dynamicsymbols(sm.Matrix(nonholonomic)))))

#########
# Inertia
#########

print('Defining inertia.')

# NOTE : You cannot define the wheel inertias with respect to their
# respective frames because the generalized inertia force calcs will fail
# because there is no direction cosine matrix relating the wheel frames
# back to the other reference frames so I define them here with respect to
# the rear and front frames.

# NOTE : Changing 0.0 to 0 or sm.S(0) changes the floating point errors.

Ic = mec.inertia(C, ic11, ic22, ic33, 0.0, 0.0, ic31)
Id = mec.inertia(C, id11, id22, id11, 0.0, 0.0, 0.0)
Ie = mec.inertia(E, ie11, ie22, ie33, 0.0, 0.0, ie31)
If = mec.inertia(E, if11, if22, if11, 0.0, 0.0, 0.0)
Ig = mec.inertia(G, mg/12*d7**2, mg/12*d7**2, mg/2*(d7/10)**2)
Ih = mec.inertia(H, mh/12*d8**2, mh/12*d8**2, mh/2*(d8/10)**2)
Ii = mec.inertia(I, mi/12*d7**2, mi/12*d7**2, mi/2*(d7/10)**2)
Ij = mec.inertia(J, mj/12*d8**2, mj/12*d8**2, mj/2*(d8/10)**2)

##############
# Rigid Bodies
##############

print('Defining the rigid bodies.')

rear_frame = mec.RigidBody('Rear Frame', co, C, mc, (Ic, co))
rear_wheel = mec.RigidBody('Rear Wheel', do, D, md, (Id, do))
front_frame = mec.RigidBody('Front Frame', eo, E, me, (Ie, eo))
front_wheel = mec.RigidBody('Front Wheel', fo, F, mf, (If, fo))
rupper_arm = mec.RigidBody('Right Upper Arm', go, G, mg, (Ig, go))
rlower_arm = mec.RigidBody('Right Lower Arm', ho, H, mh, (Ih, ho))
lupper_arm = mec.RigidBody('Left Upper Arm', io, I, mi, (Ii, io))
llower_arm = mec.RigidBody('Left Lower Arm', jo, J, mj, (Ij, jo))

bodies = [rear_frame, rear_wheel, front_frame, front_wheel, rupper_arm,
          rlower_arm, lupper_arm, llower_arm]

###########################
# Generalized Active Forces
###########################

print('Defining the forces and torques.')

# gravity
Fco = (co, mc*g*A['3'])
Fdo = (do, md*g*A['3'])
Feo = (eo, me*g*A['3'])
Ffo = (fo, mf*g*A['3'])
Fgo = (go, mg*g*A['3'])
Fho = (ho, mh*g*A['3'])
Fio = (io, mi*g*A['3'])
Fjo = (jo, mj*g*A['3'])

# input torques
Tc = (C, T4*A['1'] - T6*B['2'] - T7*C['3'])
Td = (D, T6*C['2'])
Te = (E, T7*C['3'])

forces = [Fco, Fdo, Feo, Ffo, Fgo, Fho, Fio, Fjo, Tc, Td, Te]

# Manually compute the ground contact velocities.
kindiffdict = sm.solve(kinematical, [q3.diff(t), q4.diff(t), q5.diff(t),
                                     q7.diff(t), q11.diff(t), q12.diff(t),
                                     q13.diff(t), q14.diff(t), q15.diff(t),
                                     q16.diff(t)], dict=True)[0]
u1_def = -rr*(u5 + u6)*sm.cos(q3)
u1p_def = u1_def.diff(t).xreplace(kindiffdict)
u2_def = -rr*(u5 + u6)*sm.sin(q3)
u2p_def = u2_def.diff(t).xreplace(kindiffdict)

###############################
# Prep symbolic data for output
###############################

q_ind = (q3, q4, q7)  # yaw, roll, steer
q_dep = (q5, q11, q12, q13, q14, q15, q16)  # pitch
# NOTE : I think q3 is an ignorable coordinate too.
# rear contact 1 dist, rear contact 2 dist, rear wheel angle, front wheel angle
q_ign = (q1, q2, q6, q8)
u_ind = (u4, u6, u7)  # roll rate, rear wheel rate, steer rate
u_dep = (u3, u5, u8, u11, u12, u13, u14, u15, u16)  # yaw rate, pitch rate, front wheel rate
p = (d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, g, ic11, ic22, ic31, ic33,
     id11, id22, ie11, ie22, ie31, ie33, if11, if22, l1, l2, l3, l4, mc, md,
     me, mf, mg, mh, mi, mj, rf, rr)
r = (T4, T6, T7)
u = (u3, u4, u5, u6, u7, u8, u11, u12, u13, u14, u15, u16)
q = (q3, q4, q5, q6, q7, q8, q11, q12, q13, q14, q15, q16)

points = (
    dn,  # rear contact
    do,  # rear wheel center
    cgr,  # right shoulder
    go,
    gm,
    gh,  # right elbow
    hm,
    ho,
    hc,  # right hand
    ch_r,
    ch_l,
    jc,  # left hand
    jo,
    jm,
    ji,
    im,
    io,
    cgl,
    do,  # rear wheel center
    ce,  # steer axis
    fo,  # front wheel center
    fn,
)

###############
# Kane's Method
###############

print("Generating Kane's equations.")

kane = mec.KanesMethod(
    N,
    q_ind,
    u_ind,
    kd_eqs=kinematical,
    q_dependent=q_dep,
    configuration_constraints=holonomic,
    u_dependent=u_dep,
    velocity_constraints=nonholonomic,
    constraint_solver='CRAMER',
)

Fr, Frs = kane.kanes_equations(bodies, loads=forces)

p_vals = np.array([
    0.9534570696121849,  # d1
    0.2676445084476887,  # d2
    0.03207142672761929,  # d3
    0.8,  # d4
    0.2,  # d5, shoulder half width
    -1.0,  # d6
    0.3,  # d7, upper arm length
    0.35,  # d8, lower arm length
    0.06,  # d9
    0.2,  # d10, handlebar half width
    -0.5,  # d11
    9.81,  # g
    7.178169776497895,  # ic11
    11.0,  # ic22
    3.8225535938357873,  # ic31
    4.821830223502103,  # ic33
    0.0603,  # id11
    0.12,  # id22
    0.05841337700152972,  # ie11
    0.06,  # ie22
    0.009119225261946298,  # ie31
    0.007586622998470264,  # ie33
    0.1405,  # if11
    0.28,  # if22
    0.4707271515135145,  # l1
    -0.47792881146460797,  # l2
    -0.00597083392418685,  # l3
    -0.3699518200282974,  # l4
    85.0,  # mc
    2.0,  # md
    4.0,  # me
    3.0,  # mf
    2.3,  # mg
    1.7,  # mh
    2.3,  # mi
    1.7,  # mj
    0.35,  # rf
    0.3,  # rr
])


# start with some initial guess for the configuration and choose q1 as
# independent
q_vals = np.array([
    np.deg2rad(0.0),  # 0: q3 [rad]
    np.deg2rad(0.0),  # 1: q4 [rad]
    np.deg2rad(np.pi/10.0),  # 2: q5 [rad]
    np.deg2rad(0.0),  # 3: q6 [rad]
    np.deg2rad(0.0),  # 4: q7 [rad]
    np.deg2rad(0.0),  # 5: q8 [rad]
    np.deg2rad(10.0),  # 6: q11 [rad]
    np.deg2rad(0.0),  # 7: q12 [rad]
    np.deg2rad(30.0),  # 8: q13 [rad]
    np.deg2rad(10.0),  # 9: q14 [rad]
    np.deg2rad(0.0),  # 10: q15 [rad]
    np.deg2rad(30.0),  # 11: q16 [rad]
])

eval_holonomic = sm.lambdify((q, p), holonomic, cse=True)
# x = [q5, q11, ..., q16]
knw_idxs = [0, 1, 3, 4, 5]
unk_idxs = [2, 6, 7, 8, 9, 10, 11]
print(eval_holonomic(q_vals, p_vals))
q_sol = fsolve(lambda x: eval_holonomic((
    q_vals[0],
    q_vals[1],
    x[0],
    q_vals[3],
    q_vals[4],
    q_vals[5],
    x[1],
    x[2],
    x[3],
    x[4],
    x[5],
    x[6]), p_vals).squeeze(), q_vals[unk_idxs])
# update all q_vals with constraint consistent values
q_vals[unk_idxs] = q_sol

print(np.rad2deg(q_vals))

u_vals = np.array([
    0.0,  # u3
    0.0,  # u4
    0.0,  # u5
    0.0,  # u6
    0.0,  # u7
    0.0,  # u8
    0.0,  # u11
    0.0,  # u12
    0.0,  # u13
    0.0,  # u14
    0.0,  # u15
    0.0,  # u16
])

mpl_frame = mec.ReferenceFrame('M')
mpl_frame.orient_body_fixed(N, (sm.pi/2, sm.pi, 0), 'ZXZ')
coordinates = points[0].pos_from(points[0]).to_matrix(mpl_frame)
for point in points[1:]:
    coordinates = coordinates.row_join(point.pos_from(points[0]).to_matrix(mpl_frame))
eval_point_coords = sm.lambdify((q, p), coordinates, cse=True)

plot_data = plot_config(*eval_point_coords(q_vals, p_vals))
fig, lines_top, lines_3d, lines_front, lines_right = plot_data
plt.show()

ud = sm.Matrix([u3d, u4d, u4d, u5d, u6d, u7d, u8d, u11d, u12d, u13d, u14d, u15d, u16d])
# TODO : If you use ud.diff() instead of replacing and using ud and use
# cse=True, lambdify fails (but not with cse=False), report to sympy.
eval_kane = sm.lambdify((ud, u, q, p), (Fr + Frs).xreplace(dict(zip(u.diff(), ud))), cse=True)
eval_Mdgd = sm.lambdify((u, q, p), (kane.mass_matrix, kane.forcing), cse=True)


def eval_eom(t, x, xd, residual, p):
    """Returns the residual vector of the equations of motion.

    Parameters
    ==========
    t : float
       Time at evaluation.
    x : ndarray, shape(4,)
       State vector at time t: x = [q1, q2, q3, q4, u1, u2, u3, u4].
    xd : ndarray, shape(4,)
       Time derivative of the state vector at time t:
       xd = [q1d, q2d, q3d, q4d, u1d, u2d, u3d, u4d].
    residual : ndarray, shape(4,)
       Vector to store the residuals in: residuals = [fk, fd, fh1, fh2, fh3].
    p : ndarray, shape(6,)
       Constant parameters: p = []

    """
    q = x[0:4]
    u = x[4:8]
    qd = xd[0:4]
    ud = xd[4:8]
    residual[0:4] = u - qd
    residual[4] = eval_kane(ud, u, q, p).squeeze()  # only eq for independent u
    residual[5:] = eval_holonomic(q, p).squeeze()



solver = dae('ida',
             eval_eom,
             rtol=1e-5,
             atol=1e-5,
             algebraic_vars_idx=[5, 6, 7],
             user_data=p_vals,
             old_api=False)

x0 = np.hstack((q_vals, u_vals))
ud0 = np.linalg.solve(*eval_Mdgd(u_vals, q_vals, p_vals)).squeeze()
xd0 = np.hstack((u_vals, ud0))
resid = np.empty(8)
eval_eom(0.1, x0, xd0, resid, p_vals)
print(resid)
ts = np.linspace(0.0, 1.0, num=101)
solution = solver.solve(ts, x0, xd0)

ts = solution.values.t
xs = solution.values.y


def animate(i):
    x, y, z = eval_point_coords(xs[i, :4], p_vals)
    lines_top.set_data(x, y)
    lines_3d.set_data_3d(x, y, z)
    lines_front.set_data(y, z)
    lines_right.set_data(x, z)


ani = FuncAnimation(fig, animate, len(ts))

#plt.figure()
#plt.plot(ts, xs)

plt.show()
