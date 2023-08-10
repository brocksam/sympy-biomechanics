import numpy as np
from scikits.odes import dae
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sm
import sympy.physics.mechanics as me
from sympy.physics.mechanics._pathway import LinearPathway

from biomechanics import (
    ExtensorPathway,
    FirstOrderActivationDeGroote2016,
    MusculotendonDeGroote2016,
)
from biomechanics.plot import plot_config

# q1 : steer angle
# q2 : shoulder extension
# q3 : shoulder rotation
# q4 : elbow extension
# q1' = u1, q2' = u2, q3' = u3, q4' = u4
q1, q2, q3, q4 = me.dynamicsymbols('q1, q2, q3, q4')
u1, u2, u3, u4 = me.dynamicsymbols('u1, u2, u3, u4')
u1d, u2d, u3d, u4d = me.dynamicsymbols('u1d, u2d, u3d, u4d')

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
g = sm.symbols('g')

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
                     inertia=me.Inertia(IA, O))
humerous = me.RigidBody('humerus',
                        masscenter=Co,
                        frame=C,
                        mass=mC,
                        inertia=me.Inertia(IC, Co))
radius = me.RigidBody('radius',
                      masscenter=Do,
                      frame=D,
                      mass=mD,
                      inertia=me.Inertia(ID, Do))

grav_steer = me.Force(steer, mA*g*N.z)
grav_humerous = me.Force(humerous, mC*g*N.z)
grav_radius = me.Force(radius, mD*g*N.z)

loads = (
    [grav_steer, grav_humerous, grav_radius]
)

t = me.dynamicsymbols._t

kane = me.KanesMethod(
    N,
    q_ind=(q1,),
    u_ind=(u1,),
    kd_eqs=[
        u1 - q1.diff(),
        u2 - q2.diff(),
        u3 - q3.diff(),
        u4 - q4.diff(),
    ],
    q_dependent=(q2, q3, q4),
    configuration_constraints=holonomic,
    velocity_constraints=holonomic.diff(t),
    u_dependent=(u2, u3, u4),
    bodies=(steer, humerous, radius),
    forcelist=loads,
)

Fr, Frs = kane.kanes_equations()

p_vals = np.array([
    -0.3,  # dx [m]
    0.2,  # dy [m]
    -0.3,  # dz [m]
    0.2,   # lA [m]
    0.3,  # lC [m]
    0.3,  # lD [m]
    1.0,  # mA [kg]
    2.3,  # mC [kg]
    1.7,  # mD [kg]
    9.81,  # g [m/s/s]
])

# start with some initial guess for the configuration and choose q1 as
# independent
q_vals = np.array([
    np.deg2rad(0.0),  # q1 [rad]
    np.deg2rad(0.0),  # q2 [rad]
    np.deg2rad(0.0),  # q3 [rad]
    np.deg2rad(90.0),  # q4 [rad]
])

eval_holonomic = sm.lambdify((q, p), holonomic, cse=True)
q_sol = fsolve(lambda x: eval_holonomic((q_vals[0], x[0], x[1], x[2]), p_vals).squeeze(), q_vals[1:])
# update all q_vals with constraint consistent values
q_vals[1], q_vals[2], q_vals[3] = q_sol[0], q_sol[1], q_sol[2]

print(np.rad2deg(q_vals))

u_vals = np.array([
    0.0,
    0.0,
    0.0,
    0.0,
])

mpl_frame = me.ReferenceFrame('M')
mpl_frame.orient_body_fixed(N, (sm.pi/2, sm.pi, 0), 'ZXZ')
coordinates = O.pos_from(O).to_matrix(mpl_frame)
for point in [P1, P4, Do, Dm, P3, Cm, Co, P2]:
    coordinates = coordinates.row_join(point.pos_from(O).to_matrix(mpl_frame))
eval_point_coords = sm.lambdify((q, p), coordinates, cse=True)

plot_data = plot_config(*eval_point_coords(q_vals, p_vals))
fig, lines_top, lines_3d, lines_front, lines_right = plot_data

ud = sm.Matrix([u1d, u2d, u3d, u4d])
# TODO : If you use ud.diff() instead of replacing and using ud and use
# cse=True, lambdify fails (but not with cse=False), report to sympy.
eval_kane = sm.lambdify((ud, u, q, p), (Fr + Frs).xreplace(dict(zip(u.diff(), ud))), cse=True)
t = me.dynamicsymbols._t
vel_con = holonomic.diff(t).xreplace(dict(zip(q.diff(), u)))
Mh = vel_con.jacobian([u2, u3, u4])
gh = vel_con.xreplace({u2: 0, u3: 0, u4: 0})
eval_Mhgh = sm.lambdify((u1, q, p), (Mh, gh), cse=True)
eval_Mdgd = sm.lambdify((u, q, p), (kane.mass_matrix, kane.forcing), cse=True)


def eval_eom(t, x, xd, residual, p):
    """Returns the residual vector of the equations of motion.

    Parameters
    ==========
    t : float
       Time at evaluation.
    x : ndarray, shape(10,)
       State vector at time t: x = [q1, q2, q3, q4, u1, u2, u3, u4, a_bicep, a_tricep].
    xd : ndarray, shape(10,)
       Time derivative of the state vector at time t:
       xd = [q1d, q2d, q3d, q4d, u1d, u2d, u3d, u4d, da_bicep, da_tricep].
    residual : ndarray, shape(5,)
       Vector to store the residuals in: residuals = [fk, fd, fh1, fh2, fh3].
    p : ndarray, shape(12,)
       Constant parameters: [dx, dy, dz, lA, lC, lD, mA, mC, mD, g, kA, cA, r]

    """
    q = x[0:4]
    u = x[4:8]
    qd = xd[0:4]
    ud = xd[4:8]
    residual[0:4] = u - qd
    residual[4] = eval_kane(ud, u, q, p).squeeze()  # only eq for independent u
    residual[5:8] = eval_holonomic(q, p).squeeze()



solver = dae('ida',
             eval_eom,
             rtol=1e-5,
             atol=1e-5,
             algebraic_vars_idx=[5, 6, 7],
             user_data=p_vals,
             old_api=False)

t0, tF = 0.0, 2.0
x0 = np.hstack((q_vals, u_vals))
ud0 = np.linalg.solve(*eval_Mdgd(u_vals, q_vals, p_vals)).squeeze()
xd0 = np.hstack((u_vals, ud0))
resid = np.empty(8)
eval_eom(0.1, x0, xd0, resid, p_vals)
print(resid)
ts = np.linspace(t0, tF, num=101)
solution = solver.solve(ts, x0, xd0)

ts = solution.values.t
xs = solution.values.y


def animate(i):
    x, y, z = eval_point_coords(xs[i, :4], p_vals)
    fig.suptitle(f'{ts[i]:.2f} s')
    lines_top.set_data(x, y)
    lines_3d.set_data_3d(x, y, z)
    lines_front.set_data(y, z)
    lines_right.set_data(x, z)


ani = FuncAnimation(fig, animate, len(ts), interval=10)

# plt.figure()
# plt.plot(ts, xs)

plt.show()
