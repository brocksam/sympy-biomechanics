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
from biomechanics.plot import plot_config, plot_traj

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
    r,
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
B.set_ang_vel(N, u2*N.y)
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

steer_resistance = me.Torque(A, (-kA*q1 - cA*u2)*N.z)
steer_resistance = me.Torque(A, 0*N.z)

# musculotendons
bicep_pathway = LinearPathway(Cm, Dm)
bicep_activation = FirstOrderActivationDeGroote2016.with_default_constants('bicep')
bicep = MusculotendonDeGroote2016('bicep', bicep_pathway, activation_dynamics=bicep_activation)
bicep_constants = {
    bicep._F_M_max: 500.0,
    bicep._l_M_opt: 0.6 * 0.3,
    bicep._l_T_slack: 0.55 * 0.3,
    bicep._v_M_max: 10.0,
    bicep._alpha_opt: 0.0,
    bicep._beta: 0.1,
}

tricep_pathway = ExtensorPathway(C.y, P3, -C.z, D.z, Cm, Dm, r, q4)
tricep_activation = FirstOrderActivationDeGroote2016.with_default_constants('tricep')
tricep = MusculotendonDeGroote2016('tricep', tricep_pathway, activation_dynamics=tricep_activation)
tricep_constants = {
    tricep._F_M_max: 500.0,
    tricep._l_M_opt: 0.6 * 0.3,
    tricep._l_T_slack: 0.65 * 0.3,
    tricep._v_M_max: 10.0,
    tricep._alpha_opt: 0.0,
    tricep._beta: 0.1,
}
musculotendon_constants = {**bicep_constants, **tricep_constants}
mt = sm.Matrix(list(musculotendon_constants.keys()))

a = list(bicep.activation_dynamics.state_variables) + list(tricep.activation_dynamics.state_variables)
e = list(bicep.activation_dynamics.control_variables) + list(tricep.activation_dynamics.control_variables)
da = list(bicep.activation_dynamics.state_equations.values()) + list(tricep.activation_dynamics.state_equations.values())
eval_da = sm.lambdify((e, a), da, cse=True)

gravA = me.Force(humerous, mC*g*N.z)
gravB = me.Force(radius, mD*g*N.z)

loads = (
    bicep.to_loads() +
    tricep.to_loads() +
    [steer_resistance, gravA, gravB]
)

t = me.dynamicsymbols._t

kane = me.KanesMethod(
    N,
    (q1,),
    (u1,),
    kd_eqs=(
        u1 - q1.diff(),
        u2 - q2.diff(),
        u3 - q3.diff(),
        u4 - q4.diff(),
    ),
    q_dependent=(q2, q3, q4),
    configuration_constraints=holonomic,
    velocity_constraints=holonomic.diff(t),
    u_dependent=(u2, u3, u4),
    bodies=(steer, humerous, radius),
    forcelist=loads,
)

Fr, Frs = kane.kanes_equations()

p_vals = np.array([
    -0.31,  # dx [m]
    0.15,  # dy [m]
    -0.31,  # dz [m]
    0.2,   # lA [m]
    0.3,  # lC [m]
    0.3,  # lD [m]
    1.0,  # mA [kg]
    2.3,  # mC [kg]
    1.7,  # mD [kg]
    9.81,  # g [m/s/s]
    10.0,  # kA [Nm/rad]
    0.2,  # cA [Nms/rad]
    0.03,  # r [m]
])
mt_vals = np.array(list(musculotendon_constants.values()))

# start with some initial guess for the configuration and choose q1 as
# independent
q_vals = np.array([
    np.deg2rad(2.0),  # q1 [rad]
    np.deg2rad(-10.0),  # q2 [rad]
    np.deg2rad(0.0),  # q3 [rad]
    np.deg2rad(75.0),  # q4 [rad]
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

a_vals = np.array([
    0.01,  # a_bicep, nondimensional
    0.01,  # a_tricep, nondimensional
])

mpl_frame = me.ReferenceFrame('M')
mpl_frame.orient_body_fixed(N, (sm.pi/2, sm.pi, 0), 'ZXZ')
coordinates = O.pos_from(O).to_matrix(mpl_frame)
for point in [P1, P4, Do, Dm, P3, Cm, Co, P2]:
    coordinates = coordinates.row_join(point.pos_from(O).to_matrix(mpl_frame))
eval_point_coords = sm.lambdify((q, p), coordinates, cse=True)

plot_data = plot_config(*eval_point_coords(q_vals, p_vals), xlim=(-0.2, 0.4),
                        ylim=(-0.4, 0.2), zlim=(-0.2, 0.4))
fig, lines_top, lines_3d, lines_front, lines_right = plot_data

ud = sm.Matrix([u1d, u2d, u3d, u4d])
# TODO : If you use ud.diff() instead of replacing and using ud and use
# cse=True, lambdify fails (but not with cse=False), report to sympy.
eval_kane = sm.lambdify((ud, u, q, a, p, mt), (Fr + Frs).xreplace(dict(zip(u.diff(), ud))), cse=True)
t = me.dynamicsymbols._t
vel_con = holonomic.diff(t).xreplace(dict(zip(q.diff(), u)))
Mh = vel_con.jacobian([u2, u3, u4])
gh = vel_con.xreplace({u2: 0, u3: 0, u4: 0})
eval_Mhgh = sm.lambdify((u1, q, a, p, mt), (Mh, gh), cse=True)
eval_Mdgd = sm.lambdify((u, q, a, p, mt), (kane.mass_matrix, kane.forcing), cse=True)


def eval_excitation(t):
    """Return the excitation of the bicep and tricep at a given time.

    Parameters
    ==========
    t : float
        Time in seconds

    Returns
    =======
    e : ndarray, shape(2,)
        Excitation of the bicep and tricep at time t.

    """
    e_bicep = 0.01 if t < 0.1 else 0.9
    e_tricep = 0.01 if t < 0.1 else 0.5
    e = np.array([e_bicep, e_tricep])
    return e


def eval_eom(t, x, xd, residual, constants):
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
    mt : ndarray(), shape(12,)
        Musculotendon constant parameters: [F_M_max_bicep, l_M_opt_bicep,
        l_T_slack_bicep, v_M_max_bicep, alpha_opt_bicep, beta_bicep,
        F_M_max_tricep, l_M_opt_tricep, l_T_slack_tricep, v_M_max_tricep,
        alpha_opt_tricep, beta_tricep]

    """
    p, mt = constants
    q = x[0:4]
    u = x[4:8]
    a = x[8:10]
    qd = xd[0:4]
    ud = xd[4:8]
    ad = xd[8:10]
    residual[0:4] = u - qd
    residual[4] = eval_kane(ud, u, q, a, p, mt).squeeze()  # only eq for independent u
    residual[5:8] = eval_holonomic(q, p).squeeze()
    e = eval_excitation(t)
    residual[8:10] = eval_da(e, a) - ad



solver = dae('ida',
             eval_eom,
             first_step_size=0.01,
             rtol=1e-5,
             atol=1e-5,
             algebraic_vars_idx=[5, 6, 7],
             user_data=(p_vals, mt_vals),
             old_api=False)

t0, tF = 0.0, 2.0
x0 = np.hstack((q_vals, u_vals, a_vals))
ud0 = np.linalg.solve(*eval_Mdgd(u_vals, q_vals, a_vals, p_vals, mt_vals)).squeeze()
da0 = eval_da(eval_excitation(t0), a_vals)
xd0 = np.hstack((u_vals, ud0, da0))
resid = np.empty(10)
eval_eom(0.01, x0, xd0, resid, (p_vals, mt_vals))
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

plot_traj(ts, xs, (q1, q2, q3, q4, u1, u2, u3, u4, bicep.a, tricep.a))

plt.show()
