import numpy as np
import sympy as sm
import sympy.physics.mechanics as mec


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


def setup_symbolics():
    """Returns the fully defined symbolics of the system ready for formulation
    of the equations of motion."""

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
    # q13: elbow angle

    print('Defining time varying symbols.')

    q1, q2, q3, q4 = mec.dynamicsymbols('q1 q2 q3 q4')
    q5, q6, q7, q8 = mec.dynamicsymbols('q5 q6 q7 q8')
    q11, q12, q13 = mec.dynamicsymbols('q11, q12, q13')

    u1, u2, u3, u4 = mec.dynamicsymbols('u1 u2 u3 u4')
    u5, u6, u7, u8 = mec.dynamicsymbols('u5 u6 u7 u8')
    u11, u12, u13 = mec.dynamicsymbols('u11, u12, u13')

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
    # upper arm
    G.orient_body_fixed(C, (q11, q12, 0), '232')
    # lower arm
    H.orient_axis(G, q13, G['2'])

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
    mc, md, me, mf, mg, mh = sm.symbols('mc, md, me, mf, mg, mh')

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

    # rear wheel center to right shoulder
    cg = mec.Point('cg')
    cg.set_pos(do, d4*C['1'] + d5*C['2'] + d6*C['3'])

    # right shoulder to elbow
    gh = mec.Point('gh')
    gh.set_pos(cg, d7*G['3'])

    # right shoulder to upper arm mass center
    go = mec.Point('go')
    go.set_pos(cg, d7/2*G['3'])

    # right shoulder to upper arm muscle attachment
    gm = mec.Point('gm')
    gm.set_pos(cg, 2*d7/3*G['3'])

    # right elbow to lower arm muscle atachment
    hm = mec.Point('hm')
    hm.set_pos(gh, d8/3*H['3'])

    # right elbow to lower arm mass center
    ho = mec.Point('ho')
    ho.set_pos(gh, d8/2*H['3'])

    # elbow to hand
    hc = mec.Point('hc')
    hc.set_pos(gh, d8*H['3'])

    # steer axis point to the front wheel center
    fo = mec.Point('fo')
    fo.set_pos(ce, d2*E['3'] + d3*E['1'])

    # front wheel center to handgrip
    ch = mec.Point('ch')
    ch.set_pos(fo, d9*E['1'] + d10*E['2'] + d11*E['3'])

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
    holonomic_hand = (hc.pos_from(co) - ch.pos_from(co)).to_matrix(C)
    holonomic = holonomic_wheel.col_join(holonomic_hand)

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
        q11.diff(t) - u11,  # shoulder extension
        q12.diff(t) - u12,  # shoulder rotation
        q13.diff(t) - u13,  # elbow extension
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
    go.v2pt_theory(cg, N, G)
    ho.v2pt_theory(gh, N, H)

    # arm & handlebar joints
    cg.v2pt_theory(co, N, C)
    gh.v2pt_theory(cg, N, G)
    hc.v2pt_theory(gh, N, H)
    ch.v2pt_theory(fo, N, F)

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
        holonomic_hand[0].diff(t),
        holonomic_hand[1].diff(t),
        holonomic_hand[2].diff(t),
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

    ##############
    # Rigid Bodies
    ##############

    print('Defining the rigid bodies.')

    rear_frame = mec.RigidBody('Rear Frame', co, C, mc, (Ic, co))
    rear_wheel = mec.RigidBody('Rear Wheel', do, D, md, (Id, do))
    front_frame = mec.RigidBody('Front Frame', eo, E, me, (Ie, eo))
    front_wheel = mec.RigidBody('Front Wheel', fo, F, mf, (If, fo))
    upper_arm = mec.RigidBody('Upper Arm', go, G, mg, (Ig, go))
    lower_arm = mec.RigidBody('Lower Arm', ho, H, mh, (Ih, ho))

    bodies = [rear_frame, rear_wheel, front_frame, front_wheel, upper_arm,
              lower_arm]

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

    # input torques
    Tc = (C, T4*A['1'] - T6*B['2'] - T7*C['3'])
    Td = (D, T6*C['2'])
    Te = (E, T7*C['3'])

    forces = [Fco, Fdo, Feo, Ffo, Fgo, Fho, Tc, Td, Te]

    # Manually compute the ground contact velocities.
    kindiffdict = sm.solve(kinematical, [q3.diff(t), q4.diff(t), q5.diff(t),
                                         q7.diff(t), q11.diff(t), q12.diff(t),
                                         q13.diff(t)], dict=True)[0]
    u1_def = -rr*(u5 + u6)*sm.cos(q3)
    u1p_def = u1_def.diff(t).xreplace(kindiffdict)
    u2_def = -rr*(u5 + u6)*sm.sin(q3)
    u2p_def = u2_def.diff(t).xreplace(kindiffdict)

    ###############################
    # Prep symbolic data for output
    ###############################

    newto = N
    q_ind = (q3, q4, q7)  # yaw, roll, steer
    q_dep = (q5, q11, q12, q13)  # pitch
    # NOTE : I think q3 is an ignorable coordinate too.
    # rear contact 1 dist, rear contact 2 dist, rear wheel angle, front wheel angle
    q_ign = (q1, q2, q6, q8)
    u_ind = (u4, u6, u7)  # roll rate, rear wheel rate, steer rate
    u_dep = (u3, u5, u8, u11, u12, u13)  # yaw rate, pitch rate, front wheel rate
    const = (d1, d2, d3, d4, d5, d6, d7, d8, g, ic11, ic22, ic31, ic33, id11,
             id22, ie11, ie22, ie31, ie33, if11, if22, l1, l2, l3, l4, mc, md,
             me, mf, mg, mh, rf, rr)
    speci = (T4, T6, T7)
    holon = holonomic
    nonho = tuple(nonholonomic)
    us = tuple(sm.ordered((u1, u2) + u_ind + u_dep))
    qs = tuple(sm.ordered(q_ign + q_ind + q_dep))
    # TODO : Reduced these to work with formulate_equations_motion().
    spdef = {ui: qi.diff(t) for ui, qi in zip((u3, u4, u5, u7),
                                              (q3, q4, q5, q7))}
    exdef = {u1: u1_def, u2: u2_def, u1.diff(t): u1p_def, u2.diff(t): u2p_def}

    system_symbolics = {
        'bodies': tuple(bodies),
        'constants': const,
        'dependent generalized coordinates': q_dep,
        'dependent generalized speeds': u_dep,
        'extra definitions': exdef,
        'generalized coordinates': qs,
        'generalized speeds': us,
        'holonomic constraints': holon,
        'ignorable coordinates': q_ign,
        'independent generalized coordinates': q_ind,
        'independent generalized speeds': u_ind,
        'kinematical differential equations': kinematical,
        'loads': tuple(forces),
        'newtonian reference frame': newto,
        'nonholonomic constraints': nonho,
        'specified quantities': speci,
        'speed definitions': spdef,
        'time': t,
    }

    return system_symbolics


symbolics = setup_symbolics()

###############
# Kane's Method
###############

print("Generating Kane's equations.")

kane = mec.KanesMethod(
    symbolics['newtonian reference frame'],
    symbolics['independent generalized coordinates'],
    symbolics['independent generalized speeds'],
    kd_eqs=symbolics['kinematical differential equations'],
    q_dependent=symbolics['dependent generalized coordinates'],
    configuration_constraints=symbolics['holonomic constraints'],
    u_dependent=symbolics['dependent generalized speeds'],
    velocity_constraints=symbolics['nonholonomic constraints']
)

kane.kanes_equations(symbolics['bodies'], loads=symbolics['loads'])
