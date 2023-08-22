import matplotlib.pyplot as plt
import numpy as np

from sympy import Symbol, lambdify

from sympy.physics._biomechanics import (
    FiberForceLengthActiveDeGroote2016,
    FiberForceLengthPassiveDeGroote2016,
    FiberForceVelocityDeGroote2016,
    TendonForceLengthDeGroote2016,
)


if False:
    l_T_tilde = Symbol('l_T_tilde')
    fl_T = TendonForceLengthDeGroote2016.with_default_constants(l_T_tilde)

    eval_fl_T = lambdify(l_T_tilde, fl_T, cse=True)

    l_T_tilde_vals = np.linspace(0.9, 1.1)

    plt.plot(l_T_tilde_vals, eval_fl_T(l_T_tilde_vals))
    plt.show()

if False:
    l_M_tilde = Symbol('l_M_tilde')
    fl_M_pas = FiberForceLengthPassiveDeGroote2016.with_default_constants(l_M_tilde)

    eval_fl_M_pas = lambdify(l_M_tilde, fl_M_pas, cse=True)

    l_M_tilde_vals = np.linspace(0.0, 2.0)

    plt.plot(l_M_tilde_vals, eval_fl_M_pas(l_M_tilde_vals))
    plt.show()

if False:
    l_M_tilde = Symbol('l_M_tilde')
    fl_M_act = FiberForceLengthActiveDeGroote2016.with_default_constants(l_M_tilde)

    eval_fl_M_act = lambdify(l_M_tilde, fl_M_act, cse=True)

    l_M_tilde_vals = np.linspace(0.0, 2.0)

    plt.plot(l_M_tilde_vals, eval_fl_M_act(l_M_tilde_vals))
    plt.show()

if True:
    v_M_tilde = Symbol('v_M_tilde')
    fv_M = FiberForceVelocityDeGroote2016.with_default_constants(v_M_tilde)

    eval_fv_M = lambdify(v_M_tilde, fv_M, cse=True)

    v_M_tilde_vals = np.linspace(-1.0, 1.0)

    plt.plot(v_M_tilde_vals, eval_fv_M(v_M_tilde_vals))
    plt.show()
