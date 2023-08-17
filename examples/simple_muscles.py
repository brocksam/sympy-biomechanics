import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sm
import sympy.physics.mechanics as me
from sympy.physics.mechanics.pathway import LinearPathway

from biomechanics import (FirstOrderActivationDeGroote2016,
                          MusculotendonDeGroote2016)
from biomechanics.plot import plot_config, plot_traj

q, u = me.dynamicsymbols('q, u')
m, c = sm.symbols('m, c')

N = me.ReferenceFrame('N')
O, P = sm.symbols('O, P', cls=me.Point)

P.set_pos(O, q*N.x)
O.set_vel(N, 0)
P.set_vel(N, u*N.x)

viscous_drag = me.Force(P, -c*u*N.x)

muscle_pathway = LinearPathway(O, P)
muscle_activation = FirstOrderActivationDeGroote2016('big_dog') #.with_default_constants('muscle')
muscle = MusculotendonDeGroote2016('muscle', muscle_pathway,
                                   activation_dynamics=muscle_activation)

block = me.Particle('block', P, m)

kane = me.KanesMethod(N, (q,), (u,), kd_eqs=(u - q.diff(),))
kane.kanes_equations((block,), (muscle.to_loads() + [viscous_drag]))
