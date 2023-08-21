from enum import IntEnum, auto, unique

from sympy import Function, Integer, Symbol, sin, sqrt
from sympy.physics.mechanics.actuator import ForceActuator

from biomechanics.activation import ActivationBase
from biomechanics.curve import (
    FiberForceLengthActiveDeGroote2016,
    FiberForceLengthPassiveDeGroote2016,
    FiberForceLengthPassiveInverseDeGroote2016,
    FiberForceVelocityDeGroote2016,
    FiberForceVelocityInverseDeGroote2016,
    TendonForceLengthDeGroote2016,
    TendonForceLengthInverseDeGroote2016,
)
from biomechanics.mixin import _NamedMixin


__all__ = [
    'MusculotendonDeGroote2016',
    'MusculotendonDynamics',
]


@unique
class MusculotendonDynamics(IntEnum):
    RIGID_TENDON = auto()


class MusculotendonDeGroote2016(ForceActuator, _NamedMixin):

    def __init__(
        self,
        name,
        pathway,
        *,
        activation_dynamics,
        musculotendon_dynamics=MusculotendonDynamics.RIGID_TENDON,
        tendon_slack_length=None,  # Sensible value: 0.05
        peak_isometric_force=None,  # Sensible value: 1000
        optimal_fiber_length=None,  # Sensible value: 0.25
        maximal_fiber_velocity=None,  # Default value: 10
        optimal_pennation_angle=None,  # Default value: 0
        fiber_damping_coefficient=None,  # Default value: 0.1
    ):
        self.name = name
        super().__init__(Symbol('F'), pathway)

        # Activation dynamics
        self._child_objects = ()
        self.activation_dynamics = activation_dynamics

        # Constants
        self.tendon_slack_length = tendon_slack_length
        self.peak_isometric_force = peak_isometric_force
        self.optimal_fiber_length = optimal_fiber_length
        self.maximal_fiber_velocity = maximal_fiber_velocity
        self.optimal_pennation_angle = optimal_pennation_angle
        self.fiber_damping_coefficient = fiber_damping_coefficient

        # Musculotendon dynamics
        self.musculotendon_dynamics = musculotendon_dynamics
        self._force = -self._F_T

    @property
    def tendon_slack_length(self):
        return self._l_T_slack

    @tendon_slack_length.setter
    def tendon_slack_length(self, l_T_slack):
        if l_T_slack is not None:
            self._l_T_slack = l_T_slack
        else:
            self._l_T_slack = Symbol(f'l_T_slack_{self.name}')

    @property
    def peak_isometric_force(self):
        return self._F_M_max

    @peak_isometric_force.setter
    def peak_isometric_force(self, F_M_max):
        if F_M_max is not None:
            self._F_M_max = F_M_max
        else:
            self._F_M_max = Symbol(f'F_M_max_{self.name}')

    @property
    def optimal_fiber_length(self):
        return self._l_M_opt

    @optimal_fiber_length.setter
    def optimal_fiber_length(self, l_M_opt):
        if l_M_opt is not None:
            self._l_M_opt = l_M_opt
        else:
            self._l_M_opt = Symbol(f'l_M_opt_{self.name}')

    @property
    def maximal_fiber_velocity(self):
        return self._v_M_max

    @maximal_fiber_velocity.setter
    def maximal_fiber_velocity(self, v_M_max):
        if v_M_max is not None:
            self._v_M_max = v_M_max
        else:
            self._v_M_max = Symbol(f'v_M_max_{self.name}')

    @property
    def optimal_pennation_angle(self):
        return self._alpha_opt

    @optimal_pennation_angle.setter
    def optimal_pennation_angle(self, alpha_opt):
        if alpha_opt is not None:
            self._alpha_opt = alpha_opt
        else:
            self._alpha_opt = Symbol(f'alpha_opt_{self.name}')

    @property
    def fiber_damping_coefficient(self):
        return self._beta

    @fiber_damping_coefficient.setter
    def fiber_damping_coefficient(self, beta):
        if beta is not None:
            self._beta = beta
        else:
            self._beta = Symbol(f'beta_{self.name}')

    @property
    def state_variables(self):
        y_vars = self._state_variables
        for c in self._child_objects:
            y_vars += c.state_variables
        return y_vars

    @property
    def control_variables(self):
        u_vars = self._control_variables
        for c in self._child_objects:
            u_vars += c.control_variables
        return u_vars

    @property
    def state_equations(self):
        y_eqns = self._state_equations
        for c in self._child_objects:
            y_eqns = {**y_eqns, **c.state_equations}
        return y_eqns

    @property
    def activation_dynamics(self):
        return self._activation_dynamics

    @activation_dynamics.setter
    def activation_dynamics(self, activation_dynamics):
        if hasattr(self, '_activation_dynamics'):
            msg = (
                f'Can\'t set attribute `activation_dynamics` to {activation} '
                f'as it is immutable.'
            )
            raise AttributeError(msg)
        if not isinstance(activation_dynamics, ActivationBase):
            msg = (
                f'Can\'t set attribute `activation_dynamics` to '
                f'{activation_dynamics} as it must be of type `str`, not '
                f'{type(activation_dynamics)}.'
            )
            raise TypeError(msg)
        self._activation_dynamics = activation_dynamics
        self._child_objects += (self._activation_dynamics, )

    @property
    def a(self):
        return self.activation_dynamics.a

    @property
    def e(self):
        return self.activation_dynamics.e

    @property
    def musculotendon_dynamics(self):
        return self._musculotendon_dynamics

    @musculotendon_dynamics.setter
    def musculotendon_dynamics(self, musculotendon_dynamics):
        if musculotendon_dynamics == MusculotendonDynamics.RIGID_TENDON:
            self._type_0_musculotendon_dynamics()
        else:
            msg = (
                f'Musculotendon dynamics {repr(musculotendon_dynamics)} '
                f'passed to `musculotendon_dynamics` was of type '
                f'{type(musculotendon_dynamics)}, must be '
                f'{MusculotendonDynamics}.'
            )
            raise TypeError(msg)
        self._musculotendon_dynamics = musculotendon_dynamics

    def _type_0_musculotendon_dynamics(self):
        """Rigid tendon musculotendon."""
        self._l_MT = self.pathway.length
        self._v_MT = -self.pathway.extension_velocity
        self._l_T = self._l_T_slack
        self._l_T_tilde = Integer(1)
        self._l_M = sqrt((self._l_MT - self._l_T)**2 + (self._l_M_opt * sin(self._alpha_opt))**2)
        self._l_M_tilde = self._l_M / self._l_M_opt
        self._v_M = self._v_MT * (self._l_MT - self._l_T_slack) / self._l_M
        self._v_M_tilde = self._v_M / self._v_M_max
        self._fl_T = TendonForceLengthDeGroote2016.with_default_constants(self._l_T_tilde)
        self._fl_M_pas = FiberForceLengthPassiveDeGroote2016.with_default_constants(self._l_M_tilde)
        self._fl_M_act = FiberForceLengthActiveDeGroote2016.with_default_constants(self._l_M_tilde)
        self._fv_M = FiberForceVelocityDeGroote2016.with_default_constants(self._v_M_tilde)
        self._F_M_tilde = self.a * self._fl_M_act * self._fv_M + self._fl_M_pas + self._beta * self._v_M_tilde
        self._F_T_tilde = self._F_M_tilde
        self._F_M = self._F_M_tilde * self._F_M_max
        self._F_T = self._F_M

        # Containers
        self._state_variables = ()
        self._control_variables = ()
        self._state_equations = {}
        self._auxiliary_data = {}

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.name!r}, '
            f'pathway={self.pathway!r}, '
            f'activation_dynamics={self.activation_dynamics!r}, '
            f'musculotendon_dynamics={self.musculotendon_dynamics!r}, '
            f'tendon_slack_length={self._l_T_slack}, '
            f'peak_isometric_force={self._F_M_max}, '
            f'optimal_fiber_length={self._l_M_opt}, '
            f'maximal_fiber_velocity={self._v_M_max}, '
            f'optimal_pennation_angle={self._alpha_opt}, '
            f'fiber_damping_coefficient={self._beta})'
        )
