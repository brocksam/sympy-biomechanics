from abc import ABC, abstractmethod
from functools import cached_property

from sympy import Float, Integer, Rational, Symbol, tanh

from biomechanics.mixin import _NamedMixin


__all__ = [
    'FirstOrderActivationDeGroote2016',
    'ZerothOrderActivation',
]


class ActivationBase(ABC, _NamedMixin):

    def __init__(self, name):
        self.name = str(name)

        # Symbols
        self._a = Symbol(f"a_{name}")
        self._e = Symbol(f"e_{name}")

    @property
    def a(self):
        """Symbol representing activation."""
        return self._a

    @property
    def e(self):
        """Symbol representing excitation."""
        return self._e

    @property
    def order(self):
        """Order of the (differential) equation governing activation."""
        return self._ORDER

    @property
    @abstractmethod
    def state_variables(self):
        pass

    @property
    @abstractmethod
    def control_variables(self):
        pass

    @property
    @abstractmethod
    def state_equations(self):
        pass

    @property
    @abstractmethod
    def auxiliary_data(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass


class ZerothOrderActivation(ActivationBase):

    _ORDER = 0

    def __init__(self, name):
        super().__init__(name)

    @property
    def state_variables(self):
        return ()

    @property
    def control_variables(self):
        return (self.e, )

    @property
    def state_equations(self):
        return {}

    @property
    def auxiliary_data(self):
        aux_data = {
            self.a: self.e,
        }
        return aux_data

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name})'


class FirstOrderActivationDeGroote2016(ActivationBase):

    _ORDER = 1

    def __init__(self,
        name,
        activation_time_constant=None,
        deactivation_time_constant=None,
    ):
        super().__init__(name)

        # Symbols
        self.tau_a = activation_time_constant
        self.tau_d = deactivation_time_constant

    @classmethod
    def with_default_constants(cls, name):
        tau_a = Float('0.015')
        tau_d = Float('0.060')
        return cls(name, tau_a, tau_d)

    @property
    def tau_a(self):
        return self._tau_a

    @tau_a.setter
    def tau_a(self, tau_a):
        if hasattr(self, '_tau_a'):
            msg = (
                f'Can\'t set attribute `tau_a` to {repr(tau_a)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        if tau_a is None:
            self._tau_a = Symbol(f"tau_a_{self.name}")
        else:
            self._tau_a = tau_a

    @property
    def tau_d(self):
        return self._tau_d

    @tau_d.setter
    def tau_d(self, tau_d):
        if hasattr(self, '_tau_d'):
            msg = (
                f'Can\'t set attribute `tau_d` to {repr(tau_d)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        if tau_d is None:
            self._tau_d = Symbol(f"tau_d_{self.name}")
        else:
            self._tau_d = tau_d

    @property
    def state_variables(self):
        return (self.a, )

    @property
    def control_variables(self):
        return (self.e, )

    @property
    def state_equations(self):
        return {self.a: self._da_eqn}

    @property
    def auxiliary_data(self):
        aux_data = {}
        return aux_data

    @cached_property
    def _da_eqn(self):
        HALF = Rational(1, 2)
        c0 = Integer(100)
        a0 = HALF * tanh(c0 * (self.e - self.a))
        a1 = (HALF + Rational(3, 2) * self.a)
        a2 = (HALF + a0) / (self.tau_a * a1)
        a3 = a1 * (HALF - a0) / self.tau_d
        activation_dynamics_equation = (a2 + a3) * (self.e - self.a)
        return activation_dynamics_equation

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.name}, '
            f'activation_time_constant={self.tau_a}, '
            f'deactivation_time_constant={self.tau_d})'
        )
