from sympy import (
    Float,
    Function,
    Integer,
    UnevaluatedExpr,
    exp,
    log,
    sin,
    sinh,
    sqrt,
)
from sympy.core.function import ArgumentIndexError


__all__ = [
    'FiberForceLengthActiveDeGroote2016',
    'FiberForceLengthPassiveDeGroote2016',
    'FiberForceLengthPassiveInverseDeGroote2016',
    'FiberForceVelocityDeGroote2016',
    'FiberForceVelocityInverseDeGroote2016',
    'TendonForceLengthDeGroote2016',
    'TendonForceLengthInverseDeGroote2016'
]


class CharacteristicCurveFunction(Function):

    def _eval_evalf(self, prec):
        return self.doit(deep=False, evaluate=False)._eval_evalf(prec)

    def _print_code(self, printer):
        return f'({printer.doprint(self.doit(deep=False, evaluate=False))})'

    _ccode = _print_code
    _cupycode = _print_code
    _cxxcode = _print_code
    _fcode = _print_code
    _jaxcode = _print_code
    _lambdacode = _print_code
    _mpmathcode = _print_code
    _octave = _print_code
    _pythoncode = _print_code
    _numpycode = _print_code
    _scipycode = _print_code


class TendonForceLengthDeGroote2016(CharacteristicCurveFunction):

    @classmethod
    def with_default_constants(cls, l_T_tilde):
        c0 = Float('0.2')
        c1 = Float('0.995')
        c2 = Float('0.25')
        c3 = Float('33.93669377311689')
        return cls(l_T_tilde, c0, c1, c2, c3)

    @classmethod
    def eval(cls, l_T_tilde, c0, c1, c2, c3):
        pass

    def doit(self, deep=True, evaluate=True, **hints):
        l_T_tilde, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            l_T_tilde = l_T_tilde.doit(deep=deep, **hints)
            c0, c1, c2, c3 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            c0, c1, c2, c3 = constants

        if evaluate:
            return c0 * exp(c3 * (l_T_tilde - c1)) - c2

        return c0 * exp(c3 * UnevaluatedExpr(l_T_tilde - c1)) - c2

    def fdiff(self, argindex=1):
        l_T_tilde, c0, c1, c2, c3 = self.args
        if argindex == 1:
            return c0 * c3 * exp(c3 * UnevaluatedExpr(l_T_tilde - c1))
        elif argindex == 2:
            return exp(c3 * UnevaluatedExpr(l_T_tilde - c1))
        elif argindex == 3:
            return -c0 * c3 * exp(c3 * UnevaluatedExpr(l_T_tilde - c1))
        elif argindex == 4:
            return Integer(-1)
        elif argindex == 5:
            return c0 * (l_T_tilde - c1) * exp(c3 * UnevaluatedExpr(l_T_tilde - c1))

        raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        return TendonForceLengthInverseDeGroote2016


class TendonForceLengthInverseDeGroote2016(CharacteristicCurveFunction):

    @classmethod
    def with_default_constants(cls, fl_T):
        c0 = Float('0.2')
        c1 = Float('0.995')
        c2 = Float('0.25')
        c3 = Float('33.93669377311689')
        return cls(fl_T, c0, c1, c2, c3)

    @classmethod
    def eval(cls, fl_T, c0, c1, c2, c3):
        pass

    def doit(self, deep=True, evaluate=True, **hints):
        fl_T, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            fl_T = fl_T.doit(deep=deep, **hints)
            c0, c1, c2, c3 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            c0, c1, c2, c3 = constants

        if evaluate:
            return log((fl_T + c2) / c0) / c3 + c1

        return log(UnevaluatedExpr((fl_T + c2) / c0)) / c3 + c1

    def fdiff(self, argindex=1):
        fl_T, c0, c1, c2, c3 = self.args
        if argindex == 1:
            return 1 / (c3 * (fl_T + c2))
        elif argindex == 2:
            return -1 / (c0 * c3)
        elif argindex == 3:
            return Integer(1)
        elif argindex == 4:
            return 1 / (c3 * (fl_T + c2))
        elif argindex == 5:
            return -log(UnevaluatedExpr((fl_T + c2) / c0)) / c3**2

        raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        return TendonForceLengthDeGroote2016


class FiberForceLengthPassiveDeGroote2016(CharacteristicCurveFunction):

    @classmethod
    def with_default_constants(cls, l_M_tilde):
        c0 = Float('0.6')
        c1 = Float('4.0')
        return cls(l_M_tilde, c0, c1)

    @classmethod
    def eval(cls, l_M_tilde, c0, c1):
        pass

    def doit(self, deep=True, evaluate=True, **hints):
        l_M_tilde, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            l_M_tilde = l_M_tilde.doit(deep=deep, **hints)
            c0, c1 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            c0, c1 = constants

        if evaluate:
            return (exp((c1 * (l_M_tilde - 1)) / c0) - 1) / (exp(c1) - 1)

        return (exp((c1 * UnevaluatedExpr(l_M_tilde - 1)) / c0) - 1) / (exp(c1) - 1)

    def fdiff(self, argindex=1):
        raise NotImplementedError

    def inverse(self, argindex=1):
        return FiberForceLengthPassiveInverseDeGroote2016


class FiberForceLengthPassiveInverseDeGroote2016(CharacteristicCurveFunction):

    @classmethod
    def with_default_constants(cls, fl_M_pas):
        c0 = Float('0.6')
        c1 = Float('4.0')
        return cls(fl_M_pas, c0, c1)

    @classmethod
    def eval(cls, fl_M_pas, c0, c1):
        pass

    def doit(self, deep=True, evaluate=True, **hints):
        fl_M_pas, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            fl_M_pas = fl_M_pas.doit(deep=deep, **hints)
            c0, c1 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            c0, c1 = constants

        return (c0 * log(((exp(c1) - 1) * fl_M_pas) + 1) / c1) + 1

    def fdiff(self, argindex=1):
        raise NotImplementedError

    def inverse(self, argindex=1):
        return FiberForceLengthPassiveDeGroote2016


class FiberForceLengthActiveDeGroote2016(CharacteristicCurveFunction):

    @classmethod
    def with_default_constants(cls, l_M_tilde):
        c0 = Float('0.814')
        c1 = Float('1.06')
        c2 = Float('0.162')
        c3 = Float('0.0633')
        c4 = Float('0.433')
        c5 = Float('0.717')
        c6 = Float('-0.0299')
        c7 = Float('0.200')
        c8 = Float('0.100')
        c9 = Float('1.00')
        c10 = Float('0.354')
        c11 = Float('0.00')
        return cls(l_M_tilde, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11)

    @classmethod
    def eval(cls, l_M_tilde, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11):
        pass

    def doit(self, deep=True, evaluate=True, **hints):
        l_M_tilde, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            l_M_tilde = l_M_tilde.doit(deep=deep, **hints)
            c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = constants

        a0 = l_M_tilde - c1
        a1 = c2 + c3 * l_M_tilde
        a2 = c0 * exp(-0.5 * ((a0 * a0) / (a1 * a1)))
        a3 = l_M_tilde - c5
        a4 = c6 + c7 * l_M_tilde
        a5 = c4 * exp(-0.5 * ((a3 * a3) / (a4 * a4)))
        a6 = l_M_tilde - c9
        a7 = c10 + c11 * l_M_tilde
        a8 = c8 * exp(-0.5 * ((a6 * a6) / (a7 * a7)))
        a9 = a2 + a5 + a8
        return a9

    def fdiff(self, argindex=1):
        raise NotImplementedError

    def inverse(self, argindex=1):
        raise NotImplementedError


class FiberForceVelocityDeGroote2016(CharacteristicCurveFunction):

    @classmethod
    def with_default_constants(cls, v_M_tilde):
        c0 = Float('-0.318')
        c1 = Float('-8.149')
        c2 = Float('-0.374')
        c3 = Float('0.886')
        return cls(v_M_tilde, c0, c1, c2, c3)

    @classmethod
    def eval(cls, v_M_tilde, c0, c1, c2, c3):
        pass

    def doit(self, deep=True, evaluate=True, **hints):
        v_M_tilde, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            v_M_tilde = v_M_tilde.doit(deep=deep, **hints)
            c0, c1, c2, c3 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            c0, c1, c2, c3 = constants

        a0 = c1 * v_M_tilde + c2
        a1 = sqrt(a0**2 + 1)
        a2 =  c0 * log(a0 + a1) + c3
        return a2

    def fdiff(self, argindex=1):
        raise NotImplementedError

    def inverse(self, argindex=1):
        return FiberForceVelocityInverseDeGroote2016


class FiberForceVelocityInverseDeGroote2016(CharacteristicCurveFunction):

    @classmethod
    def with_default_constants(cls, fv_M):
        c0 = Float('-0.318')
        c1 = Float('-8.149')
        c2 = Float('-0.374')
        c3 = Float('0.886')
        return cls(fv_M, c0, c1, c2, c3)

    @classmethod
    def eval(cls, fv_M, c0, c1, c2, c3):
        pass

    def doit(self, deep=True, evaluate=True, **hints):
        fv_M, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            fv_M = fv_M.doit(deep=deep, **hints)
            c0, c1, c2, c3 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            c0, c1, c2, c3 = constants

        return (sinh((fv_M - c3) / c0) - c2) / c1

    def fdiff(self, argindex=1):
        raise NotImplementedError

    def inverse(self, argindex=1):
        return FiberForceVelocityDeGroote2016
