"""This file is used for specifying various schedulers that evolve over
time throughout the execution of the algorithm, such as:
 - exploration epsilon for the epsilon greedy exploration strategy
 - beta parameter for beta parameter in prioritized replay
Each scheduler has a function `value(t)` which returns the current value
of the parameter given the timestep t of the optimization procedure.
"""
import math
from abc import abstractmethod
from typing import Optional, Tuple, List

from prorl.common.enum_utils import ExtendedEnum


class EpsilonType(str, ExtendedEnum):
    Scalar = 'scalar'
    LinearDecay = 'linear-decay'
    AlternateLinearDecay = 'alternate-linear-decay'
    ExponentialDecay = 'exponential-decay'


class Scheduler(object):

    @abstractmethod
    def value(self, t: int, bootstrapped_steps: int = 0, index: Optional[int] = None):
        """Value of the schedule at time t"""
        pass


class ConstantScheduler(Scheduler):

    def __init__(self, value: float, **kwargs):
        """Value remains constant over time.
        Parameters
        ----------
        value: float
            Constant value of the schedule
        """
        self._v = value

    def value(self, t, bootstrapped_steps: int = 0, index: Optional[int] = None):
        """See Scheduler.value"""
        return self._v


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)


class PiecewiseScheduler(Scheduler):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None, **kwargs):
        """Piecewise scheduler.
        endpoints: [(int, int)]
            list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints = endpoints

    def value(self, t, bootstrapped_steps: int = 0, index: Optional[int] = None):
        """See Scheduler.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value


class LinearScheduler(Scheduler):

    def __init__(self, total: int, end: float, start=1.0, **kwargs):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        total: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        start: float
            initial output value
        end: float
            final output value
        """
        self.total_steps = total
        self.final_p = end
        self.initial_p = start

    def value(self, t: int, bootstrapped_steps: int = 0, index: Optional[int] = None):
        """See Scheduler.value"""
        if self.total_steps == 0:
            return self.final_p
        fraction = min(max(0.0, float(t) - bootstrapped_steps) / self.total_steps, 1.0)
        value = self.initial_p + fraction * (self.final_p - self.initial_p)
        # the max_min check ensures that value is in [self.initial_p, self.final_p]
        return max(min(self.initial_p, self.final_p), value)


class AlternateLinearScheduler(Scheduler):

    def __init__(
            self,
            total: int,
            end: float,
            start=1.0,
            alternate_values: Optional[List[Tuple[int, int]]] = None,
            **kwargs
    ):
        self.total_steps = total
        self.final_p = end
        self.initial_p = start
        self.alternate_values: Optional[List[Tuple[int, int]]] = alternate_values

    def value(self, t: int, bootstrapped_steps: int = 0, index: Optional[int] = None):
        """See Scheduler.value"""
        if self.total_steps == 0:
            return self.final_p
        time = max(0.0, float(t) - bootstrapped_steps)
        if index is not None and self.alternate_values is not None:
            index_range = self.alternate_values[index]
            if index_range[0] < time < index_range[1]:
                return self.final_p
        fraction = min(time / self.total_steps, 1.0)
        value = self.initial_p + fraction * (self.final_p - self.initial_p)
        # the max_min check ensures that value is in [self.initial_p, self.final_p]
        return max(min(self.initial_p, self.final_p), value)


class ExponentialScheduler(Scheduler):

    def __init__(self, decay: int, end: float, start: float = 1.0, **kwargs):
        """Exponential interpolation between initial_p and final_p. The decay factor
        defines how fast the value reaches final_p.
        Parameters
        ----------
        decay: int
            Exponential decay speed
        start: float
            initial output value
        end: float
            final output value
        """
        self.decay = decay
        self.final_p = end
        self.initial_p = start

    def value(self, t: int, bootstrapped_steps: int = 0, index: Optional[int] = None):
        """See Scheduler.value"""
        if self.decay == 0:
            return self.final_p
        time_step = max(0, t - bootstrapped_steps)
        exponential_factor = math.exp(-1 * time_step / self.decay)
        return self.final_p + (self.initial_p - self.final_p) * exponential_factor


def get_epsilon_scheduler(epsilon_type: EpsilonType = EpsilonType.Scalar, **epsilon_parameters) -> Scheduler:
    switcher = {
        EpsilonType.Scalar: ConstantScheduler,
        EpsilonType.LinearDecay: LinearScheduler,
        EpsilonType.AlternateLinearDecay: AlternateLinearScheduler,
        EpsilonType.ExponentialDecay: ExponentialScheduler,
    }

    if epsilon_type in switcher:
        return switcher[epsilon_type](**epsilon_parameters)
    else:
        raise AttributeError(f'EpsilonType {epsilon_type} not support. Available are: {EpsilonType.list()}')
