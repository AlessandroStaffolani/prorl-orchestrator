import math

from prorl.common.enum_utils import ExtendedEnum
from prorl.environment.agent.parameter_schedulers import EpsilonType


def scalar_epsilon(value: float, **kwargs) -> float:
    return value


def linear_decay(start: float, end: float, total: float, step=0, **kwargs) -> float:
    return end + max(0, (start - end) * ((total - step)/total))


def exponential_decay(start: float, end: float, decay: float, step=0, **kwargs) -> float:
    return end + (start - end) * math.exp(-1 * step / decay)


def get_epsilon_value(epsilon_type: EpsilonType = EpsilonType.Scalar, **epsilon_parameters) -> float:
    switcher = {
        EpsilonType.Scalar: scalar_epsilon,
        EpsilonType.LinearDecay: linear_decay,
        EpsilonType.ExponentialDecay: exponential_decay,
    }

    if epsilon_type in switcher:
        return switcher[epsilon_type](**epsilon_parameters)
    else:
        raise AttributeError(f'EpsilonType {epsilon_type} not support. Available are: {EpsilonType.list()}')
