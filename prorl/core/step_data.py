import json
from collections import namedtuple
from typing import Union, Dict, List

import numpy as np

from prorl.common.enum_utils import ExtendedEnum
from prorl.core.timestep import Step

StepDataEntry = namedtuple('StepDataEntry', 'resource value base_station step')


class Operators(str, ExtendedEnum):
    Sum = 'sum'
    Diff = 'diff'
    Mul = 'mul'
    Div = 'div'
    Exp = 'exp'


def operation(left: Union[int, float], right: Union[int, float], operator: Operators) -> Union[int, float]:
    if operator == Operators.Sum:
        return left + right
    elif operator == Operators.Diff:
        return left - right
    elif operator == Operators.Mul:
        return left * right
    elif operator == Operators.Div:
        return left / right
    elif operator == Operators.Exp:
        return left ** right
    else:
        raise AttributeError(f'Operator: {operator} not supported')


class SavingFormat(str, ExtendedEnum):
    FullData = 'full-data'
    OnlyData = 'only-data'
    Array = 'array'


class DataEntryException(Exception):

    def __init__(self, message, *args):
        super(DataEntryException, self).__init__(message, *args)


class StepData:

    def __init__(self, *entries, generator_model=None):
        self._data: Dict[str, List[Dict[str, float]]] = {}
        self.generator_model = generator_model
        self.step: Union[Step, None] = None
        self._add_multiple_entries(*entries)

    def add_entry(self, entry: StepDataEntry, *extra_entries):
        self._add_data_entry(entry),
        self._add_multiple_entries(*extra_entries)

    def _add_multiple_entries(self, *entries):
        for entry in entries:
            if not isinstance(entry, StepDataEntry):
                raise DataEntryException('Passed a not tuple of GeneratedData tuples')
            self._add_data_entry(entry)

    def _add_data_entry(self, entry: StepDataEntry):
        resource, value, base_station, step = entry
        if self.step is None:
            self.step = step
        elif self.step != step:
            raise DataEntryException('Inconsistent step value for resource {} and base station {}'
                                     .format(resource, base_station))
        if resource not in self._data:
            self._data[resource] = []
        self._data[resource].append({base_station: value})

    def get_resource_values(self, resource, as_array=False):
        if as_array:
            data = []
            for res_values in self._data[resource]:
                for _, value in res_values.items():
                    data.append(value)
            return data
        else:
            return self[resource]

    def get_base_station_values(self, resource, index, base_station):
        return self._data[resource][index][base_station]

    def __getitem__(self, item):
        return self._data[item]

    def to_array(self):
        data = []
        for resource, _ in self._data.items():
            data.append(self.get_resource_values(resource, as_array=True))
        return data

    def to_numpy(self, *args, **kwargs):
        return np.array(self.to_array(), *args, **kwargs)

    def to_dict(self, add_generator_model=False):
        d = {
            'step': self.step.to_dict(),
            'data': self._data
        }
        if add_generator_model:
            d['generator_model'] = self.generator_model
        return d

    def to_json(self):
        return json.dumps(self.to_dict())

    def __str__(self):
        return f'<StepData step={self.step} data={self._data} >'

    def apply_operation(self, operator: Operators, scalar: Union[int, float], inplace=False) -> 'StepData':
        if isinstance(scalar, (int, float, np.int64, np.float64)):
            new_data = {}
            for res, all_bs_data in self._data.items():
                new_data[res] = []
                for index, bs_data in enumerate(all_bs_data):
                    new_data[res].append({})
                    for bs_name, bs_value in bs_data.items():
                        new_data[res][index][bs_name] = operation(bs_value, scalar, operator)
            if inplace:
                self._data = new_data
            else:
                return self.from_dict({
                    'step': self.step.to_dict(),
                    'data': new_data
                })
        else:
            raise AttributeError('StepData operation can be applied only on scalar values')

    def __mul__(self, other) -> 'StepData':
        return self.apply_operation(Operators.Mul, other)

    def __rmul__(self, other) -> 'StepData':
        return self.apply_operation(Operators.Mul, other)

    def __add__(self, other) -> 'StepData':
        return self.apply_operation(Operators.Sum, other)

    def __sub__(self, other) -> 'StepData':
        return self.apply_operation(Operators.Diff, other)

    def __truediv__(self, other) -> 'StepData':
        return self.apply_operation(Operators.Div, other)

    def __pow__(self, power, modulo=None) -> 'StepData':
        return self.apply_operation(Operators.Exp, power)

    def to_saving_format(self, saving_format: SavingFormat):
        if saving_format == SavingFormat.FullData:
            return self.to_dict()
        if saving_format == SavingFormat.OnlyData:
            return self.to_dict()['data']
        if saving_format == SavingFormat.Array:
            return self.to_array()

    @classmethod
    def from_dict(cls, dict_values):
        data = cls()
        data._data = dict_values['data']
        data.step = Step(**dict_values['step'])
        if 'generator_model' in dict_values:
            data.generator_model = dict_values['generator_model']
        return data

    @classmethod
    def from_json(cls, string_value):
        return StepData.from_dict(json.loads(string_value))
