import json
from typing import Tuple, Dict, Union, Optional, List
from collections import namedtuple

import numpy as np
import torch

from prorl.common.encoders import NumpyEncoder


StateFeatureValue = namedtuple('StateFeatureValue', 'name value')


class StateFeature(
    namedtuple('StateFeature',
               'name size start')
):

    def to_dict(self):
        if getattr(self, '_asdict', None) is not None:
            return self._asdict()
        else:
            fields = self._fields
            d = {}
            for field in fields:
                d[field] = getattr(self, field)
            return d

    def __dict__(self):
        return self.to_dict()


class State(np.ndarray):

    def __new__(cls,
                *args,
                features: Tuple[StateFeature, ...] = None,
                feature_values: Optional[Tuple[StateFeatureValue, ...]] = None,
                dtype=np.float64,
                **kwargs):
        if features is None and feature_values is None:
            # args or kwargs contain the array of values but we have not any features
            return np.array(*args, dtype=dtype, **kwargs).view(cls)
        if feature_values is None:
            # args or kwargs contain the array of values and we have the features
            obj = np.array(*args, dtype=dtype, **kwargs).view(cls)
            obj._features = features
            return obj
        # we build the array and the features from the feature values
        state_features: List[StateFeature] = []
        values = []
        last_end = 0
        for feature in feature_values:
            if isinstance(feature.value, list) or isinstance(feature.value, np.ndarray):
                state_features.append(StateFeature(name=feature.name, size=len(feature.value), start=last_end))
                values += list(feature.value)
                last_end += len(feature.value)
            else:
                raise AttributeError('The value of a StateFeatureValue tuple must be an array')
        obj = np.array(values, *args, dtype=dtype, **kwargs).view(cls)
        obj._features = tuple(state_features)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._features: Tuple[StateFeature, ...] = getattr(obj, '_feature', None)

    def features(self, as_dict=False) -> Union[Tuple[StateFeature, ...], List[Dict[str, Union[str, int]]]]:
        if as_dict:
            return [feature.to_dict() for feature in self._features]
        return self._features

    def get_feature(self, name, values=False) -> Union[StateFeature, Tuple[StateFeature, np.ndarray]]:
        f = None
        for feature in self._features:
            if feature.name == name:
                f = feature
        if values:
            return f, self.get_feature_value(name)
        return f

    def get_feature_value(self, name) -> np.ndarray:
        feature: StateFeature = self.get_feature(name)
        return self[feature.start: feature.start + feature.size]

    def set_feature_value(self,
                          name: str,
                          new_value: Union[List[float], float, int],
                          index: Optional[int] = None,
                          reset_value: Optional[Union[int, float]] = None
                          ):
        feature: StateFeature = self.get_feature(name=name, values=False)
        if isinstance(new_value, list):
            self[feature.start: feature.start + feature.size] = np.array(new_value)
        elif isinstance(new_value, float) or isinstance(new_value, int):
            if index is None:
                raise AttributeError('State set_feature_value passed single value for update but index is None')
            for i in range(feature.start, feature.start + feature.size):
                if index == i - feature.start:
                    self[i] = new_value
                elif reset_value is not None:
                    self[i] = reset_value
        else:
            raise AttributeError('State set_feature_value new_value must be List[float], float or int')

    def to_dict(self):
        return {
            'features': self._features,
            'dtype': str(self.dtype),
            'data': self,
        }

    def __hash__(self):
        return hash(json.dumps(self, cls=NumpyEncoder))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def to_tensor(self, device='cpu') -> torch.Tensor:
        return torch.from_numpy(self).float().to(device=device)

    def to_json(self):
        return json.dumps(self.to_dict(), cls=NumpyEncoder)

    def to_matrix(self) -> np.ndarray:
        max_len_feature = 0
        for feature in self._features:
            if feature.size > max_len_feature:
                max_len_feature = feature.size
        matrix = np.zeros((len(self._features), max_len_feature))
        for i, feature in enumerate(self._features):
            feature_values = self.get_feature_value(feature.name)
            for j in range(max_len_feature):
                if j < len(feature_values):
                    matrix[i, j] = feature_values[j]
                else:
                    matrix[i, j] = None
        return matrix

    @classmethod
    def from_dict(cls, dict_state):
        if dict_state['features'] is not None:
            features = [StateFeature(*feature) for feature in dict_state['features']]
            return cls(dict_state['data'], features=features, dtype=dict_state['dtype'])
        else:
            return cls(dict_state['data'], dtype=dict_state['dtype'])

    @classmethod
    def from_json(cls, json_str):
        return State.from_dict(json.loads(json_str))

    @classmethod
    def add_feature(cls, state, feature: StateFeatureValue):
        current_features: List[StateFeature] = list(state.features())
        if isinstance(feature.value, list) or isinstance(feature.value, np.ndarray):
            new_feature: StateFeature = StateFeature(name=feature.name, size=len(feature.value), start=len(state))
        else:
            raise AttributeError('The value of a StateFeatureValue tuple must be an array')
        current_features.append(new_feature)
        values = state.tolist()
        values += feature.value
        return cls(values, features=current_features)

    @classmethod
    def get_state_slice(cls, state, features: List[str]):
        features_values: List[StateFeatureValue] = []
        for f_name in features:
            features_values.append(StateFeatureValue(name=f_name, value=state.get_feature_value(f_name)))
        return cls(feature_values=tuple(features_values))

    @staticmethod
    def slice_multi_dim_state(states: np.ndarray, feature_slice: slice):
        return states[:, feature_slice]
