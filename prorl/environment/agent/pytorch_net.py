from typing import Union, List, Tuple

import torch
import torch.nn as nn


from prorl.environment.config import NetworkType


def get_hidden_layers_units(hidden_units: Union[int, List[int]]) -> List[int]:
    if isinstance(hidden_units, list):
        return [unit for unit in hidden_units]
    else:
        return [hidden_units]


ACTIVATION_MAPPING = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'identity': nn.Identity,
    'sigmoid': nn.Sigmoid,
    'gelu': nn.GELU,
    'elu': nn.ELU
}


def fully_connected(
        input_size: int,
        output_size: int,
        hidden_units: Union[int, List[int]],
        activation='relu',
        batch_normalization=False,
        add_output_layer: bool = True,
        **kwargs,
) -> Tuple[nn.Module, int]:
    hidden_layers_units: List[int] = get_hidden_layers_units(hidden_units)
    layers = []
    previous_layer = input_size
    last_units = input_size
    for units in hidden_layers_units:
        layers.append(nn.Linear(previous_layer, units))
        if batch_normalization:
            layers.append(nn.BatchNorm1d(units))
        layers.append(ACTIVATION_MAPPING[activation]())
        previous_layer = units
        last_units = units

    if add_output_layer:
        layers.append(nn.Linear(last_units, output_size))

    return nn.Sequential(*layers), last_units


def fully_connected_variable(
        input_size: int,
        output_size: int,
        hidden_layers: int = 2,
        activation='relu',
        batch_normalization=False,
        add_output_layer: bool = True,
        **kwargs,
) -> Tuple[nn.Module, int]:
    hidden_layers = hidden_layers
    modules = []
    previous_layer_output = input_size
    # create hidden layers
    for i in range(hidden_layers):
        next_layer_input = int(previous_layer_output / 2)
        next_layer_input = next_layer_input if next_layer_input > 1 else 1
        modules.append(nn.Linear(previous_layer_output, next_layer_input))
        if batch_normalization:
            modules.append(nn.BatchNorm1d(next_layer_input))
        modules.append(ACTIVATION_MAPPING[activation]())
        previous_layer_output = next_layer_input

    if add_output_layer:
        # finally add the output layer
        modules.append(nn.Linear(previous_layer_output, output_size))

    return nn.Sequential(*modules), previous_layer_output


NET_MAPPING = {
    NetworkType.FullyConnected: fully_connected,
}


def get_network(net_type: NetworkType, input_size: int, output_size: int,
                add_output_layer: bool = True, **net_params) -> Tuple[nn.Module, int]:
    if net_type in NET_MAPPING:
        net_params['add_output_layer'] = add_output_layer
        return NET_MAPPING[net_type](input_size, output_size, **net_params)
    else:
        raise AttributeError('net_type not available')
