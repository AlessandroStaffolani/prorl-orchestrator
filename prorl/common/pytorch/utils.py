import math
from typing import Optional

import torch
from torch import nn
from torch.nn.init import calculate_gain, _calculate_correct_fan, _calculate_fan_in_and_fan_out


def kaiming_uniform_(tensor, a=0.0,
                     mode='fan_in',
                     nonlinearity='leaky_relu',
                     generator: Optional[torch.Generator] = None
                     ) -> torch.Tensor:
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound, generator=generator)


def kaiming_normal_(tensor, a=0.0, mode='fan_in', nonlinearity='leaky_relu',
                    generator: Optional[torch.Generator] = None) -> torch.Tensor:
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std, generator=generator)


def init_bias(bias, weight, generator: Optional[torch.Generator] = None):
    fan_in, _ = _calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in)
    with torch.no_grad():
        return bias.uniform_(-bound, bound, generator=generator)


def init_linear_net_weights(net: nn.Module,
                            skip_not_gradient_layers=True,
                            init_function=kaiming_uniform_,
                            generator: Optional[torch.Generator] = None,
                            ):
    for layer in net.children():
        if layer.children() is not None:
            init_linear_net_weights(layer)
        # We skip layers without weights (such as Activation Functions)
        # and eventually, we skip layers with gradient disabled (such as target network,since it will be copied)
        if getattr(layer, 'weight', None) is not None:
            if skip_not_gradient_layers:
                if layer.weight.requires_grad is True:
                    layer.weight.data = init_function(layer.weight, a=math.sqrt(5), generator=generator)
                    if layer.bias is not None:
                        init_bias(layer.bias, layer.weight, generator=generator)
            else:
                layer.weight.data = init_function(layer.weight, a=math.sqrt(5), generator=generator)
                if layer.bias is not None:
                    init_bias(layer.bias, layer.weight, generator=generator)
