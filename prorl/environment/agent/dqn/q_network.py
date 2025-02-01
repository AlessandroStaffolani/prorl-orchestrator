import copy
import logging
from typing import Union, List

import torch
from torch import nn

from prorl.environment.agent.pytorch_net import fully_connected
from prorl.environment.config import NetworkType


class InvalidQNetModelError(Exception):

    def __init__(self, model: str, *args):
        super(InvalidQNetModelError, self).__init__(
            f'The requested model "{model}" is invalid. Use one between "online" and "target"',
            *args
        )


class QNet(torch.nn.Module):

    def __init__(self, online_net, state_dict=None):
        super(QNet, self).__init__()

        # Online network -> the action-value function Q
        self.online = online_net
        if state_dict is not None:
            self.online.load_state_dict(state_dict)

        # Target network -> the target action-value function
        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, net_input, model: str = 'online'):
        if model == 'online':
            return self.online(net_input)
        elif model == 'target':
            return self.target(net_input)
        else:
            raise InvalidQNetModelError(model)

    def copy_online_on_target(self):
        self.target.load_state_dict(copy.deepcopy(self.online.state_dict()))

    def log_weights(self, model: str, logger: logging.Logger, level=logging.INFO, title=None):
        if model == 'online':
            parameters = self.online.parameters()
        elif model == 'target':
            parameters = self.target.parameters()
        else:
            raise InvalidQNetModelError(model)
        title_message = f'{model.capitalize()} model parameters'
        if title is not None:
            title_message += f': {title}'
        logger.log(level, title_message)
        logger.log(level, self)
        for i, param in enumerate(parameters):
            logger.log(level, f"Level {i} with shape: {param.shape} and values: {param.tolist()}")

    def count_trainable_parameters(self, model):
        if model == 'online':
            parameters = self.online.parameters()
        elif model == 'target':
            parameters = self.target.parameters()
        else:
            raise InvalidQNetModelError(model)
        return sum(p.numel() for p in parameters if p.requires_grad)

    def export_state(self, model='online'):
        if model == 'online':
            return self.online.state_dict()
        elif model == 'target':
            return self.target.state_dict()
        else:
            raise InvalidQNetModelError(model)

    def load_model(self, model_state, model='online', copy_online_on_target=True):
        if model == 'online':
            self.online.load_state_dict(copy.deepcopy(model_state))
            if copy_online_on_target:
                self.copy_online_on_target()
        elif model == 'target':
            self.target.load_state_dict(copy.deepcopy(model_state))
        else:
            raise InvalidQNetModelError(model)

    def set_eval(self):
        self.online.eval()
        self.target.eval()
        self.eval()

    def set_train(self):
        self.online.train()
        self.target.train()
        self.train()


class FullyConnected(QNet):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_units: Union[int, List[int]],
                 activation: str = 'relu',
                 batch_normalization=True,
                 state_dict=None,
                 **kwargs):
        net, _ = fully_connected(
            input_size=input_size,
            output_size=output_size,
            hidden_units=hidden_units,
            activation=activation,
            batch_normalization=batch_normalization,
            add_output_layer=True
        )

        super(FullyConnected, self).__init__(
            online_net=net,
            state_dict=state_dict
        )


class DuelingQNet(nn.Module):

    def __init__(
            self,
            input_size: int,
            output_size: int,
            hidden_units: Union[int, List[int]],
            activation: str = 'relu',
            batch_normalization: bool = True
    ):
        super(DuelingQNet, self).__init__()
        self.shared_layers, last_layer_units = fully_connected(
            input_size=input_size,
            output_size=output_size,
            hidden_units=hidden_units,
            activation=activation,
            batch_normalization=batch_normalization,
            add_output_layer=False
        )
        self.value = nn.Linear(last_layer_units, 1)
        self.advantage = nn.Linear(last_layer_units, output_size)

    def forward(self, x):
        shared_features = self.shared_layers(x)
        value = self.value(shared_features)
        raw_advantage = self.advantage(shared_features)
        reduce_over = tuple(range(1, raw_advantage.dim()))
        advantage = raw_advantage - raw_advantage.mean(dim=reduce_over, keepdim=True)
        return value + advantage  # return the q-values


class DuelingQNetWrapper(QNet):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_units: Union[int, List[int]],
                 activation: str = 'relu',
                 batch_normalization=True,
                 state_dict=None,
                 **kwargs):
        super(DuelingQNetWrapper, self).__init__(
            online_net=DuelingQNet(input_size, output_size, hidden_units, activation, batch_normalization),
            state_dict=state_dict,
        )


class QNetworkFactory:
    __available_networks = {
        NetworkType.FullyConnected: FullyConnected,
        NetworkType.Dueling: DuelingQNetWrapper,
    }

    @staticmethod
    def get_net(type_name: NetworkType, device=torch.device('cpu'), **kwargs) -> QNet:
        if type_name in QNetworkFactory.__available_networks:
            return QNetworkFactory.__available_networks[type_name](**kwargs).to(device)
        else:
            raise AttributeError('Q-Network {} not available. Try one of the available {}'
                                 .format(type_name, NetworkType.list()))
