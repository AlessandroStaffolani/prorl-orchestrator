from typing import Dict, Callable


class ActionError(Exception):
    def __init__(self, action: str, controller_name: str, *args):
        message = f'Requested not available action "{action}" to controller "{controller_name}"'
        super(ActionError, self).__init__(message, *args)


class Controller:

    def __init__(self, name='Controller'):
        self.name = name
        self.actions: Dict[str, Callable] = {}
        self.action_aliases: Dict[str, str] = {}

    def _add_action(self, name: str, action: Callable):
        if name not in self.actions:
            self.actions[name] = action
        else:
            raise AttributeError(f'Training to override action function for action {name} on controller {self.name}')

    def _add_action_alias(self, alias_name, alias_target):
        self.action_aliases[alias_name] = alias_target

    def get_action(self, action_name: str):
        action_name = action_name.lower()
        if action_name in self.actions:
            return self.actions[action_name]
        elif action_name in self.action_aliases:
            return self.actions[self.action_aliases[action_name]]
        else:
            return None

    def execute(self, action_name, *action_args, **action_kwargs):
        action = self.get_action(action_name)
        if action is not None:
            return action(*action_args, **action_kwargs)
        else:
            raise ActionError(action_name, self.name)
