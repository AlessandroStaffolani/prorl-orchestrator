import logging
from typing import Dict

from prorl.common.controller.abstract_controller import Controller, ActionError


class ActionResult:

    def __init__(self, result=None, error=False, error_info=None):
        self.result = result
        self.error: bool = error
        self.error_info = error_info

    def add_error(self, info):
        self.error = True
        self.error_info = info


class MainController:

    def __init__(self, logger: logging.Logger):
        self.logger: logging.Logger = logger
        self.controllers: Dict[str, Controller.__class__] = {}
        self.controller_aliases: Dict[str, str] = {}

    def add_controller(self, name: str, controller: Controller.__class__):
        self.controllers[name] = controller

    def add_alias(self, name, target):
        self.controller_aliases[name] = target

    def get_controller(self, controller_name: str):
        controller_name = controller_name.lower()
        if controller_name in self.controllers:
            return self.controllers[controller_name]
        elif controller_name in self.controller_aliases:
            return self.controllers[self.controller_aliases[controller_name]]
        else:
            return None

    def execute(self, controller: str, action: str, action_arguments) -> ActionResult:
        action_result = ActionResult()
        try:
            controller_class = self.get_controller(controller)
            if controller_class is None:
                message = f'Controller "{controller}" is not available.'
                self.logger.error(message)
                action_result.add_error({'message': message})
            else:
                action = action.lower()
                controller_instance: Controller = controller_class()
                self.logger.info(f'Starting controller {controller} with action {action}')
                action_result.result = controller_instance.execute(action, **action_arguments)
        except ActionError:
            message = f'Action "{action}" is not available on controller "{controller}".'
            self.logger.error(message)
            action_result.add_error({'message': message})
        except Exception as e:
            self.logger.exception(e)
            action_result.add_error({'message': str(e)})
        finally:
            return action_result
