from typing import Dict

from prorl.common.controller import Controller, MainController
from prorl.environment import logger
from prorl.run.controller.runner_controller import RunnerController
from prorl.run.controller.scheduler_controller import SchedulerController
from prorl.run.controller.worker_controller import WorkerController

CONTROLLERS: Dict[str, Controller.__class__] = {
    'runner': RunnerController,
    'worker': WorkerController,
    'scheduler': SchedulerController,
}
CONTROLLER_ALIASES: Dict[str, str] = {}


def get_main_controller() -> MainController:
    main_controller = MainController(logger)
    for name, controller in CONTROLLERS.items():
        main_controller.add_controller(name, controller)
    for name, target in CONTROLLER_ALIASES.items():
        main_controller.add_alias(name, target)
    return main_controller
