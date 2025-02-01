from typing import Dict

from prorl.common.controller.abstract_controller import Controller
from prorl.emulator import logger
from prorl.common.controller.main_controller import MainController
from prorl.emulator.controller.tim_dataset_controller import TimDatasetController


CONTROLLERS: Dict[str, Controller.__class__] = {
    'tim-dataset': TimDatasetController
}
CONTROLLER_ALIASES: Dict[str, str] = {

}


def get_main_controller() -> MainController:
    main_controller = MainController(logger)
    for name, controller in CONTROLLERS.items():
        main_controller.add_controller(name, controller)
    for name, target in CONTROLLER_ALIASES.items():
        main_controller.add_alias(name, target)
    return main_controller
