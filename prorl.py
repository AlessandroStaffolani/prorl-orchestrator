from prorl import ROOT_DIR, logger
from prorl.global_parser import global_parse_arguments
from prorl.common.controller import get_global_controller
from prorl.common.controller.main_controller import MainController
from prorl.common.command_args import get_main_controller_args
from prorl.emulator.controller import get_main_controller as emulator_controller
from prorl.run.controller import get_main_controller as run_controller


MODULE_MAIN_CONTROLLER_MAPPING = {
    'emulator': emulator_controller,
    'run': run_controller,
    'global': get_global_controller,
}


if __name__ == '__main__':
    logger.info(f'Project root dir: {ROOT_DIR}')
    args = global_parse_arguments()
    module, controller, action, action_arguments = get_main_controller_args(args)
    current_controller: MainController = MODULE_MAIN_CONTROLLER_MAPPING[module]()
    current_controller.execute(controller, action, action_arguments)
    logger.close()
