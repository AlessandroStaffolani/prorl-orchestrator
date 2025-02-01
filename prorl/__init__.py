import os

from dotenv import load_dotenv

from prorl.global_parser import global_parse_arguments
from prorl.common.filesystem import ROOT_DIR
from prorl.common.logger import get_logger
from prorl.run.config import SingleRunConfig, get_single_run_config
from prorl.run.multi_run_config import MultiRunConfig, get_multi_run_config
from prorl.emulator import init_emulator_module
from prorl.environment import init_environment_module

load_dotenv(dotenv_path=os.path.join(ROOT_DIR, '.env'))

if os.getenv('ENV') == 'test':
    single_run_config: SingleRunConfig = get_single_run_config(root_dir=ROOT_DIR,
                                                               config_path=None,
                                                               log_level=10)
    multi_run_config: MultiRunConfig = get_multi_run_config(root_dir=ROOT_DIR, config_path=None)
else:
    args = global_parse_arguments()
    single_run_config: SingleRunConfig = get_single_run_config(root_dir=ROOT_DIR,
                                                               config_path=args.config_path,
                                                               log_level=args.log_level)
    multi_run_config: MultiRunConfig = get_multi_run_config(root_dir=ROOT_DIR, config_path=args.multi_config_path)

logger = get_logger(single_run_config.logger)
init_emulator_module(single_run_config.emulator, logger)
init_environment_module(single_run_config.environment, logger)

