import logging
from typing import Optional

from prorl.emulator.config import EmulatorConfig

emulator_config: Optional[EmulatorConfig] = None

logger: Optional[logging.Logger] = None


def init_emulator_module(conf: EmulatorConfig, log: logging.Logger):
    global emulator_config
    global logger
    emulator_config = conf
    logger = log
