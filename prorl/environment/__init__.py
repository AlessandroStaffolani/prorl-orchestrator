import logging
from typing import Optional

from prorl.common.filesystem import ROOT_DIR
from prorl.environment.config import EnvConfig

env_config: Optional[EnvConfig] = None

logger: Optional[logging.Logger] = None


def init_environment_module(conf: EnvConfig, log: logging.Logger):
    global env_config
    global logger
    env_config = conf
    logger = log

