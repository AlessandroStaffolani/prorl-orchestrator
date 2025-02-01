from logging import Logger
from typing import List, Dict

from torch.utils.tensorboard import SummaryWriter

from prorl.common.filesystem import get_absolute_path, create_directory


class TensorboardWrapper:

    def __init__(self, log_dir: str, logger: Logger, *args, **kwargs):
        self.logger: Logger = logger
        self.log_dir = get_absolute_path(log_dir)
        create_directory(self.log_dir)
        self.logger.debug('Tensorboard log dir: {}'.format(self.log_dir))
        self.writer = SummaryWriter(log_dir=self.log_dir, *args, **kwargs)
        self.average_reward: Dict[str, List[float]] = {
            '60': [],
            '3600': [],
            'add_60': [],
            'add_3600': [],
        }
        self.logger.info('Tensorboard initialized')

    def add_scalar(self, tag, value, step, *args, **kwargs):
        self.writer.add_scalar(tag, value, step, *args, **kwargs)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step=global_step, walltime=walltime)

    def add_graph(self, model, input_to_model=None, verbose=False):
        self.writer.add_graph(model, input_to_model, verbose)

    def add_hparams(self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None):
        self.writer.add_hparams(hparam_dict, metric_dict, hparam_domain_discrete, run_name)

    def close(self):
        self.writer.flush()
        self.writer.close()
        self.logger.info('Tensorboard writer has been closed')
