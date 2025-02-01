from prorl.common.config import ExportMode
from prorl import logger, single_run_config, multi_run_config
from prorl.common.controller.abstract_controller import Controller


class ConfigController(Controller):

    def __init__(self):
        super(ConfigController, self).__init__('ConfigController')
        self._add_action('export', self.export)

    def export(self, export_target: str, mode: str, export_multi_run_config: bool = False, **kwargs):
        try:
            export_mode = ExportMode(mode.upper())
        except Exception:
            raise AttributeError(f'{self.name}.export action: export mode "{mode}" not valid')
        if export_multi_run_config:
            multi_run_config.export(export_target, mode=export_mode)
            logger.info(f'MultiRunConfig exported with success on: {export_target}')
        else:
            single_run_config.export(export_target, mode=export_mode)
            logger.info(f'SingleRunConfig exported with success on: {export_target}')
