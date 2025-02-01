from typing import Union, Optional, List
from uuid import uuid4

from prorl import single_run_config, MultiRunConfig, multi_run_config, SingleRunConfig
from prorl.common.controller import Controller
from prorl.common.object_handler import create_object_handler
from prorl.common.object_handler.base_handler import ObjectHandler
from prorl.common.object_handler.minio_handler import MinioObjectHandler
from prorl.environment import logger
from prorl.run.run_scheduler import RunScheduler
from prorl.run.runner import Runner, TestRunner


class RunnerController(Controller):

    def __init__(self):
        super(RunnerController, self).__init__('RunnerController')
        self._add_action('train', self.train)
        self._add_action('test-runs', self.test_runs)
        self.object_handler: Union[ObjectHandler, MinioObjectHandler] = create_object_handler(
            logger=logger,
            enabled=single_run_config.saver.enabled,
            mode=single_run_config.saver.mode,
            base_path=single_run_config.saver.get_base_path(),
            default_bucket=single_run_config.saver.default_bucket
        )

    def _single_train(self):
        run_code: str = str(uuid4())
        self.run_scheduler: RunScheduler = RunScheduler(use_ssh_tunnel=False)
        self.run_scheduler.schedule_run(run_code, config=single_run_config)
        runner = Runner(
            run_code=run_code
        )
        runner.run()

    def train(self, **kwargs):
        self._single_train()

    def test_runs(self, multi: bool = False, **kwargs):
        if multi:
            runs: List[SingleRunConfig] = multi_run_config.generate_runs_config()
            for run_config in runs:
                runner = TestRunner(run_code=str(uuid4()), config=run_config)
                runner.run()
        else:
            run_code: str = str(uuid4())
            runner = TestRunner(run_code=run_code, config=single_run_config)
            runner.run()
