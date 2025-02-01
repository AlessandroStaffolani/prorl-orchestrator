from typing import Optional

from prorl.common.controller import Controller
from prorl.run.workers.run_worker import WorkerType, Worker


class WorkerController(Controller):

    def __init__(self):
        super(WorkerController, self).__init__('WorkerController')
        self._add_action('run-worker', self.run_worker)
        self._add_action('val-run-worker', self.validation_run_worker)
        self._add_action('eval-run-worker', self.eval_run_worker)

    def run_worker(self, processes: int, use_ssh_tunnel: bool = False,
                   queue: Optional[str] = None, validation_queue: Optional[str] = None,
                   stop_empty: bool = False, **kwargs):
        worker = Worker(
            processes=processes,
            worker_type=WorkerType.RunWorker,
            use_tunnelling=use_ssh_tunnel,
            queue_name=queue,
            validation_queue_name=validation_queue,
            stop_empty=stop_empty
        )
        worker.start()

    def validation_run_worker(self, processes: int, use_ssh_tunnel: bool = False,
                              queue: Optional[str] = None, stop_empty: bool = False, **kwargs):
        worker = Worker(
            processes=processes,
            worker_type=WorkerType.ValidationWorker,
            use_tunnelling=use_ssh_tunnel,
            queue_name=queue,
            stop_empty=stop_empty
        )
        worker.start()

    def eval_run_worker(self, processes: int, use_ssh_tunnel: bool = False,
                        queue: Optional[str] = None, stop_empty: bool = False, **kwargs):
        worker = Worker(
            processes=processes,
            worker_type=WorkerType.EvalWorker,
            use_tunnelling=use_ssh_tunnel,
            queue_name=queue,
            stop_empty=stop_empty
        )
        worker.start()
