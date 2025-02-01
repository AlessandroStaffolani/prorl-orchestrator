import os
import time
from typing import List, Tuple, Optional, Dict
from uuid import uuid4

from prorl import SingleRunConfig, MultiRunConfig, logger
from prorl.common.print_utils import print_status
from prorl.common.remote.redis_wrapper import RedisQueue
from prorl.run.remote import MongoRunWrapper
from prorl.run.remote.redis_constants import REDIS_RUNS_QUEUE, REDIS_EVAL_RUNS_QUEUE


class RunScheduler:

    def __init__(self):
        mongo_host = None
        mongo_port = None
        redis_host = os.getenv('REDIS_HOST')
        redis_port = int(os.getenv('REDIS_PORT'))
        time.sleep(5)
        self.mongo: MongoRunWrapper = MongoRunWrapper(host=mongo_host, port=mongo_port)
        self.redis_queue: RedisQueue = RedisQueue(
            host=redis_host,
            port=redis_port,
            db=int(os.getenv('REDIS_DB')),
            password=os.getenv('REDIS_PASSWORD')
        )

    def schedule_run(self, run_code: str, config: SingleRunConfig, is_eval_run=False,
                     queue_name: Optional[str] = None):
        self.mongo.add_scheduled_run(run_code, run_config=config)
        queue = REDIS_RUNS_QUEUE if queue_name is None else queue_name
        if is_eval_run:
            queue = REDIS_EVAL_RUNS_QUEUE if queue_name is None else queue_name
        self.redis_queue.push(key=queue, value=run_code, allow_duplicates=False)

    def schedule_multi_runs(self, config: MultiRunConfig, is_eval_run=False,
                            queue_name: Optional[str] = None) -> Tuple[int, str]:
        runs: List[SingleRunConfig] = config.generate_runs_config()
        multi_run_code = runs[0].multi_run.multi_run_code
        total = len(runs)
        for i, run_config in enumerate(runs):
            run_code = str(uuid4())
            self.schedule_run(run_code, run_config, is_eval_run=is_eval_run, queue_name=queue_name)
            print_status(current=i+1, total=total, pre_message=f'Runs scheduled for multi run: {multi_run_code}',
                         loading_len=40)
        print()
        return len(runs), multi_run_code
