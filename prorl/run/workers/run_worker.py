import json
import os
import socket
import time
from multiprocessing import Pool
from threading import Thread
from typing import Optional, List, Dict, Tuple

import redis

from prorl import logger
from prorl.common.enum_utils import ExtendedEnum
from prorl.common.remote.redis_wrapper import RedisQueue
from prorl.run.remote.redis_constants import REDIS_RUNS_QUEUE, REDIS_MANAGEMENT_KEY, \
    REDIS_MANAGEMENT_EVENT_KEY, REDIS_ALL_WORKERS_CODE, REDIS_ALL_VALIDATION_WORKERS_CODE, \
    REDIS_ALL_RUN_WORKERS_CODE, REDIS_VALIDATION_RUNS_QUEUE, FAILED_VALIDATION_RUNS_QUEUE, FAILED_RUNS_QUEUE, \
    REDIS_EVAL_RUNS_QUEUE, FAILED_EVAL_RUNS_QUEUE, REDIS_ALL_EVAL_WORKERS_CODE
from prorl.run.runner import Runner, ValRunRunner


class ManagementEventType(str, ExtendedEnum):
    RESET = 'RESET'
    STOP = 'STOP'
    STOP_EMPTY = 'STOP_EMPTY'


class WorkerStatus(str, ExtendedEnum):
    STARTING = 'STARTING'
    RUNNING = 'RUNNING'
    STOPPED = 'STOPPED'
    ERROR = 'ERROR'


class WorkerType(str, ExtendedEnum):
    RunWorker = 'RUN-WORKER'
    ValidationWorker = 'VALIDATION_WORKER'
    EvalWorker = 'EVAL_WORKER'


STOP_CONSTANT = '**STOP**'


def schedule_run(
        run_code: str,
        p_index: Optional[int] = None,
        validation_run_queue: Optional[str] = None,
        **kwargs
):
    runner = Runner(run_code, validation_run_queue=validation_run_queue)
    runner.run()
    return p_index


def schedule_val_run(
        run_code: str,
        p_index: Optional[int] = None,
        **kwargs
):
    runner = ValRunRunner(val_run_code=run_code)
    runner.run()
    return p_index


TYPE_WORKER_MAPPING = {
    WorkerType.RunWorker: {
        'queue': REDIS_RUNS_QUEUE,
        'worker_type_name': REDIS_ALL_RUN_WORKERS_CODE,
        'schedule_func': schedule_run,
        'run_type': 'training-run',
        'failed_queue': FAILED_RUNS_QUEUE
    },
    WorkerType.ValidationWorker: {
        'queue': REDIS_VALIDATION_RUNS_QUEUE,
        'worker_type_name': REDIS_ALL_VALIDATION_WORKERS_CODE,
        'schedule_func': schedule_val_run,
        'run_type': 'validation-run',
        'failed_queue': FAILED_VALIDATION_RUNS_QUEUE
    },
    WorkerType.EvalWorker: {
        'queue': REDIS_EVAL_RUNS_QUEUE,
        'worker_type_name': REDIS_ALL_EVAL_WORKERS_CODE,
        'schedule_func': schedule_run,
        'run_type': 'eval-run',
        'failed_queue': FAILED_EVAL_RUNS_QUEUE
    }
}


class Worker:

    def __init__(
            self,
            processes: int,
            worker_type: WorkerType,
            queue_name: Optional[str] = None,
            validation_queue_name: Optional[str] = None,
            stop_empty: bool = False
    ):
        redis_host = os.getenv('REDIS_HOST')
        redis_port = int(os.getenv('REDIS_PORT'))
        self.redis_queue: RedisQueue = RedisQueue(
            host=redis_host,
            port=redis_port,
            db=int(os.getenv('REDIS_DB')),
            password=os.getenv('REDIS_PASSWORD'),
        )
        self.redis: redis.Redis = self.redis_queue.redis_connection
        self.processes = processes
        self.worker_type: WorkerType = worker_type
        self.worker_code: str = self.get_worker_code(index=0)
        self.uses_sub_processes: bool = True if processes > 0 else False
        self.pubsub: redis.client.PubSub = self.redis.pubsub()
        self.management_thread: Optional[Thread] = None
        self.run_queue_name: str = TYPE_WORKER_MAPPING[self.worker_type]['queue']
        self.failed_queue_name: str = TYPE_WORKER_MAPPING[self.worker_type]['failed_queue']
        self.validation_queue_name: str = TYPE_WORKER_MAPPING[WorkerType.ValidationWorker]['queue']
        if queue_name is not None:
            if self.worker_type == WorkerType.ValidationWorker:
                self.run_queue_name = f'{queue_name}-validation'
            else:
                self.run_queue_name = queue_name
            self.failed_queue_name = f'{queue_name}-failed'
            self.validation_queue_name = f'{queue_name}-validation'
        if validation_queue_name is not None:
            self.validation_queue_name = validation_queue_name
        self.stop_empty: bool = stop_empty
        if self.worker_type == WorkerType.ValidationWorker:
            self.stop_empty: bool = False

        self.processes_running: List[bool] = []
        self.pool: Optional[Pool] = None
        self.status: WorkerStatus = WorkerStatus.STARTING
        self._set_status(WorkerStatus.STARTING)
        self.stop = False
        self._init()

    def get_worker_code(self, index: int = 0) -> str:
        worker_ip = socket.gethostbyname(socket.gethostname())
        worker_code = f'{self.worker_type.value}-{worker_ip}-{index}'
        all_workers = self.redis.hgetall(REDIS_MANAGEMENT_KEY)
        if worker_code.encode('utf-8') not in all_workers:
            return worker_code
        else:
            status = all_workers[worker_code.encode('utf-8')].decode('utf-8')
            if WorkerStatus(status) == WorkerStatus.STOPPED or WorkerStatus(status) == WorkerStatus.ERROR:
                return worker_code
            else:
                return self.get_worker_code(index + 1)

    def _init(self):
        if self.uses_sub_processes:
            self.pool: Pool = Pool(processes=self.processes)
            self.processes_running: List[bool] = [False] * self.processes

    def _set_status(self, status: WorkerStatus):
        self.status = status
        self.redis.hset(
            key=self.worker_code,
            name=REDIS_MANAGEMENT_KEY,
            value=status
        )

    def _handle_run_completed(self, result):
        if self.uses_sub_processes:
            self.processes_running[result] = False
            logger.info(f'Process {result} is now free')

    def _handle_management_messages(self, message):
        try:
            if message['type'] == 'message':
                event_data = json.loads(message['data'])
                event_type = ManagementEventType(event_data['type'])
                worker_code = event_data['worker_code']
                worker_type_name = TYPE_WORKER_MAPPING[self.worker_type]['worker_type_name']
                if worker_code == self.worker_code or worker_code == REDIS_ALL_WORKERS_CODE \
                        or worker_code == worker_type_name:
                    logger.info(f'Received management event of type: {event_type.value}')
                    if event_type == ManagementEventType.STOP:
                        self.stop = True
                        if self.status == WorkerStatus.ERROR:
                            self.stop_worker()
                    elif event_type == ManagementEventType.STOP_EMPTY \
                            and self.worker_type == WorkerType.ValidationWorker:
                        self.stop_empty = True
                        queue_elements = self.redis.lrange(self.run_queue_name, 0, -1)
                        if len(queue_elements) == 0:
                            self.stop = True
                            self.stop_worker()
                    else:
                        logger.warning('No callback for event of type {}'.format(event_type.value))
            if message['type'] == 'subscribe':
                logger.info('{} correctly subscribed to {}'.format(self.worker_code, REDIS_MANAGEMENT_EVENT_KEY))
        except Exception as e:
            logger.warning(e)

    def _get_next_run_code(self):
        run_code = self.redis_queue.redis_connection.lpop(self.run_queue_name)
        if run_code is not None and isinstance(run_code, bytes):
            run_code_decoded = run_code.decode('utf-8')
            return run_code_decoded
        else:
            if self.stop_empty:
                self.stop = True
                self.stop_worker()
                if self.worker_type == WorkerType.RunWorker:
                    self.redis.publish(channel=REDIS_MANAGEMENT_EVENT_KEY, message=json.dumps({
                        'type': ManagementEventType.STOP_EMPTY,
                        'worker_code': TYPE_WORKER_MAPPING[WorkerType.ValidationWorker]['worker_type_name']
                    }))
            return None

    def _wait_for_next_run_code(self):
        try:
            change_message = self.redis_queue.get(key=self.run_queue_name, timeout=0.01, is_keyspace=True)
            change_message = change_message.decode('utf-8')
            if isinstance(change_message, bytes) and (change_message == 'rpush' or change_message == 'lpush'):
                return self._get_next_run_code()
            return None
        except TimeoutError:
            return None

    def _get_next_available_process(self) -> Optional[int]:
        for i, is_running in enumerate(self.processes_running):
            if not is_running:
                return i
        return None

    def get_next_run_code(self) -> Tuple[Optional[str], Optional[int]]:
        run_code = None
        process_index: Optional[int] = None
        while run_code is None and self.stop is False:
            if self.uses_sub_processes:
                while process_index is None and self.stop is False:
                    process_index = self._get_next_available_process()
                    if process_index is None:
                        time.sleep(1)
            run_code = self._get_next_run_code()
            if run_code is None:
                run_code = self._wait_for_next_run_code()
                if run_code is None:
                    time.sleep(1)
        return run_code, process_index

    def start_run(self, run_code: Optional[str] = None, process_index: Optional[int] = None):
        if run_code is not None:
            func = TYPE_WORKER_MAPPING[self.worker_type]['schedule_func']
            run_type = TYPE_WORKER_MAPPING[self.worker_type]['run_type']
            if self.uses_sub_processes:
                if process_index is not None:
                    self.processes_running[process_index] = True
                    self.pool.apply_async(
                        func=func,
                        kwds={
                            'run_code': run_code,
                            'p_index': process_index,
                            'validation_run_queue': self.validation_queue_name
                        },
                        error_callback=lambda error: logger.exception(error),
                        callback=self._handle_run_completed
                    )
                    logger.info(f'Scheduled new {run_type} with code {run_code}')
            else:
                logger.info(f'Started new {run_type} with code {run_code}')
                try:
                    func(run_code=run_code, p_index=process_index,
                         validation_run_queue=self.validation_queue_name)
                except Exception as e:
                    self._push_failed_run(run_code)
                    logger.exception(e)

    def _push_failed_run(self, run_code: str):
        self.redis.rpush(self.failed_queue_name, run_code)
        logger.error(f'Run with code {run_code} failed the execution')

    def start(self):
        self.pubsub.subscribe(**{REDIS_MANAGEMENT_EVENT_KEY: self._handle_management_messages})
        self.management_thread = self.pubsub.run_in_thread(sleep_time=0.001)
        self._set_status(WorkerStatus.RUNNING)
        message = f'Started run worker {self.worker_code} with {self.processes} processes ' \
                  f'listening on queue "{self.run_queue_name}"'
        if self.worker_type == WorkerType.RunWorker:
            message += f' and using as validation queue "{self.validation_queue_name}"'
        logger.info(message)
        while not self.stop:
            try:
                if self.status != WorkerStatus.ERROR:
                    run_code: str
                    process_index: Optional[int]
                    run_code, process_index = self.get_next_run_code()
                    self.start_run(run_code, process_index)
            except Exception as e:
                logger.exception(e)
                self._set_status(WorkerStatus.ERROR)
            finally:
                time.sleep(0.05)
        self.stop_worker()

    def stop_worker(self):
        if self.uses_sub_processes:
            self.pool.close()
            logger.info('Requested stop, waiting for the running runs')
            self.pool.join()
            logger.info('All the runs started have terminated their execution')
        self._set_status(WorkerStatus.STOPPED)
        if self.management_thread is not None:
            self.management_thread.stop()
            time.sleep(0.01)
        if self.pubsub is not None:
            self.pubsub.close()
            time.sleep(0.01)
        self.redis_queue.close()
        self.redis.close()
        logger.info(f'Worker {self.worker_code} stopped')
