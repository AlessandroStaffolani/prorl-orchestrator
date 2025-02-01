import sys
import os

from prorl.common.data_structure import RunMode

if sys.version_info >= (3, 8):
    from typing import TypedDict, Optional, List, Tuple, Union, Dict
else:
    from typing import Optional, List, Tuple, Union, Dict
    from typing_extensions import TypedDict
from datetime import datetime

import pandas as pd

from pymongo import DESCENDING, ASCENDING
from bson.objectid import ObjectId

from prorl import SingleRunConfig
from prorl.common.config import ExportMode
from prorl.common.object_handler import SaverMode
from prorl.common.remote.mongo import MongoWrapper
from prorl.run import RunStatus
from prorl.run.multi_run_config import AGENT_WITH_VALIDATION
from prorl.run.config import MultiRunParamConfig


class PersistenceInfo(TypedDict):
    path: str
    save_mode: SaverMode
    host: str


class MongoRunWrapper(MongoWrapper):

    def __init__(
            self,
            run_db=os.getenv('MONGO_DB'),
            scheduled_runs_collection='scheduled_runs',
            scheduled_val_runs_collection='validation_runs',
            evaluation_runs_collection='evaluation_runs',
            completed_runs_collection='completed_runs',
            multi_run_db='lb_multi_runs',
            host: Optional[str] = None,
            port: Optional[int] = None,
            user: Optional[str] = None,
            db: Optional[str] = None,
            password: Optional[str] = None,
            use_tunnelling: bool = False
    ):
        super(MongoRunWrapper, self).__init__(host, port, user, db, password, use_tunnelling)
        self.run_db = run_db
        self.scheduled_runs_collection = scheduled_runs_collection
        self.scheduled_val_runs_collection = scheduled_val_runs_collection
        self.evaluation_runs_collection = evaluation_runs_collection
        self.completed_runs_collection = completed_runs_collection
        self.mutl_run_db = multi_run_db
        self.single_run_db = self.client[self.run_db]
        self.multi_run_db = self.client[self.mutl_run_db]

    def add_scheduled_run(
            self,
            run_code: str,
            run_config: SingleRunConfig,
            status: RunStatus = RunStatus.SCHEDULED,
            result_path: Optional[PersistenceInfo] = None,
            validation_runs: Optional[List[ObjectId]] = None,
            total_steps: Optional[int] = None
    ):
        now = datetime.utcnow()
        doc = {
            'run_code': run_code,
            'config': run_config.export(mode=ExportMode.DICT),
            'status': status,
            'result_path': result_path,
            'total_steps': total_steps,
            'validation_runs': [] if validation_runs is None else validation_runs,
            'created_at': now,
            'last_modified': now,
        }
        if status == RunStatus.RUNNING:
            doc['started_at'] = now
        return self.save(
            collection=self.scheduled_runs_collection,
            document=doc,
            db=self.run_db
        )

    def update_scheduled_run(
            self,
            run_code: str,
            status: Optional[RunStatus] = None,
            result_path: Optional[PersistenceInfo] = None,
            validation_runs: Optional[List[ObjectId]] = None,
            total_steps: Optional[int] = None,
            run_stats: Optional[Dict[str, Union[int, float]]] = None,
            run_performance: Optional[int] = None,
            data_folder: Optional[str] = None,
            best_validation_performance: Optional[float] = None,
            best_validation_performance_iteration: Optional[int] = None,
    ):
        update_obj = {}
        now = datetime.utcnow()
        if status is not None:
            update_obj['status'] = status
        if result_path is not None:
            update_obj['result_path'] = result_path
        if data_folder is not None:
            update_obj['data_folder'] = data_folder
        if total_steps is not None:
            update_obj['total_steps'] = total_steps
        if run_performance is not None:
            update_obj['run_performance'] = run_performance
        if run_stats is not None:
            for key, val in run_stats.items():
                update_obj[key] = val
        if validation_runs is not None and isinstance(validation_runs, list):
            update_obj['validation_runs'] = validation_runs
        if len(update_obj) > 0:
            update_obj['last_modified'] = now
            if status is not None and status == RunStatus.RUNNING:
                update_obj['started_at'] = now
        if best_validation_performance is not None:
            update_obj[f'{RunMode.Validation.value}/best_performance'] = best_validation_performance
        if best_validation_performance_iteration is not None:
            update_obj[f'{RunMode.Validation.value}/best_performance_iteration'] = best_validation_performance_iteration
        return self.update(
            collection=self.scheduled_runs_collection,
            query={'run_code': run_code},
            update_obj={'$set': update_obj},
            db=self.run_db
        )

    def append_validation_run_id(self, run_code: str, val_run_id: ObjectId):
        return self.update(
            collection=self.scheduled_runs_collection,
            query={'run_code': run_code},
            update_obj={
                '$push': {
                    'validation_runs': {'$ref': self.scheduled_val_runs_collection, '$id': val_run_id}
                }
            },
            db=self.run_db
        )

    def append_evaluation_run_id(self, run_code: str, val_run_id: ObjectId):
        return self.update(
            collection=self.scheduled_runs_collection,
            query={'run_code': run_code},
            update_obj={
                '$push': {
                    'evaluation_runs': {'$ref': self.evaluation_runs_collection, '$id': val_run_id}
                }
            },
            db=self.run_db
        )

    def add_scheduled_validation_run(
            self,
            run_code: str,
            val_code: str,
            val_config: SingleRunConfig,
            agent_state: bytes,
            save_folder_path: PersistenceInfo,
            training_current_step: int,
            status: RunStatus = RunStatus.SCHEDULED,
            total_steps: Optional[int] = None,
            best_index: Optional[int] = None
    ):
        now = datetime.utcnow()
        doc = {
            'run_code': run_code,
            'validation_run_code': val_code,
            'config': val_config.export(mode=ExportMode.DICT),
            'agent_state': agent_state,
            'save_folder_path': save_folder_path,
            'training_current_step': training_current_step,
            'status': status,
            'total_steps': total_steps,
            'best_index': best_index,
            'created_at': now,
            'last_modified': now,
        }
        if status == RunStatus.RUNNING:
            doc['started_at'] = now
        insert_result = self.save(
            collection=self.scheduled_val_runs_collection,
            document=doc,
            db=self.run_db
        )
        self.append_validation_run_id(run_code, insert_result.inserted_id)
        return insert_result

    def add_validation_run(self,
                           run_code: str,
                           iteration: int,
                           status: RunStatus,
                           validation_data: dict,
                           validation_steps: int
                           ):
        doc = {
            'run_code': run_code,
            'validation_run_code': f'{run_code}_{iteration}',
            'iteration': iteration,
            'status': status,
            'total_steps': validation_steps,
        }
        for key, value in validation_data.items():
            doc[key] = value
        insert_result = self.save(
            collection=self.scheduled_val_runs_collection,
            document=doc,
            db=self.run_db
        )
        self.append_validation_run_id(run_code, insert_result.inserted_id)
        return insert_result

    def add_evaluation_run(self,
                           run_code: str,
                           iteration: int,
                           status: RunStatus,
                           evaluation_data: dict,
                           steps: int
                           ):
        doc = {
            'run_code': run_code,
            'iteration': iteration,
            'status': status,
            'total_steps': steps,
        }
        for key, value in evaluation_data.items():
            doc[key] = value
        insert_result = self.save(
            collection=self.evaluation_runs_collection,
            document=doc,
            db=self.run_db
        )
        self.append_evaluation_run_id(run_code, insert_result.inserted_id)
        return insert_result

    def update_scheduled_validation_run(
            self,
            run_code: str,
            val_code: str,
            status: RunStatus = RunStatus.SCHEDULED,
            total_steps: Optional[int] = None,
            best_index: Optional[int] = None,
            delete_agent_state: bool = False,
            run_stats: Optional[Dict[str, Union[int, float]]] = None,
    ):
        update_obj = {}
        now = datetime.utcnow()
        if status is not None:
            update_obj['status'] = status
        if total_steps is not None:
            update_obj['total_steps'] = total_steps
        if best_index is not None:
            update_obj['best_index'] = best_index
        if run_stats is not None:
            for key, val in run_stats.items():
                update_obj[f'metric/{key}'] = val
        if delete_agent_state:
            update_obj['agent_state'] = None
        if len(update_obj) > 0:
            update_obj['last_modified'] = now
            if status is not None and status == RunStatus.RUNNING:
                update_obj['started_at'] = now
        return self.update(
            collection=self.scheduled_val_runs_collection,
            query={'run_code': run_code, 'validation_run_code': val_code},
            update_obj={'$set': update_obj},
            db=self.run_db
        )

    def get_all_by_status(self, collection: str, status: Optional[RunStatus] = None, populate=False):
        query = {}
        if status is not None:
            query = {'status': status}
        populate_field = None
        if populate:
            populate_field = {
                'field': 'validation_runs',
                'collection': self.scheduled_val_runs_collection
            }
        return self.get_many(
            collection=collection,
            query=query,
            populate=populate_field,
            db=self.run_db
        )

    def get_by_run_code(self, run_code: str, populate=False):
        populate_field = None
        if populate:
            populate_field = {
                'field': 'validation_runs',
                'collection': self.scheduled_val_runs_collection
            }
        return self.get_by_query(
            collection=self.scheduled_runs_collection,
            query={'run_code': run_code},
            populate=populate_field,
            db=self.run_db
        )

    def get_validation_run_by_code(self, val_code: str):
        return self.get_by_query(
            collection=self.scheduled_val_runs_collection,
            query={'validation_run_code': val_code},
            populate=None,
            db=self.run_db
        )

    def is_training_run_in_status(self, status: RunStatus, run_code: str, check_validation_runs: bool = False):
        train_run = self.get_by_run_code(run_code, populate=check_validation_runs)
        if RunStatus(train_run['status']) != status:
            return False
        if not check_validation_runs:
            return RunStatus(train_run['status']) == status
        for val_run in train_run['validation_runs']:
            if RunStatus(val_run['status']) != status:
                return False
        return True

    def is_training_run_completed(self, run_code: str, check_validation_runs: bool = False) -> bool:
        train_run = self.get_by_run_code(run_code, populate=check_validation_runs)
        if RunStatus(train_run['status']) != RunStatus.COMPLETED:
            return False
        if not check_validation_runs:
            return RunStatus(train_run['status']) == RunStatus.COMPLETED
        for val_run in train_run['validation_runs']:
            if RunStatus(val_run['status']) != RunStatus.COMPLETED:
                return False
        return True

    def is_run_saved_in_multi_run_table(self, run_code, multi_run_code):
        res = self.get_by_query(
            collection=multi_run_code,
            query={'run_code': run_code, 'multi_run_code': multi_run_code}
        )
        return res is not None

    def get_run_best_validation_runs(self, run_code: str, limit: int, metric: str,
                                     sort_order=DESCENDING,
                                     status: RunStatus = RunStatus.EXECUTED):
        query = {
            'run_code': run_code,
            'status': status,
        }
        sort = [(metric, sort_order)]
        self.set_db(self.run_db)
        results = self.db[self.scheduled_val_runs_collection].find(query).sort(sort).limit(limit)
        return [res for res in results]

    def save_multi_run_single_run(
            self,
            multi_run_code: str,
            run_code: str,
            run_params: List[MultiRunParamConfig],
    ):
        run_data = self.get_by_run_code(run_code, populate=True)
        best_val_run = None
        best_value = None
        # if 'validation_runs' in run_data and len(run_data['validation_runs']) > 0:
        #     for val_run in run_data['validation_runs']:
        #         if best_value is None or best_value < val_run[validation_best_metric]:
        #             best_value = val_run[validation_best_metric]
        #             best_val_run = val_run

        doc = {}
        for param in run_params:
            doc[f'hp/{param.key_short}'] = param.value
        if 'run_performance' in run_data:
            doc['run_performance'] = run_data['run_performance']
        for key, data in run_data.items():
            if key.find(f'{RunMode.Train.value}/') == 0 or key.find(f'{RunMode.Eval.value}/') == 0 \
                    or key.find(f'{RunMode.Validation.value}/') == 0:
                doc[key] = data
            elif key.find('metric/') == 0:
                doc[key.replace('metric/', f'{RunMode.Train.value}/')] = data
        # doc['training/total_steps'] = run_data['total_steps']
        # if best_val_run is not None:
        #     for key, data in best_val_run.items():
        #         if key.find('metric/') == 0:
        #             doc[key.replace('metric/', 'validation/')] = data
        #     doc['validation/total_steps'] = best_val_run['total_steps']
        doc['multi_run_code'] = multi_run_code
        doc['agent'] = run_data['config']['environment']['agent']['type']
        doc['run_code'] = run_code
        doc['created_at'] = datetime.utcnow()
        doc['last_modified'] = doc['created_at']
        return self.save(
            collection=multi_run_code,
            document=doc,
            db=self.run_db
        )

    def get_multi_run_data(self, multi_run_code):
        return self.get_many(
            collection=multi_run_code,
            query={},
            db=self.run_db
        )

    def is_multi_run_completed(self, multi_run_code) -> bool:
        query = {'config.multi_run.multi_run_code': multi_run_code}
        runs = self.get_many(
            collection=self.scheduled_runs_collection,
            query=query,
            db=self.run_db,
            projection={'_id': 0, 'run_code': 1, 'status': 1}
        )
        if len(runs) > 0:
            for run in runs:
                status = RunStatus(run['status'])
                if status == RunStatus.COMPLETED:
                    if not self.is_training_run_in_status(
                            status=RunStatus.COMPLETED, run_code=run['run_code'], check_validation_runs=True):
                        return False
                else:
                    return False
            return True
        else:
            return False

    def get_multi_run_val_runs(self, multi_run_code: str, status: Optional[RunStatus] = None):
        query = {'config.multi_run.multi_run_code': multi_run_code}
        if status is not None:
            query['status'] = status
        return self.get_many(
            collection=self.scheduled_val_runs_collection,
            query=query,
            db=self.run_db,
            projection={'run_code': 1, 'validation_run_code': 1, 'status': 1, '_id': 0}
        )

    def get_multi_run_best(self, multi_run_df: pd.DataFrame, agent: str, params: Tuple[str, ...], include_nan=False,
                           metric='validation_best_total_reward') -> List[dict]:
        if include_nan:
            agent_df = multi_run_df[(multi_run_df.agent.isna()) | (multi_run_df.agent == agent)]
        else:
            agent_df = multi_run_df[multi_run_df.agent == agent]
        best_combination = agent_df.groupby(list(params))[metric].mean().idxmax()
        best_value = agent_df.groupby(list(params))[metric].mean().max()
        if not isinstance(best_combination, tuple) and not isinstance(best_value, tuple):
            best_combination = (best_combination, )
            best_value = (best_value, )
        print(f'Best hyperparameter combination is {best_combination} with average value of {best_value}')
        best_run_codes = []
        best_val_runs: List[dict] = []
        for _, row in agent_df.iterrows():
            is_best_combination = True
            for i, param in enumerate(params):
                if row[param] != best_combination[i]:
                    is_best_combination = False
            if is_best_combination:
                best_run_codes.append(row.run_code)
                if agent in AGENT_WITH_VALIDATION:
                    best_val_runs.append(
                        self.get_run_best_validation_runs(row.run_code,
                                                          limit=1,
                                                          metric='best_index',
                                                          sort_order=ASCENDING,
                                                          status=RunStatus.COMPLETED)[0])
                else:
                    best_val_runs.append(self.get_by_run_code(row.run_code, populate=False))
        return best_val_runs
