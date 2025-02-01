from typing import List

import numpy as np

from prorl import single_run_config
from prorl.common.data_structure import RunMode
from prorl.emulator import logger
from prorl.common.controller.abstract_controller import Controller
from prorl.emulator.models.tim_dataset_model import generate_tim_model_index, tim_model_index_on_mongo
from prorl.environment.data_structure import EnvResource
from prorl.run.remote import MongoRunWrapper


class TimDatasetController(Controller):

    def __init__(self):
        super(TimDatasetController, self).__init__('TimDatasetController')
        self._add_action('create-index', self.create_index)
        self._add_action('upload-index', self.upload_index)

    def create_index(self, output: str, **kwargs):
        logger.info(f'Generating indexes')
        resources_info = []
        for res_conf in single_run_config.environment.resources:
            resources_info.append(EnvResource(
                name=res_conf.name,
                bucket_size=res_conf.bucket_size,
                min_buckets=res_conf.min_resource_buckets_per_node,
                total_available=res_conf.total_available,
                allocated=res_conf.allocated,
                classes=res_conf.classes
            ))
        resource_names: List[str] = [res.name for res in resources_info]
        generate_tim_model_index(
            config=single_run_config,
            save_path=output,
            base_station_names=single_run_config.emulator.model.base_station_names(),
            resource_names=resource_names,
            em_config=single_run_config.emulator,
            random_state=np.random.RandomState(42),
            log=logger,
            run_mode=RunMode.Train,
            random_seed=42
        )
        logger.info(f'Indexes created with success and saved in "{output}"')

    def upload_index(self, source_path: str, collection_name: str, **kwargs):
        logger.info(f'Uploading Indexes on Mongo collection: {collection_name}')
        mongo = MongoRunWrapper()
        tim_model_index_on_mongo(
            source_path,
            mongo,
            collection_name
        )
        logger.info('Indexes uploaded with success')
