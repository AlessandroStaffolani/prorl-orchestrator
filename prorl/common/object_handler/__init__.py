import logging
import os
from datetime import datetime
from typing import Union, Optional
from uuid import uuid4

from prorl.common.enum_utils import ExtendedEnum
from prorl.common.object_handler.base_handler import ObjectHandler
from prorl.common.object_handler.minio_handler import MinioObjectHandler


class SaverMode(str, ExtendedEnum):
    Disk = 'disk'
    Minio = 'minio'


def get_save_folder_from_config(
        save_name: str,
        save_name_with_uuid: bool,
        save_name_with_date: bool,
        save_prefix: str,
        uuid: str = None
):
    save_folder = save_name
    if save_name_with_uuid or uuid is not None:
        uuid_val = uuid if uuid is not None else str(uuid4())
        save_folder = f'{uuid_val}_{save_folder}'
    if len(save_prefix) > 0:
        save_folder = f'{save_prefix}_{save_folder}'
    if save_name_with_date:
        if save_folder.endswith('_'):
            save_folder += f'started_at={datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}'
        else:
            save_folder += f'_started_at={datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}'
    if len(save_folder) == 0:
        raise AttributeError('Save folder is empty, all the configuration are disabled')
    return save_folder


def create_object_handler(
        enabled: bool,
        mode: SaverMode,
        base_path: str,
        default_bucket: str,
        logger: logging.Logger,
        base=None,
        minio_endpoint: Optional[str] = None,
) -> Union[ObjectHandler, MinioObjectHandler]:
    if enabled:
        handler_type: SaverMode = mode
        if handler_type == SaverMode.Disk:
            base_path = base_path if base is None else base
            return ObjectHandler(logger=logger, base_path=base_path)
        elif handler_type == 'minio':
            default_bucket = default_bucket if base is None else base
            endpoint = minio_endpoint if minio_endpoint is not None else os.getenv('MINIO_ENDPOINT')
            access_key = os.getenv('MINIO_ACCESSKEY')
            secret_key = os.getenv('MINIO_SECRETKEY')
            secure = bool(os.getenv('MINIO_SECURE'))
            return MinioObjectHandler(
                endpoint=endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure,
                default_bucket=default_bucket,
                logger=logger,
            )
        else:
            raise AttributeError('Object handler requested not valid')
    else:
        logger.info('Saver disabled')
