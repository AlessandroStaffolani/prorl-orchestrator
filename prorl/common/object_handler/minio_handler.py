import functools
import io
import logging
import os
import time
from typing import Any, Optional, Dict

import torch

from minio import Minio
from minio.commonconfig import CopySource
from minio.error import MinioException
from urllib3.exceptions import MaxRetryError

from prorl.common.object_handler import ObjectHandler
from prorl.common.object_handler.minio_progress import MinioProgress
from prorl.common.encoders import object_to_binary, binary_to_object


def get_kw_args(kwargs, key, default=None):
    if key in kwargs:
        return kwargs[key]
    else:
        return default


class MinioUnreachableException(Exception):

    def __init__(self, message, *args):
        super(MinioUnreachableException, self).__init__(message, *args)


def handle_minio_call(func):
    @functools.wraps(func)
    def wrapper_function(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except MaxRetryError:
            message = 'Minio unreachable at {}'.format(self.endpoint)
            self.logger.error(message)
            raise MinioUnreachableException(message)
        except MinioException:
            message = 'Minio failed to perform the requested operation using the provided access key and secret key'
            self.logger.error(message)
            raise MinioUnreachableException(message)
        except Exception as e:
            self.logger.exception(e)
            raise e

    return wrapper_function


class MinioObjectHandler(ObjectHandler):

    def __init__(self,
                 endpoint: str,
                 access_key: str,
                 secret_key: str,
                 secure: bool,
                 logger: logging.Logger,
                 default_bucket: str = 'data'):
        super(MinioObjectHandler, self).__init__(logger=logger, base_path=default_bucket)
        self.endpoint = endpoint
        self.client: Minio = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        self._create_bucket(bucket=self.base_path)

    def _concat_to_base_path(self, path, **kwargs):
        base_path = kwargs['base_path'] if 'base_path' in kwargs else None
        if base_path is None:
            concatenated_path = os.path.join(self.base_path, path)
        else:
            concatenated_path = os.path.join(base_path, path)
        return concatenated_path

    @handle_minio_call
    def save(self, obj, filename, path, pickle_encoding=False, **kwargs):
        content_type = get_kw_args(kwargs, 'content_type', 'application/octet-stream')
        use_progress = get_kw_args(kwargs, 'use_progress', False)
        bucket = self.base_path
        max_retries = get_kw_args(kwargs, 'max_retries', None)
        trial = get_kw_args(kwargs, 'trial', 0)
        wait_timeout = get_kw_args(kwargs, 'wait_timeout', 0)
        if 'bucket' in kwargs:
            bucket = kwargs['bucket']
            self._create_bucket(bucket=bucket)
        object_name = filename
        if path is not None and len(path) > 0:
            object_name = f'{path}/{filename}'
        progress = None
        if use_progress:
            progress = MinioProgress(logger=self.logger)
        obj_bytes = object_to_binary(obj, pickle_encoding=pickle_encoding)
        obj_stream = io.BytesIO(obj_bytes)
        try:
            result = self.client.put_object(
                bucket_name=bucket,
                object_name=object_name,
                data=obj_stream,
                length=len(obj_bytes),
                content_type=content_type,
                progress=progress,
                num_parallel_uploads=5
            )
            if progress is not None:
                progress.stop()
            self.logger.debug('Uploaded to minio object {} with result: {}'.format(filename, result))
            return True
        except Exception as e:
            if progress is not None:
                progress.stop()
            if max_retries is not None:
                # check if trial not exceed max_retries if it doesn't retry else raise the exception
                if trial < max_retries:
                    self.logger.warning('Error while uploading file to minio, current trial is {} '
                                        'max_retries is {}, so I\'m going to retry in {} seconds. Object name = {}'
                                        .format(trial, max_retries, wait_timeout, filename, object_name))
                    time.sleep(wait_timeout)
                    return self.save(obj=obj,
                                     filename=filename,
                                     path=path,
                                     pickle_encoding=pickle_encoding,
                                     bucket=bucket,
                                     trial=trial + 1,
                                     max_retries=max_retries,
                                     wait_timeout=wait_timeout,
                                     use_progress=use_progress,
                                     num_parallel_uploads=5
                                     )
                else:
                    self.logger.error('Impossible to upload for {} trials. Raising the error'.format(max_retries))
                    raise e
            else:
                raise e

    def save_to_disk(self, obj, filename, path, pickle_encoding=False, **kwargs):
        return super().save(obj, filename, path, pickle_encoding, **kwargs)

    @handle_minio_call
    def load(self, file_path, pickle_encoding=False, **kwargs):
        bucket = self.base_path
        if 'bucket' in kwargs:
            bucket = kwargs['bucket']
        keep_binaries = False
        if 'keep_binaries' in kwargs:
            keep_binaries = kwargs['keep_binaries']
        response = self.client.get_object(
            bucket_name=bucket,
            object_name=file_path
        )
        obj_bytes = response.data

        if not keep_binaries:
            obj = binary_to_object(obj_bytes, pickle_encoding=pickle_encoding)
        else:
            obj = obj_bytes

        response.close()
        response.release_conn()
        return obj

    def load_from_disk(self, file_path, pickle_encoding=False, **kwargs):
        return super().load(file_path, pickle_encoding, **kwargs)

    @handle_minio_call
    def save_agent_model(self, agent_model: Dict[str, Any], filename, path, **kwargs):
        bucket = self.base_path
        use_progress = kwargs['use_progress'] if 'use_progress' in kwargs else False
        content_type = kwargs['content_type'] if 'content_type' in kwargs else 'application/octet-stream'
        max_retries = get_kw_args(kwargs, 'max_retries', None)
        trial = get_kw_args(kwargs, 'trial', 0)
        wait_timeout = get_kw_args(kwargs, 'wait_timeout', 0)
        if 'bucket' in kwargs:
            bucket = kwargs['bucket']
        object_name = filename
        if path is not None and len(path) > 0:
            object_name = f'{path}/{filename}'
        progress = None
        if use_progress:
            progress = MinioProgress(logger=self.logger)
        buffer = io.BytesIO()
        torch.save(agent_model, buffer)
        obj_buffer = io.BytesIO(buffer.getvalue())
        buffer.close()
        try:
            result = self.client.put_object(
                bucket_name=bucket,
                object_name=object_name,
                data=obj_buffer,
                length=len(obj_buffer.getbuffer()),
                content_type=content_type,
                progress=progress,
                num_parallel_uploads=5
            )
            if progress is not None:
                progress.stop()
            self.logger.debug('Uploaded to minio object {} with result: {}'.format(filename, result))
            return True
        except Exception as e:
            if progress is not None:
                progress.stop()
            if max_retries is not None:
                # check if trial not exceed max_retries if it doesn't retry else raise the exception
                if trial < max_retries:
                    self.logger.warning('Error while uploading agent model to minio, current trial is {} '
                                        'max_retries is {}, so I\'m going to retry in {} seconds. Object name = {}'
                                        .format(trial, max_retries, wait_timeout, filename, object_name))
                    time.sleep(wait_timeout)
                    return self.save_agent_model(obj=agent_model,
                                                 filename=filename,
                                                 path=path,
                                                 bucket=bucket,
                                                 trial=trial + 1,
                                                 max_retries=max_retries,
                                                 wait_timeout=wait_timeout,
                                                 use_progress=use_progress,
                                                 num_parallel_uploads=5
                                                 )
                else:
                    self.logger.error('Impossible to upload agent model for {} trials. Raising the error'
                                      .format(max_retries))
                    raise e
            else:
                raise e

    @handle_minio_call
    def load_agent_model(self, file_path, map_location: Optional[torch.device] = None, **kwargs) -> Dict[str, Any]:
        bucket = self.base_path
        if 'bucket' in kwargs:
            bucket = kwargs['bucket']
        response = self.client.get_object(
            bucket_name=bucket,
            object_name=file_path
        )
        obj_bytes = response.data
        return torch.load(io.BytesIO(obj_bytes), map_location=map_location)

    @handle_minio_call
    def list_objects_name(self, path=None, recursive=True, **kwargs):
        bucket = self.base_path
        if 'bucket' in kwargs:
            bucket = kwargs['bucket']
        file_suffix = None if 'file_suffix' not in kwargs else kwargs['file_suffix']
        results = self.client.list_objects(bucket, prefix=path, recursive=recursive)
        names = []
        for obj in results:
            object_name = obj.object_name
            if file_suffix is not None:
                if file_suffix in object_name:
                    names.append(object_name)
            else:
                names.append(obj.object_name)
        return names

    @handle_minio_call
    def delete(self, file_path, **kwargs):
        bucket = self.base_path
        if 'bucket' in kwargs:
            bucket = kwargs['bucket']
        self.client.remove_object(
            bucket_name=bucket,
            object_name=file_path
        )
        return True

    @handle_minio_call
    def exists(self, file_path: str, bucket_name=None):
        bucket = bucket_name if bucket_name is not None else self.base_path
        try:
            if file_path.endswith('/'):
                names = self.list_objects_name(file_path, recursive=True)
                if len(names) == 0:
                    return False
                else:
                    return True
            else:
                _ = self.client.stat_object(bucket_name=bucket, object_name=file_path)
                return True
        except MinioException:
            return False

    @handle_minio_call
    def rename(self, source_path, dest_path, **kwargs):
        bucket = self.base_path
        if 'bucket' in kwargs:
            bucket = kwargs['bucket']
        dest_bucket = bucket
        if 'dest_bucket' in kwargs:
            dest_bucket = kwargs['dest_bucket']
        result = self.client.copy_object(
            bucket_name=dest_bucket,
            object_name=dest_path,
            source=CopySource(bucket, source_path)
        )
        if result.object_name == dest_path:
            # delete the old file
            self.delete(source_path)

    def remove_folder(self, folder_path, **kwargs):
        pass

    def build_path(self, path, **kwargs):
        return self._concat_to_base_path(path, **kwargs)

    def get_path(self, path, **kwargs):
        return path

    @handle_minio_call
    def _create_bucket(self, bucket):
        if self.client.bucket_exists(bucket_name=bucket) is False:
            self.client.make_bucket(bucket_name=bucket)

    @handle_minio_call
    def is_connected(self) -> bool:
        try:
            self._create_bucket(bucket=self.base_path)
            return True
        except Exception as e:
            return False

    def close(self):
        del self.client
