import json
import os
import pickle
import shutil
from glob import glob
from logging import Logger
from typing import Any, Dict, Optional

import yaml
import torch

from prorl.common.encoders import binary_to_object
from prorl.common.filesystem import save_file, get_absolute_path, ROOT_DIR,\
    create_directory, create_directory_from_filepath


class ObjectHandler:

    def __init__(self, logger: Logger, base_path=ROOT_DIR):
        self.base_path = base_path
        self.logger: Logger = logger

    def save(self, obj, filename, path, pickle_encoding=False, **kwargs):
        try:
            save_path = self._concat_to_base_path(path, **kwargs)
            save_file(path=save_path, filename=filename, content=obj)
            return True
        except Exception as e:
            self.logger.exception(e)
            raise e

    def load(self, file_path, pickle_encoding=False, **kwargs):
        try:
            load_path = self._concat_to_base_path(file_path, **kwargs)
            if os.path.exists(load_path):
                if load_path.endswith('.json'):
                    with open(load_path, 'r') as f:
                        obj = json.load(f)
                elif load_path.endswith('.yaml') or load_path.endswith('.yml'):
                    with open(load_path, 'r') as f:
                        obj = yaml.safe_load(f.read())
                elif load_path.endswith('.pth') or load_path.endswith('.pt'):
                    with open(load_path, 'rb') as f:
                        obj = pickle.load(f)
                else:
                    with open(load_path, 'rb') as f:
                        obj_bites = f.read()
                        obj = binary_to_object(obj_bites.decode('utf-8'))
                return obj
            else:
                raise AttributeError('file_path provided not exists in the host. file_path: "{}"'.format(file_path))
        except Exception as e:
            self.logger.exception(e)
            raise e

    def save_agent_model(self, agent_model: Dict[str, Any], filename, path, **kwargs):
        try:
            full_path = self.build_path(os.path.join(path, filename), **kwargs)
            torch.save(agent_model, full_path)
            return True
        except Exception as e:
            self.logger.exception(e)
            raise e

    def load_agent_model(self, file_path, map_location: Optional[torch.device] = None, **kwargs) -> Dict[str, Any]:
        try:
            load_path = self._concat_to_base_path(file_path, **kwargs)
            if os.path.exists(load_path):
                return torch.load(load_path, map_location=map_location)
            else:
                raise AttributeError('file_path provided not exists in the host. file_path: "{}"'.format(file_path))
        except Exception as e:
            self.logger.exception(e)
            raise e

    def list_objects_name(self, path=None, recursive=True, **kwargs):
        try:
            file_suffix = '' if 'file_suffix' not in kwargs else kwargs['file_suffix']
            search_path = self._concat_to_base_path(path, **kwargs)
            files = glob(f'{search_path}/*{file_suffix}')
            return files
        except Exception as e:
            self.logger.exception(e)
            raise e

    def exists(self, file_path, **kwargs):
        return os.path.exists(file_path)

    def rename(self, source_path, dest_path, **kwargs):
        full_source_path = self._concat_to_base_path(source_path)
        if self.exists(full_source_path):
            full_dest_path = self.build_path(dest_path)
            os.rename(full_source_path, full_dest_path)

    def delete(self, file_path, **kwargs):
        try:
            delete_path = self._concat_to_base_path(file_path, **kwargs)
            if os.path.exists(delete_path):
                os.remove(delete_path)
                self.logger.debug('File: "{}" removed'.format(delete_path))
                return True
        except Exception as e:
            self.logger.exception(e)
            raise e

    def remove_folder(self, folder_path, **kwargs):
        full_path = self._concat_to_base_path(folder_path, **kwargs)
        ignore_errors = False if 'ignore_errors' not in kwargs else kwargs['ignore_errors']
        if self.exists(full_path):
            shutil.rmtree(full_path, ignore_errors=ignore_errors)

    def build_path(self, path, **kwargs):
        full_path = self._concat_to_base_path(path, **kwargs)
        if not os.path.exists(full_path):
            if '.' in full_path.split('/')[-1]:
                create_directory_from_filepath(full_path)
            else:
                create_directory(full_path)
        return full_path

    def get_path(self, path, **kwargs):
        return self._concat_to_base_path(path, **kwargs)

    def _concat_to_base_path(self, path, **kwargs):
        base_path = kwargs['base_path'] if 'base_path' in kwargs else None
        if base_path is None:
            concatenated_path = get_absolute_path(path, self.base_path)
        else:
            concatenated_path = get_absolute_path(path, base_path)
        return concatenated_path

    def is_connected(self) -> bool:
        return True

    def close(self):
        pass
