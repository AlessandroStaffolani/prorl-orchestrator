import json
import os
from enum import Enum
from typing import List, Dict, Any, Union

import yaml

from prorl.core.step import Step


class ConfigValueError(Exception):

    def __init__(self, key, value, module, extra_msg=None, *args):
        message = f'Invalid value for {module} config for key: {key} with value: {value}'
        if extra_msg is not None:
            message += f'. {extra_msg}'
        super(ConfigValueError, self).__init__(message, *args)


class ExportMode(str, Enum):
    JSON = 'JSON'
    YAML = 'YAML'
    DICT = 'DICT'


class AbstractConfig:

    def __init__(self, config_object_name=None, root_dir=None, **configs_to_override):
        self._name = config_object_name
        self.root_dir = root_dir
        self.export_exclude = ['export_exclude', '_name', 'root_dir']
        self._override_configs(**configs_to_override)

    def name(self):
        return self._name

    def _after_override_configs(self):
        pass

    def _override_configs(self, **override_config):
        for key, value in override_config.items():
            try:
                attr = getattr(self, key)
                if isinstance(attr, AbstractConfig):
                    attr.set_configs(**value)
                else:
                    setattr(self, key, value)
            except AttributeError:
                pass
        self._after_override_configs()

    def get_not_excluded_properties(self):
        return [a for a in dir(self) if
                not a.startswith('__') and not callable(getattr(self, a)) and a not in self.export_exclude]

    def set_configs(self, **config):
        self._override_configs(**config)

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, item):
        if hasattr(self, item) and item in self.get_not_excluded_properties():
            return True
        else:
            return False

    def __len__(self):
        return len(self.get_not_excluded_properties())

    def __str__(self):
        return str(self.export(mode=ExportMode.DICT))

    def export(self, save_path=None, mode: ExportMode = ExportMode.YAML) -> None or dict or str:
        config_to_export = export_to_dict(self)
        if mode == ExportMode.DICT:
            return config_to_export
        path = save_path
        if save_path is not None:
            if self.root_dir is not None:
                path = os.path.join(self.root_dir, save_path)
            path_split = path.split('/')
            folder_path = '/'.join(path_split[0:-1])
            os.makedirs(folder_path, exist_ok=True)
        if mode == ExportMode.YAML:
            if save_path is None:
                return yaml.dump(config_to_export)
            else:
                with open(path, 'w') as file:
                    yaml.safe_dump(config_to_export, file, sort_keys=False)
        if mode == ExportMode.JSON:
            if save_path is None:
                return json.dumps(config_to_export)
            else:
                with open(path, 'w') as file:
                    json.dump(config_to_export, file, sort_keys=False, indent=2)


class LoggerConfig(AbstractConfig):

    def __init__(self, config_object_name: str, log_folder: str = 'load_balancer', **configs_to_override):
        self.name = config_object_name
        self.level = 10
        self.handlers: List[Dict[str, Union[str, Dict[str, str]]]] = [
            {
                'type': 'console',
                'parameters': None
            },
            {
                'type': 'file',
                'parameters':
                    {
                        'log_folder': log_folder,
                        'log_basepath': 'logs'
                    }
            }
        ]
        super(LoggerConfig, self).__init__('LoggerConfig', **configs_to_override)

    def update_file_handler_folder(self, new_folder):
        file_handler = None
        for handler in self.handlers:
            if handler['type'] == 'file':
                file_handler = handler
        if file_handler is not None:
            file_handler['parameters']['log_folder'] = new_folder



def load_config_dict(config_path, root_dir):
    path = os.path.join(root_dir, config_path)
    if not os.path.exists(path):
        raise FileExistsError('config-path not exits.')
    with open(path, 'r') as f:
        extension = path.split('.')[-1].lower()
        if extension == 'json':
            config_dict = json.load(f)
        elif extension == 'yaml' or extension == 'yml':
            config_dict = yaml.safe_load(f.read())
        else:
            raise Exception(f'Extension "{extension}" of file: {path} is not valid')
    return config_dict


def set_sub_config(name, class_name, *args, **configs_to_override):
    if name in configs_to_override:
        sub_config = class_name(*args, **configs_to_override[name])
        del configs_to_override[name]
    else:
        sub_config = class_name(*args)
    return sub_config


SPECIAL_TYPES = [
    list, dict, AbstractConfig, Enum, Step
]


def is_a_special_type(o) -> bool:
    for special_type in SPECIAL_TYPES:
        if isinstance(o, special_type):
            return True
    return False


def export_to_dict(config: AbstractConfig) -> dict:
    dict_config = {}

    def export_single_prop(prop_value):
        if isinstance(prop_value, AbstractConfig):
            return export_to_dict(prop_value)
        elif isinstance(prop_value, Enum):
            return prop_value.value
        elif isinstance(prop_value, list):
            attr_values = []
            for value in prop_value:
                if is_a_special_type(value):
                    attr_values.append(export_single_prop(value))
                else:
                    attr_values.append(value)
            return attr_values
        elif isinstance(prop_value, dict):
            attr_dict = {}
            for sub_prop, sub_value in prop_value.items():
                if isinstance(sub_prop, Enum):
                    sub_prop = sub_prop.value
                if is_a_special_type(sub_value):
                    attr_dict[sub_prop] = export_single_prop(sub_value)
                else:
                    attr_dict[sub_prop] = sub_value
            return attr_dict
        elif isinstance(prop_value, Step):
            return prop_value.to_str()
        else:
            return prop_value

    for prop in config.get_not_excluded_properties():
        attr = getattr(config, prop)
        if isinstance(prop, Enum):
            dict_config[prop.value] = export_single_prop(attr)
        dict_config[prop] = export_single_prop(attr)
    return dict_config


