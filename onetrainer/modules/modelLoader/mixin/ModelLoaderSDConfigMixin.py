import os
from abc import ABCMeta

import yaml

from onetrainer.modules.util.enum.ModelType import ModelType


def check_exist_with_dir(file_name: str, file_dir: str) -> str | None:
    if os.path.exists(file_name):
        return file_name
    elif os.path.exists(os.path.join(file_dir, file_name)):
        return os.path.join(file_dir, file_name)
    else:
        return None


class ModelLoaderSDConfigMixin(metaclass=ABCMeta):

    def _default_sd_config_name(
            self,
            model_type: ModelType,
    ) -> str | None:
        return None

    def _get_sd_config_name(
            self,
            model_type: ModelType,
            base_model_name: str | None = None,
    ) -> str | None:
        yaml_name = None
        file_dir = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "../../../")

        if base_model_name:
            new_yaml_name = check_exist_with_dir(
                os.path.splitext(base_model_name)[0] + '.yaml', file_dir)
            if new_yaml_name:
                yaml_name = new_yaml_name

            if not yaml_name:
                new_yaml_name = check_exist_with_dir(
                    os.path.splitext(base_model_name)[0] + '.yml', file_dir)
                if new_yaml_name:
                    yaml_name = new_yaml_name

        if not yaml_name:
            new_yaml_name = check_exist_with_dir(
                self._default_sd_config_name(model_type), file_dir)
            if new_yaml_name:
                yaml_name = new_yaml_name

        return yaml_name

    def _load_sd_config(
            self,
            model_type: ModelType,
            base_model_name: str | None = None,
    ) -> dict | None:
        yaml_name = self._get_sd_config_name(model_type, base_model_name)

        if yaml_name:
            with open(yaml_name, "r") as f:
                return yaml.safe_load(f)
        else:
            return None
