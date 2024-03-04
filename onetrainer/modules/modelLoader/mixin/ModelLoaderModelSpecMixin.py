import json
from abc import ABCMeta
import os

from safetensors import safe_open

from onetrainer.modules.util.enum.ModelType import ModelType
from onetrainer.modules.util.modelSpec.ModelSpec import ModelSpec


class ModelLoaderModelSpecMixin(metaclass=ABCMeta):
    def _default_model_spec_name(
            self,
            model_type: ModelType,
    ) -> str | None:
        return None

    def _load_default_model_spec(
            self,
            model_type: ModelType,
            safetensors_file_name: str | None = None,
    ) -> ModelSpec:
        model_spec = None

        model_spec_name = self._default_model_spec_name(model_type)
        if model_spec_name:
            if not os.path.exists(model_spec_name):
                model_spec_name = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "../../../", model_spec_name)
            with open(model_spec_name, "r", encoding="utf-8") as model_spec_file:
                model_spec = ModelSpec.from_dict(json.load(model_spec_file))
        else:
            model_spec = ModelSpec()

        if safetensors_file_name:
            try:
                with safe_open(safetensors_file_name, framework="pt") as f:
                    if "modelspec.sai_model_spec" in f.metadata():
                        model_spec = ModelSpec.from_dict(f.metadata())
            except:
                pass

        return model_spec
