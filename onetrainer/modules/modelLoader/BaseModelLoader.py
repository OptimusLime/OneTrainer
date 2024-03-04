from abc import ABCMeta, abstractmethod

from onetrainer.modules.model.BaseModel import BaseModel
from onetrainer.modules.util.ModelNames import ModelNames
from onetrainer.modules.util.ModelWeightDtypes import ModelWeightDtypes
from onetrainer.modules.util.enum.ModelType import ModelType


class BaseModelLoader(metaclass=ABCMeta):

    @abstractmethod
    def load(
            self,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
    ) -> BaseModel | None:
        pass
