import torch

from onetrainer.modules.dataLoader.StableDiffusionBaseDataLoader import StablDiffusionBaseDataLoader
from onetrainer.modules.model.StableDiffusionModel import StableDiffusionModel
from onetrainer.modules.util.TrainProgress import TrainProgress
from onetrainer.modules.util.config.TrainConfig import TrainConfig


class StableDiffusionFineTuneDataLoader(StablDiffusionBaseDataLoader):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            config: TrainConfig,
            model: StableDiffusionModel,
            train_progress: TrainProgress,
    ):
        super(StableDiffusionFineTuneDataLoader, self).__init__(
            train_device,
            temp_device,
            config,
            model,
            train_progress,
        )
