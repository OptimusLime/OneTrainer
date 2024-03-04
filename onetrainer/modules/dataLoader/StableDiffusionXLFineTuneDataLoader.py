import torch

from onetrainer.modules.dataLoader.StableDiffusionXLBaseDataLoader import StablDiffusionXLBaseDataLoader
from onetrainer.modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from onetrainer.modules.util.TrainProgress import TrainProgress
from onetrainer.modules.util.config.TrainConfig import TrainConfig


class StableDiffusionXLFineTuneDataLoader(StablDiffusionXLBaseDataLoader):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            config: TrainConfig,
            model: StableDiffusionXLModel,
            train_progress: TrainProgress,
    ):
        super(StableDiffusionXLFineTuneDataLoader, self).__init__(
            train_device,
            temp_device,
            config,
            model,
            train_progress,
        )
