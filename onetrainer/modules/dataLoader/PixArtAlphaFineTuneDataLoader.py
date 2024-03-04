import torch

from onetrainer.modules.dataLoader.PixArtAlphaBaseDataLoader import PixArtAlphaBaseDataLoader
from onetrainer.modules.model.StableDiffusionModel import StableDiffusionModel
from onetrainer.modules.util.TrainProgress import TrainProgress
from onetrainer.modules.util.config.TrainConfig import TrainConfig


class PixArtAlphaFineTuneDataLoader(PixArtAlphaBaseDataLoader):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            config: TrainConfig,
            model: StableDiffusionModel,
            train_progress: TrainProgress,
    ):
        super(PixArtAlphaFineTuneDataLoader, self).__init__(
            train_device,
            temp_device,
            config,
            model,
            train_progress,
        )
