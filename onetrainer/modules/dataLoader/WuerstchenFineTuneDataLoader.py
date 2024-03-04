import torch

from onetrainer.modules.dataLoader.WuerstchenBaseDataLoader import WuerstchenBaseDataLoader
from onetrainer.modules.model.WuerstchenModel import WuerstchenModel
from onetrainer.modules.util.TrainProgress import TrainProgress
from onetrainer.modules.util.config.TrainConfig import TrainConfig


class WuerstchenFineTuneDataLoader(WuerstchenBaseDataLoader):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            config: TrainConfig,
            model: WuerstchenModel,
            train_progress: TrainProgress,
    ):
        super(WuerstchenFineTuneDataLoader, self).__init__(
            train_device,
            temp_device,
            config,
            model,
            train_progress,
        )
