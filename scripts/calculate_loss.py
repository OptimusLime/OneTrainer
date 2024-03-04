from onetrainer.modules.util.args.CalculateLossArgs import CalculateLossArgs
from onetrainer.modules.module.GenerateLossesModel import GenerateLossesModel
from onetrainer.modules.util.config.TrainConfig import TrainConfig
import json
import os
import sys

sys.path.append(os.getcwd())


def main():
    args = CalculateLossArgs.parse_args()

    train_config = TrainConfig.default_values()
    with open(args.config_path, "r") as f:
        train_config.from_dict(json.load(f))

    trainer = GenerateLossesModel(train_config, args.output_path)
    trainer.start()


if __name__ == '__main__':
    main()
