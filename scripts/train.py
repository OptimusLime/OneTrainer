from onetrainer.modules.trainer.GenericTrainer import GenericTrainer
from onetrainer.modules.util.args.TrainArgs import TrainArgs
from onetrainer.modules.util.commands.TrainCommands import TrainCommands
from onetrainer.modules.util.callbacks.TrainCallbacks import TrainCallbacks
from onetrainer.modules.util.config.TrainConfig import TrainConfig
import json
import os
import sys

sys.path.append(os.getcwd())


def main():
    args = TrainArgs.parse_args()
    callbacks = TrainCallbacks()
    commands = TrainCommands()

    train_config = TrainConfig.default_values()
    with open(args.config_path, "r") as f:
        train_config.from_dict(json.load(f))

    trainer = GenericTrainer(train_config, callbacks, commands)

    trainer.start()

    canceled = False
    try:
        trainer.train()
    except KeyboardInterrupt:
        canceled = True

    if not canceled or train_config.backup_before_save:
        trainer.end()


if __name__ == '__main__':
    main()
