from onetrainer.modules.module.WDModel import WDModel
from onetrainer.modules.module.BlipModel import BlipModel
from onetrainer.modules.module.Blip2Model import Blip2Model
from onetrainer.modules.util.args.GenerateCaptionsArgs import GenerateCaptionsArgs
from onetrainer.modules.util.enum.GenerateCaptionsModel import GenerateCaptionsModel
import torch
import os
import sys

sys.path.append(os.getcwd())


def main():
    args = GenerateCaptionsArgs.parse_args()

    model = None
    if args.model == GenerateCaptionsModel.BLIP:
        model = BlipModel(torch.device(args.device), args.dtype.torch_dtype())
    elif args.model == GenerateCaptionsModel.BLIP2:
        model = Blip2Model(torch.device(args.device), args.dtype.torch_dtype())
    elif args.model == GenerateCaptionsModel.WD14_VIT_2:
        model = WDModel(torch.device(args.device), args.dtype.torch_dtype())

    model.caption_folder(
        sample_dir=args.sample_dir,
        initial_caption=args.initial_caption,
        mode=args.mode,
        error_callback=lambda filename: print(
            "Error while processing image " + filename),
        include_subdirectories=args.include_subdirectories
    )


if __name__ == "__main__":
    main()
