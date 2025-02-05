from onetrainer.modules.module.MaskByColor import MaskByColor
from onetrainer.modules.module.ClipSegModel import ClipSegModel
from onetrainer.modules.module.RembgModel import RembgModel
from onetrainer.modules.util.args.GenerateMasksArgs import GenerateMasksArgs
from onetrainer.modules.util.enum.GenerateMasksModel import GenerateMasksModel
import torch
import os
import sys

from onetrainer.modules.module.RembgHumanModel import RembgHumanModel

sys.path.append(os.getcwd())


def main():
    args = GenerateMasksArgs.parse_args()

    model = None
    if args.model == GenerateMasksModel.CLIPSEG:
        model = ClipSegModel(torch.device(args.device),
                             args.dtype.torch_dtype())
    elif args.model == GenerateMasksModel.REMBG:
        model = RembgModel(torch.device(args.device), args.dtype.torch_dtype())
    elif args.model == GenerateMasksModel.REMBG_HUMAN:
        model = RembgHumanModel(torch.device(
            args.device), args.dtype.torch_dtype())
    elif args.model == GenerateMasksModel.COLOR:
        model = MaskByColor(torch.device(args.device),
                            args.dtype.torch_dtype())

    model.mask_folder(
        sample_dir=args.sample_dir,
        prompts=args.prompts,
        mode=args.mode,
        threshold=args.threshold,
        smooth_pixels=args.smooth_pixels,
        expand_pixels=args.expand_pixels,
        alpha=args.alpha,
        error_callback=lambda filename: print(
            "Error while processing image " + filename),
        include_subdirectories=args.include_subdirectories
    )


if __name__ == "__main__":
    main()
