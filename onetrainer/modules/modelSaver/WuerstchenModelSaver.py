import copy
import json
import os.path
from pathlib import Path

import torch
from safetensors.torch import save_file

from onetrainer.modules.model.BaseModel import BaseModel
from onetrainer.modules.model.WuerstchenModel import WuerstchenModel
from onetrainer.modules.modelSaver.BaseModelSaver import BaseModelSaver
from onetrainer.modules.util.convert.convert_stable_cascade_diffusers_to_ckpt import convert_stable_cascade_diffusers_to_ckpt
from onetrainer.modules.util.enum.ModelFormat import ModelFormat
from onetrainer.modules.util.enum.ModelType import ModelType


class WuerstchenModelSaver(BaseModelSaver):

    def __save_diffusers(
            self,
            model: WuerstchenModel,
            destination: str,
            dtype: torch.dtype,
    ):
        # Copy the model to cpu by first moving the original model to cpu. This preserves some VRAM.
        pipeline = model.create_pipeline().prior_pipe
        original_device = pipeline.device
        pipeline.to("cpu")
        pipeline_copy = copy.deepcopy(pipeline)
        pipeline.to(original_device)

        pipeline_copy.to("cpu", dtype, silence_dtype_warnings=True)

        os.makedirs(Path(destination).absolute(), exist_ok=True)
        pipeline_copy.save_pretrained(destination)

        del pipeline_copy

    def __save_internal(
            self,
            model: WuerstchenModel,
            destination: str,
    ):
        # base model
        self.__save_diffusers(model, destination, torch.float32)

        # optimizer
        os.makedirs(os.path.join(destination, "optimizer"), exist_ok=True)
        torch.save(model.optimizer.state_dict(), os.path.join(
            destination, "optimizer", "optimizer.pt"))

        # ema
        if model.ema:
            os.makedirs(os.path.join(destination, "ema"), exist_ok=True)
            torch.save(model.ema.state_dict(), os.path.join(
                destination, "ema", "ema.pt"))

        # meta
        with open(os.path.join(destination, "meta.json"), "w") as meta_file:
            json.dump({
                'train_progress': {
                    'epoch': model.train_progress.epoch,
                    'epoch_step': model.train_progress.epoch_step,
                    'epoch_sample': model.train_progress.epoch_sample,
                    'global_step': model.train_progress.global_step,
                },
            }, meta_file)

        # model spec
        with open(os.path.join(destination, "model_spec.json"), "w") as model_spec_file:
            json.dump(self._create_safetensors_header(model), model_spec_file)

    def __save_safetensors(
            self,
            model: WuerstchenModel,
            destination: str,
            dtype: torch.dtype,
    ):
        if model.model_type.is_stable_cascade():
            os.makedirs(Path(destination).absolute(), exist_ok=True)

            unet_state_dict = convert_stable_cascade_diffusers_to_ckpt(
                model.prior_prior.state_dict(),
            )
            unet_save_state_dict = self._convert_state_dict_dtype(
                unet_state_dict, dtype)
            self._convert_state_dict_to_contiguous(unet_save_state_dict)
            save_file(
                unet_save_state_dict,
                os.path.join(destination, "stage_c.safetensors"),
                self._create_safetensors_header(model, unet_save_state_dict)
            )

            te_state_dict = model.prior_text_encoder.state_dict()
            te_save_state_dict = self._convert_state_dict_dtype(
                te_state_dict, dtype)
            self._convert_state_dict_to_contiguous(te_save_state_dict)
            save_file(
                te_save_state_dict,
                os.path.join(destination, "text_encoder.safetensors"),
                self._create_safetensors_header(model, te_save_state_dict)
            )
        else:
            raise NotImplementedError

    def save(
            self,
            model: BaseModel,
            model_type: ModelType,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype,
    ):
        match output_model_format:
            case ModelFormat.DIFFUSERS:
                self.__save_diffusers(model, output_model_destination, dtype)
            case ModelFormat.CKPT:
                raise NotImplementedError
            case ModelFormat.SAFETENSORS:
                self.__save_safetensors(model, output_model_destination, dtype)
            case ModelFormat.INTERNAL:
                self.__save_internal(model, output_model_destination)
