import json
import os
import traceback

import torch
from safetensors.torch import load_file
from torch import Tensor

from onetrainer.modules.model.StableDiffusionXLModel import StableDiffusionXLModel
from onetrainer.modules.modelLoader.BaseModelLoader import BaseModelLoader
from onetrainer.modules.modelLoader.StableDiffusionXLModelLoader import StableDiffusionXLModelLoader
from onetrainer.modules.modelLoader.mixin.ModelLoaderLoRAMixin import ModelLoaderLoRAMixin
from onetrainer.modules.modelLoader.mixin.ModelLoaderModelSpecMixin import ModelLoaderModelSpecMixin
from onetrainer.modules.util.ModelNames import ModelNames
from onetrainer.modules.util.ModelWeightDtypes import ModelWeightDtypes
from onetrainer.modules.util.TrainProgress import TrainProgress
from onetrainer.modules.util.enum.ModelType import ModelType


class StableDiffusionXLLoRAModelLoader(BaseModelLoader, ModelLoaderModelSpecMixin, ModelLoaderLoRAMixin):
    def __init__(self):
        super(StableDiffusionXLLoRAModelLoader, self).__init__()

    def __init_lora(
            self,
            model: StableDiffusionXLModel,
            state_dict: dict[str, Tensor],
    ):
        rank = self._get_lora_rank(state_dict)

        model.text_encoder_1_lora = self._load_lora_with_prefix(
            module=model.text_encoder_1,
            state_dict=state_dict,
            prefix="lora_te1",
            rank=rank,
        )

        model.text_encoder_2_lora = self._load_lora_with_prefix(
            module=model.text_encoder_2,
            state_dict=state_dict,
            prefix="lora_te2",
            rank=rank,
        )

        model.unet_lora = self._load_lora_with_prefix(
            module=model.unet,
            state_dict=state_dict,
            prefix="lora_unet",
            rank=rank,
            module_filter=["attentions"],
        )

    def _default_model_spec_name(
            self,
            model_type: ModelType,
    ) -> str | None:
        match model_type:
            case ModelType.STABLE_DIFFUSION_XL_10_BASE:
                return "resources/sd_model_spec/sd_xl_base_1.0-lora.json"
            case ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING:
                return "resources/sd_model_spec/sd_xl_base_1.0_inpainting-lora.json"
            case _:
                return None

    def __load_safetensors(
            self,
            model: StableDiffusionXLModel,
            lora_name: str,
    ):
        model.model_spec = self._load_default_model_spec(
            model.model_type, lora_name)

        state_dict = load_file(lora_name)
        self.__init_lora(model, state_dict)

    def __load_ckpt(
            self,
            model: StableDiffusionXLModel,
            lora_name: str,
    ):
        model.model_spec = self._load_default_model_spec(model.model_type)

        state_dict = torch.load(lora_name)
        self.__init_lora(model, state_dict)

    def __load_internal(
            self,
            model: StableDiffusionXLModel,
            lora_name: str,
    ):
        with open(os.path.join(lora_name, "meta.json"), "r") as meta_file:
            meta = json.load(meta_file)
            train_progress = TrainProgress(
                epoch=meta['train_progress']['epoch'],
                epoch_step=meta['train_progress']['epoch_step'],
                epoch_sample=meta['train_progress']['epoch_sample'],
                global_step=meta['train_progress']['global_step'],
            )

        # embedding model
        pt_lora_name = os.path.join(lora_name, "lora", "lora.pt")
        safetensors_lora_name = os.path.join(
            lora_name, "lora", "lora.safetensors")
        if os.path.exists(pt_lora_name):
            self.__load_ckpt(model, pt_lora_name)
        elif os.path.exists(safetensors_lora_name):
            self.__load_safetensors(model, safetensors_lora_name)
        else:
            raise Exception("no lora found")

        # optimizer
        try:
            model.optimizer_state_dict = torch.load(
                os.path.join(lora_name, "optimizer", "optimizer.pt"))
        except FileNotFoundError:
            pass

        # ema
        try:
            model.ema_state_dict = torch.load(
                os.path.join(lora_name, "ema", "ema.pt"))
        except FileNotFoundError:
            pass

        # meta
        model.train_progress = train_progress

        # model spec
        model.model_spec = self._load_default_model_spec(model.model_type)

    def load(
            self,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
    ) -> StableDiffusionXLModel | None:
        stacktraces = []

        base_model_loader = StableDiffusionXLModelLoader()

        if model_names.base_model is not None:
            model = base_model_loader.load(
                model_type, model_names, weight_dtypes)
        else:
            model = StableDiffusionXLModel(model_type=model_type)

        if model_names.lora:
            try:
                self.__load_internal(model, model_names.lora)
                return model
            except:
                stacktraces.append(traceback.format_exc())

            try:
                self.__load_ckpt(model, model_names.lora)
                return model
            except:
                stacktraces.append(traceback.format_exc())

            try:
                self.__load_safetensors(model, model_names.lora)
                return model
            except:
                stacktraces.append(traceback.format_exc())
        else:
            return model

        for stacktrace in stacktraces:
            print(stacktrace)
        raise Exception("could not load LoRA: " + model_names.lora)
