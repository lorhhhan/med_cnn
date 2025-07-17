import argparse
import random
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# 注册模型、数据集等
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

def parse_args():
    parser = argparse.ArgumentParser(description="MiniGPT-4 Inference Script (No Gradio)")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the GPU to load the model.")
    parser.add_argument("--img-path", required=True, help="path to the input image.")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail as if to a medical student.", help="prompt to describe image.")
    args = parser.parse_args()
    return args

def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def main():
    args = parse_args()
    if not hasattr(args, "options"):
        args.options = None

    cfg = Config(args)
    seed = getattr(cfg.run_cfg, "seed", 42)
    setup_seeds(seed + get_rank())

    conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
                 'pretrain_llama2': CONV_VISION_LLama2}
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id

    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(f'cuda:{args.gpu_id}')
    CONV_VISION = conv_dict[model_config.model_type]

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    stop_words_ids = [[835], [2277, 29937]]
    stop_words_ids = [torch.tensor(ids).to(device=f'cuda:{args.gpu_id}') for ids in stop_words_ids]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    chat = Chat(model, vis_processor, device=f'cuda:{args.gpu_id}', stopping_criteria=stopping_criteria)

    raw_image = Image.open(args.img_path).convert("RGB")
    chat_state = CONV_VISION.copy()
    img_list = []
    chat.upload_img(raw_image, chat_state, img_list)
    chat.encode_img(img_list)

    user_prompt = args.prompt
    chat.ask(user_prompt, chat_state)

    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=1,
                              temperature=1.0,
                              max_new_tokens=300,
                              max_length=2000)[0]

    print("\n Prompt :")
    print(user_prompt)
    print("\n Description :")
    print(llm_message)

if __name__ == "__main__":
    main()
