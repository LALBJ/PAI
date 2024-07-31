import argparse
import json
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from attention import llama_modify
from constants import INSTRUCTION_TEMPLATE, SYSTEM_MESSAGE
from eval_data_loader import COCODataSet
from llava.utils import disable_torch_init
from model_loader import ModelLoader
from tqdm import tqdm
from transformers.generation.logits_process import LogitsProcessorList


def setup_seeds():
    seed = 927

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


parser = argparse.ArgumentParser(description="CHAIR evaluation on LVLMs.")
parser.add_argument("--model", type=str, help="model")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)
# TODO
parser.add_argument(
    "--data-path",
    type=str,
    default="/path/to/coco/val2014/",
    help="data path",
)
parser.add_argument("--batch-size", type=int, default=1)

parser.add_argument("--beam", type=int, default=1)
parser.add_argument("--sample", action="store_true")
parser.add_argument("--use-attn", action="store_true")
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--use-mask", action="store_true")
parser.add_argument("--use-cfg", action="store_true")
parser.add_argument("--gamma", type=float, default=2)
parser.add_argument("--start-layer", type=int, default=2)
parser.add_argument("--end-layer", type=int, default=32)
parser.add_argument("--max-tokens", type=int, default=512)
args = parser.parse_known_args()[0]

setup_seeds()

disable_torch_init()

model_loader = ModelLoader(args.model)

base_dir = "./log/" + args.model
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

coco_dataset = COCODataSet(data_path=args.data_path, trans=model_loader.image_processor)
coco_loader = torch.utils.data.DataLoader(
    coco_dataset, batch_size=args.batch_size, shuffle=False, num_workers=32
)

file_parts = [
    f"chair_eval_layers_{args.start_layer}-{args.end_layer}_tokens_{args.max_tokens}_bs_{args.batch_size}",
    "_sample" if args.sample else "",
    f"_beams_{args.beam}" if args.beam != 1 else "",
    f"_attn_{args.alpha}" if args.use_attn else "",
    f"_cfg_{args.gamma}" if args.use_cfg else "",
]

file_name = "".join(file_parts)
template = INSTRUCTION_TEMPLATE[args.model]
if args.model == "llava-1.5" or args.model == "shikra":
    template = SYSTEM_MESSAGE + template

for batch_id, data in tqdm(enumerate(coco_loader), total=len(coco_loader)):
    if batch_id == 500:
        break
    img_id = data["img_id"]
    image = data["image"]

    batch_size = img_id.shape[0]
    query = ["Please help me describe the image in detail."] * batch_size
    questions, kwargs = model_loader.prepare_inputs_for_model(template, query, image)
    llama_modify(
        model_loader.llm_model,
        args.start_layer,
        args.end_layer,
        args.use_attn,
        args.alpha,
        args.use_cfg,
        model_loader.img_start_idx,
        model_loader.img_end_idx,
    )

    logits_processor = (
        model_loader.init_cfg_processor(questions, args.gamma, args.beam, args.start_layer, args.end_layer)
        if args.use_cfg
        else None
    )

    if logits_processor is not None:
        kwargs["logits_processor"] = LogitsProcessorList([logits_processor])

    with torch.inference_mode():
        outputs = model_loader.llm_model.generate(
            do_sample=args.sample,
            max_new_tokens=args.max_tokens,
            use_cache=True,
            num_beams=args.beam,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            **kwargs,
        )

    output_text = model_loader.decode(outputs)

    for i in range(len(output_text)):
        with open(os.path.join(base_dir, file_name + ".jsonl"), "a") as f:
            json.dump({"image_id": int(img_id[i]), "caption": output_text[i]}, f)
            f.write("\n")
