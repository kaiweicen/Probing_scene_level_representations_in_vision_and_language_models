# model.py

# instantiate the model
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import torch
import logging
import torch.nn as nn
import torch.distributed as dist
import yaml
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from volta.encoders import BertForVLPreTraining, BertForVLTasks
from volta.config import BertConfig
from utils import LoadDatasetEval

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

args ={
"from_pretrained" : "./checkpoints/Dr8geMQyRd",
"config_file" : "./config/ctrl_vl-bert_base.json",
"output_dir" : "results",
"save_name" : "",
"num_subiters" :2,
"zero_shot" : True,
"batch_size" : 30,
"drop_last" : False,
"seed" : 42,
"local_rank" : -1,
"num_workers" : 16,
"num_val_workers" : 10,
"in_memory" : False,
"use_chunk" : 0,
}

def main():
    # Devices
    if args["local_rank"] == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args["local_rank"])
        device = torch.device("cuda", args["local_rank"])
        n_gpu = 1
        torch.distributed.init_process_group(backend="nccl")
    default_gpu = False
    if dist.is_available() and args["local_rank"] != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True
    #logger.info(f"device: {device} n_gpu: {n_gpu}, distributed training: {bool(args["local_rank"] != -1)}")

    # Load config
    config = BertConfig.from_json_file(args["config_file"])

    # Output dirs
    if "/" in args["from_pretrained"]:
        timeStamp = args["from_pretrained"].split("/")[1]
    else:
        timeStamp = args["from_pretrained"]
    savePath = os.path.join(args["output_dir"], timeStamp)
    if default_gpu and not os.path.exists(savePath):
        os.makedirs(savePath)

    # Seed
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])

    # Dataset
    batch_size, task2num_iters, dset_val, dl_val = LoadDatasetEval(args, config)

    print("batch_size:")
    print(batch_size)
    print("tasknum_iters:")
    print(task2num_iters)
    print("dset_val:")
    print(dset_val)
    print("dl_val:")
    print(dl_val)

    # Model
    if args["zero_shot"]:
        config.visual_target_weights = {}  # [0, 0, 0, 0, 0, 0, 0]
        model = BertForVLPreTraining.from_pretrained(args["from_pretrained"], config=config)


    # Move to GPU(s)
    model.to(device)
    if args["local_rank"] != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model, deay_allreduce=True)
    elif n_gpu > 1:
        model = nn.DataParallel(model)
        raise ValueError("Please run with a single GPU")

    # Print summary
    if default_gpu:
        print("***** Running evaluation *****")
        print("  Num Iters: ", task2num_iters)
        print("  Batch size: ", batch_size)

    # Evaluate
    model.eval()
    results = []
    others = []
    #score_matrix = np.zeros((dset_val.num_entries, dset_val.num_images))
    #target_matrix = np.zeros((dset_val.num_entries, dset_val.num_images))
    #rank_vector = np.ones(dset_val.num_entries) * dset_val.num_images
    #count = 0
    for i, batch in tqdm(enumerate(dl_val), total=task2num_iters["Scenedataset"]):
        batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
        features, spatials, image_mask, question, input_mask, segment_ids, index = batch

        #features = features.squeeze(0)
        #spatials = spatials.squeeze(0)
        #image_mask = image_mask.squeeze(0)
        #question = question.repeat(features.size(0), 1)
        #segment_ids = segment_ids.repeat(features.size(0), 1)
        #input_mask = input_mask.repeat(features.size(0), 1)

        #target = target.view(-1).float().cpu().numpy()
        with torch.no_grad():
            if args["zero_shot"]:
                #_, _, vil_logit, _, _ = model(question, features, spatials, segment_ids, input_mask, image_mask, index)
                prediction_scores_t, prediction_scores_v_dict, seq_relationship_score, vqa_score, \
                   all_attention_mask, pooled_output = model(question, features, spatials, segment_ids, input_mask, image_mask)
                #print("prediction_scores_t")
                #print(prediction_scores_t)
                print("prediction_scores_v_dict")
                print(prediction_scores_v_dict)
                #print("seq_relationship_score")
                #print(seq_relationship_score)
                #print("vqa_score")
                #print(vqa_score)
                #print("all_attention_mask")
                #print(all_attention_mask)
                #print("pooled_output")
                #print(pooled_output)



if __name__ == "__main__":
    main()
