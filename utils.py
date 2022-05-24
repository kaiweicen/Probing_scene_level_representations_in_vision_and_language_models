# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from transformers import AutoTokenizer
import torch.distributed as dist
from torch.utils.data import DataLoader
import logging
logger = logging.getLogger(__name__)

def LoadDatasetEval(args, config):
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model, do_lower_case=config.do_lower_case)

    # initialize the feature reader
    # we don't need the ImageFeaturesH5Reader?
    #feats_h5path1 = args.val_features_lmdbpath or task_cfg[task]["features_h5path1"]
    #feats_h5path2 = task_cfg[task]["features_h5path2"]
    #features_reader1 = ImageFeaturesH5Reader(feats_h5path1, config, args.in_memory) if feats_h5path1 != "" else None
    #features_reader2 = ImageFeaturesH5Reader(feats_h5path2, config, args.in_memory) if feats_h5path2 != "" else None

    #!这里为什么有两个batch_size
    #batch_size = task_cfg[task].get("eval_batch_size", args.batch_size)

    #task_cfg[task].get("eval_batch_size") = 1024
    #args.batch_size = 30

    batch_size = 30

    if args["local_rank"] != -1:
        batch_size = int(batch_size / dist.get_world_size())

    #distributrd training

    logger.info("Loading %s Dataset with batch size %d" % ("Scene", batch_size))

    from scenedataset import SceneDataset

    dset_val = SceneDataset(
            dataroot= "./hl_dataset",
            #image_features_reader=features_reader1, #image_features_reader and gt_image_features_reader needs the H5reader
            #gt_image_features_reader=features_reader2,
            tokenizer=tokenizer,
            bert_model=config.bert_model,
            padding_index=tokenizer.pad_token_id,
            max_seq_length=38,
            max_region_num=36,
            num_locs=config.num_locs,
            #add_global_imgfeat=config.add_global_imgfeat, #"first"
            add_global_imgfeat="first",
            append_mask_sep=(config.fusion_method == 'vl-bert_vqa'), #all models in their control setting are "mul"
        )
        #retrieval has the paremeter num_subiters=args.num_subiters but other tasks don't have


    dl_val = DataLoader(
        dset_val,
        shuffle=False,
        batch_size=batch_size,
        num_workers=args["num_val_workers"], #how many subprocesses to use for data loading
        pin_memory=True, #increase the speed of transfering training data to GPUs
        drop_last=args["drop_last"],
    )


    task2num_iters = {"Scenedataset": len(dl_val)}

    return batch_size, task2num_iters, dset_val, dl_val