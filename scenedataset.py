# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import pickle
import logging
import copy

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)

def _load_dataset(dataroot):
    """Load entries

    dataroot: root path of dataset

    """

    question_path_test = os.path.join(dataroot,"HL_test.json")
    questions_test = json.load(open(question_path_test))["data"]

    entries = []
    image_entries = []
    for question in questions_test:
        for i,scene in enumerate(questions_test[question]["place"]):
            image_caption = {}
            image_caption["image"] = question
            image_caption["scene"] = questions_test[question]["place"][i]
            entries.append(image_caption)
            #image_entries.append(question.strip(".jpg"))

        """
        {'COCO_train2014_000000138878.jpg': {'action': ['posing for a photo', 'the person is posing for a photo', "he's sitting in an armchair."], 
        'place': ['in a car', 'the picture is taken in a car', 'in an office.'], 'reason': ['to have a picture of himself', 'he wants to share it 
         with his friends',  "he's working and took a professional photo."]}, 
        """

        """
        {'image': 'COCO_train2014_000000138878.jpg', 'place': ['in a car']}
        """

    return entries


class SceneDataset(Dataset):
    def __init__(
        self,
        dataroot,
        #annotations_jsonpath,
        #image_features_reader,
        #gt_image_features_reader,
        tokenizer,
        bert_model,
        padding_index=0,
        max_seq_length=16,
        max_region_num=101,
        num_locs=5,
        add_global_imgfeat="First",
        append_mask_sep=False,
    ):
        super().__init__()
        self._max_region_num = max_region_num + int(add_global_imgfeat is not None)
        print("_max_region_num:")
        print(self._max_region_num)
        #为什么加一个add_global_imgfeat就要在max_region_num+1
        self._max_seq_length = max_seq_length
        #self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer
        self._padding_index = padding_index
        self._num_locs = num_locs
        #self._num_locs = 5
        self._add_global_imgfeat = add_global_imgfeat
        self._append_mask_sep = append_mask_sep

        os.makedirs(os.path.join(dataroot, "cache"), exist_ok=True)
        cache_path = os.path.join(
            dataroot,
            "cache",
            "scene_probing" #task
            + "_"
            + bert_model.split("/")[-1]
            + "_"
            + str(max_seq_length)
            + ".pkl",
        )

        #if not os.path.exists(cache_path):      ##do we need this cache_path?
        self.entries = _load_dataset(dataroot)
        #print("preprocessed dataset:")
        #print(self.entries)
        self.tokenize(max_seq_length)
        self.tensorize()
        i = 0
        for index, entry in enumerate(self.entries):
            features, spatials, image_mask, question, input_mask, segment_ids, index = self.__getitem__(index)
            i+=1
        print("index:")
        print(i)
        print("len of dataset:")
        print(len(self.entries))
            #cPickle.dump(self.entries, open(cache_path, "wb"))
        #else:
            #logger.info("Loading from %s" % cache_path)
            #self.entries = cPickle.load(open(cache_path, "rb"))

        #self.qid2imgid = {e["question_id"]: e["image"] for e in self.entries}
        #self._image_entries = len(self.entries)
        #data_features_root = "/Users/cenkaiwei/Documents/volta_Probing task/features/test/"

        #for i, image_id in enumerate(self.image_entries):
            #visual_information = np.load(data_features_root + image_id + ".npz")
            #features = visual_information["features"]
            #num_boxes = visual_information["num_boxes"]
            #boxes = visual_information["boxes"]

           #mix_num_boxes = min(int(num_boxes), self._max_region_num)
            # 为什么要取最小

            #mix_boxes_pad = np.zeros((self._max_region_num, self._num_locs))
            # waht is num_locs, 为什么要pad
            # num_locs不同的model会不会不一样

            #mix_features_pad = np.zeros((self._max_region_num, 2048))

            #image_mask = [1] * (int(mix_num_boxes))
            #while len(image_mask) < self._max_region_num:
                #image_mask.append(0)
            # 为什么要这样做

            #mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
            #mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

            #features = torch.tensor(mix_features_pad).float()
            #image_mask = torch.tensor(image_mask).long()
            #spatials = torch.tensor(mix_boxes_pad).float()


    def tokenize(self, max_length=16):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_index in embedding
        """
        for entry in self.entries:
            tokens = self._tokenizer.encode(entry["scene"])
            tokens = [tokens[0]] + tokens[1:-1][: self._max_seq_length - 2] + [tokens[-1]]
            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            #print(tokens)
            #print(segment_ids)
            #print(input_mask)

            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (max_length - len(tokens))
                tokens = tokens + padding
                input_mask += [0] * len(padding)
                segment_ids += [0] * len(padding)

            assert_eq(len(tokens), max_length)
            #the actual sequence length is 38
            entry["q_token"] = tokens
            #print(entry["q_token"])
            entry["q_input_mask"] = input_mask
            #print(entry["q_input_mask"])
            entry["q_segment_ids"] = segment_ids
            #print(entry["q_segment_ids"])
            #print(entry)


    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry["q_token"]))
            entry["q_token"] = question
            q_input_mask = torch.from_numpy(np.array(entry["q_input_mask"]))
            entry["q_input_mask"] = q_input_mask
            q_segment_ids = torch.from_numpy(np.array(entry["q_segment_ids"]))
            entry["q_segment_ids"] = q_segment_ids

    def __getitem__(self,index):
        entry = self.entries[index]
        image_id = entry["image"].strip(".jpg")
        #print(image_id)
        data_features_root = "./features/test/"
        visual_information = np.load(data_features_root + image_id + ".npz")
        features = visual_information["features"] #(150,2048)
        num_boxes = visual_information["num_boxes"] #150
        boxes = visual_information["boxes"] #[150,4]

        image_h = int(visual_information["img_h"])
        image_w = int(visual_information["img_w"])
        image_location = np.zeros((boxes.shape[0], self._num_locs), dtype=np.float32)
        image_location[:, :4] = boxes
        if self._num_locs >= 5:
            image_location[:, -1] = (
                    (image_location[:, 3] - image_location[:, 1])
                    * (image_location[:, 2] - image_location[:, 0])
                    / (float(image_w) * float(image_h))
            )

        image_location[:, 0] = image_location[:, 0] / float(image_w)
        image_location[:, 1] = image_location[:, 1] / float(image_h)
        image_location[:, 2] = image_location[:, 2] / float(image_w)
        image_location[:, 3] = image_location[:, 3] / float(image_h)

        g_location = [0, 0, 1, 1] + [1] * (self._num_locs - 4)
        _image_location = np.concatenate([np.expand_dims(g_location, axis=0), image_location], axis=0)
        _boxes = _image_location

        #print("shape of boxes:")
        #shape_of_boxes = boxes.shape()
        #print(shape_of_boxes)

        # num_boxes number of detected objects
        # "features" [shape: (num_images, num_proposals, feature_size)] 什么是feature_size，num_proposals和num_boxes有什么不一样的


        mix_num_boxes = min(int(num_boxes), self._max_region_num)
        #print(self._max_region_num)
        #为什么要取最小

        mix_boxes_pad = np.zeros((self._max_region_num, self._num_locs)) #[37,5]
        #为什么要pad


        mix_features_pad = np.zeros((self._max_region_num, 2048))  #[37, 2048]

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)
        #为什么要这样做

        mix_boxes_pad[:mix_num_boxes] = _boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        features = torch.tensor(mix_features_pad).float()
        #print(features)
        image_mask = torch.tensor(image_mask).long()
        #print(image_mask)
        spatials = torch.tensor(mix_boxes_pad).float()
        #print(spatials)

        #box里面都有啥
        #image_mask有啥用处

        question = entry["q_token"]
        #print(question)
        input_mask = entry["q_input_mask"]
        #print(input_mask)
        segment_ids = entry["q_segment_ids"]
        #print(segment_ids)


        return features, spatials, image_mask, question, input_mask, segment_ids, index

    def __len__(self):
        return len(self.entries)

