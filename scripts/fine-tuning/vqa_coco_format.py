# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
import argparse
import json
import os
from pathlib import Path

import multimodal
import numpy as np
from tqdm import tqdm

seed = 42


class VQA2FusionBrain:
    """
    Class for the VQA task. Translates questions in Russian into English.
    """

    def __init__(self, dir_data, questions_file="questions.json"):
        self.dir_data = dir_data

        with open(dir_data / questions_file) as f:
            self.questions = json.load(f)
            print(f"Fusion brain vqa2 size = {len(self.questions.keys())}")

    def __iter__(self):
        self.cur_idx = 0
        return self

    def __next__(self):
        self.cur_idx += 1
        if self.cur_idx > len(self.questions):
            raise StopIteration
        return self.__getitem__(self.cur_idx - 1)  # Нумерация с нуля

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        return self.questions[str(index)]


def parse_args():
    parser = argparse.ArgumentParser("Conversion script")

    parser.add_argument(
        "--data_path",
        default="",
        required=True,
        type=str,
        help="Path to the vqa dataset",
    )

    # parser.add_argument(
    #     "--img_path",
    #     default="",
    #     required=True,
    #     type=str,
    #     help="Path to the vqa image dataset",
    # )

    parser.add_argument(
        "--out_path",
        default=None,
        type=str,
        help="Path where to export the resulting dataset.",
    )

    parser.add_argument(
        "--coco_path",
        default="",
        required=False,
        type=str,
        help="Path to coco dataset.",
    )

    parser.add_argument(
        "--fusion_brain",
        action="store_true",
        help="Whether to run inference for the fusion brain",
    )

    parser.add_argument(
        "--use_translated",
        action="store_true",
        help="Whether to use translated versions of the questions. Works only with --fusion_brain option"
    )

    return parser.parse_args()


def split_val(dataset):

    image_ids = set([x["image_id"] for x in dataset])
    minival_image_ids = np.random.choice(list(image_ids), 2000, replace=False)

    minival_dataset = []
    train_val_dataset = []
    for item in dataset:
        if item["image_id"] in minival_image_ids:
            minival_dataset.append(item)
        else:
            train_val_dataset.append(item)

    return train_val_dataset, minival_dataset


def convert(split, data_path, output_path, coco_path, use_translated=False):
    if split == "fusion_brain":
        dataset = None
        questions_path = data_path / "translated_questions.json"
        # If questions have been already translated and were enabled, use them
        if use_translated and questions_path.is_file():
            print("Using translated version of the questions")
            dataset = list(VQA2FusionBrain(dir_data=data_path,
                                           questions_file="translated_questions.json"))
        else:
            dataset = list(VQA2FusionBrain(dir_data=data_path))
        categories = [{"supercategory": "object", "id": 1, "name": "object"}]
        annotations = []
        images = []
        d_name = "vqa2"

        for idx, question_data in tqdm(enumerate(dataset)):
            cur_img = {
                "file_name": question_data["file_name"],
                "height": None,  # потенциально есть в таргет, но не используется
                "width": None,  # потенциально есть в таргет, но не используется
                "id": idx,  # эквивалентно next_img_id, так как тип датасета один единственный
                "original_id": None,  # потенциально есть в таргет, но не используется
                "caption": question_data["question"],
                "questionId": idx,  # всё равно никак не влияет, так как участвует только в target
                "answer": None,
                "scores": None,
                "answer_type": "other",  # заглушка, чтобы не вылетало KeyError в self.type2id[coco_img["answer_type"]]
                "question_type": "global",  # аналогичная заглушка для gqa
                "tokens_negative": [(0, len(question_data["question"]))],  # в vqa2 не используются, так что не влияет
                "dataset_name": d_name,
            }

            images.append(cur_img)

        ds = {"info": [], "licenses": [], "images": images,
              "annotations": annotations, "categories": categories}

        print("Writing to file....")
        with open(output_path / f"inference_vqa2_fusion_brain.json",
                  "w") as j_file:
            json.dump(ds, j_file)
        print("Done!")
        return 0, 0

    dataset = list(multimodal.datasets.VQA2(dir_data=data_path, min_ans_occ=9, split=split))

    print(f"Dumping {split}...")
    next_img_id = 0
    next_id = 0

    if split in ["train"]:
        iminfo_files = ["instances_train2014"]
    elif split in ["val"]:
        iminfo_files = ["instances_val2014"]
    elif split in ["test-dev"]:
        iminfo_files = ["image_info_test-dev2015", "image_info_test2015"]
    elif split in ["test"]:
        iminfo_files = ["image_info_test2015"]
    else:
        assert False, f"Split {split} not recognized"

    imid2data = {}
    for iminfo_file in iminfo_files:
        with open(f"{coco_path}/annotations/{iminfo_file}.json", "r") as f:
            iminfo = json.load(f)
            imid2data.update({x["id"]: x for x in iminfo["images"]})

    if split == "val":
        trainval_dataset, minival_dataset = split_val(dataset)
        datasets = {"trainval": trainval_dataset, "minival": minival_dataset}
    else:
        datasets = {split: dataset}

    for dset_name, dset in datasets.items():

        categories = [{"supercategory": "object", "id": 1, "name": "object"}]
        annotations = []
        images = []
        d_name = "vqa2"

        for idx, datum in tqdm(enumerate(dset)):

            image_id = datum["image_id"]

            if dset_name in ["train", "trainval", "minival"]:

                cur_img = {
                    "file_name": imid2data[image_id]["file_name"],
                    "height": imid2data[image_id]["height"],
                    "width": imid2data[image_id]["width"],
                    "id": next_img_id,
                    "original_id": image_id,
                    "caption": datum["question"],
                    "questionId": datum["question_id"],
                    "answer": datum["multiple_choice_answer"],
                    "scores": datum["scores"],
                    "answer_type": datum["answer_type"],
                    "tokens_negative": [(0, len(datum["question"]))],
                    "dataset_name": d_name,
                }

            else:

                cur_img = {
                    "file_name": imid2data[image_id]["file_name"],
                    "height": imid2data[image_id]["height"],
                    "width": imid2data[image_id]["width"],
                    "id": next_img_id,
                    "original_id": image_id,
                    "caption": datum["question"],
                    "questionId": datum["question_id"],
                    "answer": None,
                    "scores": None,
                    "answer_type": None,
                    "tokens_negative": [(0, len(datum["question"]))],
                    "dataset_name": d_name,
                }

            next_img_id += 1
            images.append(cur_img)

        ds = {"info": [], "licenses": [], "images": images, "annotations": annotations, "categories": categories}

        print("Writing to file....")
        with open(output_path / f"finetune_vqa2_{dset_name}.json", "w") as j_file:
            json.dump(ds, j_file)
        print("Done!")

    return next_img_id, next_id


def main(args):
    data_path = Path(args.data_path)
    output_path = Path(args.out_path) if args.out_path is not None else data_path
    np.random.seed(seed)
    os.makedirs(str(output_path), exist_ok=True)

    if args.fusion_brain:
        convert("fusion_brain", data_path, output_path, args.coco_path, args.use_translated)
    else:
        for split in ["train", "val", "test-dev", "test"]:
            convert(split, data_path, output_path, args.coco_path)


if __name__ == "__main__":
    main(parse_args())
