# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/ashkamath/mdetr/blob/main/datasets/gqa.py
"""
import json
from pathlib import Path

import torch
import torchvision
from transformers import RobertaTokenizerFast

from .coco import ConvertCocoPolysToMask, ModulatedDetection, make_coco_transforms

class VQAv2Detection(ModulatedDetection):
    pass

class VQAv2QuestionAnswering(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, return_tokens, tokenizer, ann_folder):
        super(VQAv2QuestionAnswering, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, return_tokens, tokenizer=tokenizer)
        with open(ann_folder / "vqa2_answer2id.json", "r") as f:
            self.answer2id = json.load(f)
        with open(ann_folder / "vqa2_answer2id_by_type.json", "r") as f:
            self.answer2id_by_type = json.load(f)
        self.type2id = {"yes/no": 0, "number": 1, "other": 2}

    def __getitem__(self, idx):
        img, target = super(VQAv2QuestionAnswering, self).__getitem__(idx)  # target ---
        image_id = self.ids[idx]
        coco_img = self.coco.loadImgs(image_id)[0]
        caption = coco_img["caption"]
        dataset_name = coco_img["dataset_name"]
        questionId = coco_img["questionId"]
        target = {"image_id": image_id, "annotations": target, "caption": caption}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        target["dataset_name"] = dataset_name
        target["questionId"] = questionId

        if coco_img["answer"] not in self.answer2id:
            answer = "unknown"
        else:
            answer = coco_img["answer"]

        target["answer"] = torch.as_tensor(self.answer2id[answer], dtype=torch.long)
        target["answer_type"] = torch.as_tensor(self.type2id[coco_img["answer_type"]], dtype=torch.long)

        # util.misc.collate_fn requires to put 'answer' before every type of answer in target
        if coco_img["answer"] not in self.answer2id_by_type["yes/no"]:
            answer = "unknown"
        else:
            answer = coco_img["answer"]
        target["answer_yes/no"] = torch.as_tensor(
            self.answer2id_by_type["yes/no"][answer] if coco_img["answer_type"] == "yes/no" else -100,
            dtype=torch.long,
        )

        if coco_img["answer"] not in self.answer2id_by_type["number"]:
            answer = "unknown"
        else:
            answer = coco_img["answer"]
        target["answer_number"] = torch.as_tensor(
            self.answer2id_by_type["number"][answer] if coco_img["answer_type"] == "number" else -100,
            dtype=torch.long,
        )

        if coco_img["answer"] not in self.answer2id_by_type["other"]:
            answer = "unknown"
        else:
            answer = coco_img["answer"]
        target["answer_other"] = torch.as_tensor(
            self.answer2id_by_type["other"][answer] if coco_img["answer_type"] == "other" else -100,
            dtype=torch.long,
        )

        return img, target


def build(image_set, args):
    # TODO: img or all?
    img_dir = Path(args.coco_img_path)
    assert img_dir.exists(), f"provided COCO img path {img_dir} does not exist"

    tokenizer = RobertaTokenizerFast.from_pretrained(args.text_encoder_type)

    if image_set == "fusion_brain":
        ann_file = Path(args.vqa2_ann_path) / "inference_vqa2_fusion_brain.json"

        return VQAv2QuestionAnswering(
            img_dir,
            ann_file,
            transforms=make_coco_transforms("val", cautious=True),
            return_masks=args.masks,
            return_tokens=True,
            tokenizer=tokenizer,
            ann_folder=Path(args.vqa2_ann_path),
        )

    if args.do_qa:
        # Для vqa2 это не нужно:
        # assert args.vqa2_split_type is not None

        if image_set == "train":
            datasets = []
            for imset in ["train", "minival"]:
                ann_file = Path(args.vqa2_ann_path) / f"finetune_vqa2_{imset}.json"

                datasets.append(
                    VQAv2QuestionAnswering(
                        img_dir / "train2014" if imset == "train" else img_dir / "val2014",
                        ann_file,
                        transforms=make_coco_transforms(image_set, cautious=True),
                        return_masks=args.masks,
                        return_tokens=True,
                        tokenizer=tokenizer,
                        ann_folder=Path(args.vqa2_ann_path),
                    )
                )

            return torch.utils.data.ConcatDataset(datasets)
        elif image_set == "val":
            # TODO: правильный ли ann_file?
            ann_file = Path(args.vqa2_ann_path) / f"finetune_vqa2_minival.json"

            return VQAv2QuestionAnswering(
                img_dir / "val2014",
                ann_file,
                transforms=make_coco_transforms(image_set, cautious=True),
                return_masks=args.masks,
                return_tokens=True,
                tokenizer=tokenizer,
                ann_folder=Path(args.vqa2_ann_path),
            )
        elif image_set in ["test", "testdev", "trainval"]:
            ann_file = Path(args.vqa2_ann_path) / f"finetune_vqa2_{image_set}.json"

            return VQAv2QuestionAnswering(
                img_dir / "test2015",
                ann_file,
                transforms=make_coco_transforms("val", cautious=True),
                return_masks=args.masks,
                return_tokens=True,
                tokenizer=tokenizer,
                ann_folder=Path(args.vqa2_ann_path),
            )
        else:
            assert False, f"Unknown image set {image_set}"
