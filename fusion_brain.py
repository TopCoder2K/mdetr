# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import os
import random
import time
from collections import namedtuple
from copy import deepcopy
from functools import partial
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.utils
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

import util.dist as dist
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--run_name", default="", type=str)

    # Dataset specific
    parser.add_argument("--dataset_config", default=None, required=True)
    parser.add_argument("--do_qa", action="store_true", help="Whether to do question answering")
    # parser.add_argument(
    #     "--predict_final",
    #     action="store_true",
    #     help="If true, will predict if a given box is in the actual referred set. Useful for CLEVR-Ref+ only currently.",
    # )
    # parser.add_argument("--no_detection", action="store_true", help="Whether to train the detector")
    parser.add_argument(
        "--split_qa_heads", action="store_true", help="Whether to use a separate head per question type in vqa"
    )
    # parser.add_argument(
    #     "--combine_datasets", nargs="+", help="List of datasets to combine for training", default=["flickr"]
    # )
    # parser.add_argument(
    #     "--combine_datasets_val", nargs="+", help="List of datasets to combine for eval", default=["flickr"]
    # )
    parser.add_argument(
        "--combine_datasets_test", nargs="+", help="List of datasets to combine for eval", default=["vqa2"]
    )

    parser.add_argument("--coco_path", type=str, default="")
    parser.add_argument("--vg_img_path", type=str, default="")
    parser.add_argument("--vg_ann_path", type=str, default="")
    # parser.add_argument("--clevr_img_path", type=str, default="")
    # parser.add_argument("--clevr_ann_path", type=str, default="")
    # parser.add_argument("--phrasecut_ann_path", type=str, default="")
    # parser.add_argument(
    #     "--phrasecut_orig_ann_path",
    #     type=str,
    #     default="",
    # )
    # parser.add_argument("--modulated_lvis_ann_path", type=str, default="")

    # Training hyper-parameters
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--text_encoder_lr", default=5e-5, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--lr_drop", default=35, type=int)
    parser.add_argument(
        "--epoch_chunks",
        default=-1,
        type=int,
        help="If greater than 0, will split the training set into chunks and validate/checkpoint after each chunk",
    )
    # parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm")
    parser.add_argument(
        "--eval_skip",
        default=1,
        type=int,
        help='do evaluation every "eval_skip" frames',
    )

    parser.add_argument(
        "--schedule",
        default="linear_with_warmup",
        type=str,
        choices=("step", "multistep", "linear_with_warmup", "all_linear_with_warmup"),
    )
    # parser.add_argument("--ema", action="store_true")
    # parser.add_argument("--ema_decay", type=float, default=0.9998)
    parser.add_argument("--fraction_warmup_steps", default=0.01, type=float, help="Fraction of total number of steps")

    # Model parameters
    parser.add_argument(
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )

    parser.add_argument(
        "--freeze_text_encoder", action="store_true", help="Whether to freeze the weights of the text encoder"
    )

    parser.add_argument(
        "--text_encoder_type",
        default="roberta-base",
        choices=("roberta-base", "distilroberta-base", "roberta-large"),
    )

    # Backbone
    parser.add_argument(
        "--backbone",
        default="resnet101",
        type=str,
        help="Name of the convolutional backbone to use such as resnet50 resnet101 timm_tf_efficientnet_b3_ns",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )

    # Transformer
    parser.add_argument(
        "--enc_layers",
        default=6,
        type=int,
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument("--num_queries", default=100, type=int, help="Number of query slots")
    parser.add_argument("--pre_norm", action="store_true")
    parser.add_argument(
        "--no_pass_pos_and_query",
        dest="pass_pos_and_query",
        action="store_false",
        help="Disables passing the positional encodings to each attention layers",
    )

    # Segmentation
    parser.add_argument(
        "--mask_model",
        default="none",
        type=str,
        choices=("none", "smallconv", "v2"),
        help="Segmentation head to be used (if None, segmentation will not be trained)",
    )
    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument("--masks", action="store_true")

    # Loss
    # parser.add_argument(
    #     "--no_aux_loss",
    #     dest="aux_loss",
    #     action="store_false",
    #     help="Disables auxiliary decoding losses (loss at each layer)",
    # )
    # parser.add_argument(
    #     "--set_loss",
    #     default="hungarian",
    #     type=str,
    #     choices=("sequential", "hungarian", "lexicographical"),
    #     help="Type of matching to perform in the loss",
    # )
    #
    # parser.add_argument("--contrastive_loss", action="store_true", help="Whether to add contrastive loss")
    # parser.add_argument(
    #     "--no_contrastive_align_loss",
    #     dest="contrastive_align_loss",
    #     action="store_false",
    #     help="Whether to add contrastive alignment loss",
    # )
    #
    # parser.add_argument(
    #     "--contrastive_loss_hdim",
    #     type=int,
    #     default=64,
    #     help="Projection head output size before computing normalized temperature-scaled cross entropy loss",
    # )
    #
    # parser.add_argument(
    #     "--temperature_NCE", type=float, default=0.07, help="Temperature in the  temperature-scaled cross entropy loss"
    # )

    # * Matcher
    parser.add_argument(
        "--set_cost_class",
        default=1,
        type=float,
        help="Class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_bbox",
        default=5,
        type=float,
        help="L1 box coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_giou",
        default=2,
        type=float,
        help="giou box coefficient in the matching cost",
    )

    # Loss coefficients
    # parser.add_argument("--ce_loss_coef", default=1, type=float)
    # parser.add_argument("--mask_loss_coef", default=1, type=float)
    # parser.add_argument("--dice_loss_coef", default=1, type=float)
    # parser.add_argument("--bbox_loss_coef", default=5, type=float)
    # parser.add_argument("--giou_loss_coef", default=2, type=float)
    # parser.add_argument("--qa_loss_coef", default=1, type=float)
    # parser.add_argument(
    #     "--eos_coef",
    #     default=0.1,
    #     type=float,
    #     help="Relative classification weight of the no-object class",
    # )
    # parser.add_argument("--contrastive_loss_coef", default=0.1, type=float)
    # parser.add_argument("--contrastive_align_loss_coef", default=1, type=float)

    # Run specific
    parser.add_argument("--inference", action="store_true", help="Whether to run inference only")
    parser.add_argument("--test", action="store_true", help="Whether to run evaluation on val or test set")
    parser.add_argument("--test_type", type=str, default="test", choices=("testA", "testB", "test"))
    parser.add_argument("--output-dir", default="", help="path where to save, empty for no saving")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--load", default="", help="resume from checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval", action="store_true", help="Only run evaluation")
    parser.add_argument("--num_workers", default=5, type=int)
    parser.add_argument("--do_qa_with_qa_fine_tuned", action="store_true",
                        help="Have the model been already fine-tuned on other QA dataset?")

    # Distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    return parser

def run_inference(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    dset_name: str,
    args,
):
    """
    Runs inference on a dataset. Makes output/prediction_X.json according to the task output format,
    where X is in ["C2C", "HTR", "zsOD", "VQA"].
    """

    model.eval()
    fb_answers = {}

    for i, batch_dict in tqdm(enumerate(data_loader)):
        samples = batch_dict["samples"].to(device)
        positive_map = batch_dict["positive_map"].to(
            device) if "positive_map" in batch_dict else None
        targets = batch_dict["targets"]
        # answers = {k: v.to(device) for k, v in batch_dict[
        #     "answers"].items()} if "answers" in batch_dict else None
        captions = [t["caption"] for t in targets]

        # targets = utils.targets_to(targets, device)

        memory_cache = None
        if args.masks:
            outputs = model(samples, captions)
        else:
            memory_cache = model(samples, captions, encode_and_save=True)
            outputs = model(samples, captions, encode_and_save=False,
                            memory_cache=memory_cache)

        if dset_name == "vqa2":
            if args.split_qa_heads:
                id2type = {v: k for k, v in data_loader.dataset.type2id.items()}
                id2answer_by_type = {ans_type: {v: k for k, v in ans2id.items()}
                                     for ans_type, ans2id in data_loader.dataset.answer2id_by_type.items()}
                answer_type = outputs["pred_answer_type"].argmax(-1).cpu().numpy()
                # print(outputs["pred_answer_type"], answer_type)

                for j in range(len(answer_type)):
                    pred_answer_id = outputs["pred_answer_" + id2type[answer_type[j]]][j].argmax(-1).item()
                    pred_answer = id2answer_by_type[id2type[answer_type[j]]][pred_answer_id]
                    fb_answers[str(i * args.batch_size + j)] = pred_answer
                # print("pred_answer_" + id2type[answer_type[0]], outputs["pred_answer_" + id2type[answer_type[0]]].shape)

            else:
                id2answer = {idx: answer for answer, idx in data_loader.dataset.answer2id.items()}
                answers_ids = outputs["pred_answer"].argmax(-1).cpu().numpy()

                for j in range(len(answers_ids)):
                    fb_answers[str(i * args.batch_size + j)] = id2answer[answers_ids[j]]

        else:
            assert False, "Not implemented"

    if not os.path.exists("./output/"):
        os.makedirs("./output/")
    with open("./output/prediction_VQA_untranslated.json", "w") as f:
        json.dump(fb_answers, f)

def main(args):
    # Init distributed mode
    dist.init_distributed_mode(args)

    # Update dataset specific configs
    if args.dataset_config is not None:
        # https://stackoverflow.com/a/16878364
        d = vars(args)
        with open(args.dataset_config, "r") as f:
            cfg = json.load(f)
        d.update(cfg)

    print("git:\n  {}\n".format(utils.get_sha()))

    # Segmentation related
    if args.mask_model != "none":
        args.masks = True
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

    print(args)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)

    # fix the seed for reproducibility
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_deterministic(True)

    # Build the model
    model = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    # Set up optimizers
    # param_dicts = [
    #     {
    #         "params": [
    #             p
    #             for n, p in model_without_ddp.named_parameters()
    #             if "backbone" not in n and "text_encoder" not in n and p.requires_grad
    #         ]
    #     },
    #     {
    #         "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
    #         "lr": args.lr_backbone,
    #     },
    #     {
    #         "params": [p for n, p in model_without_ddp.named_parameters() if "text_encoder" in n and p.requires_grad],
    #         "lr": args.text_encoder_lr,
    #     },
    # ]
    # if args.optimizer == "sgd":
    #     optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # elif args.optimizer in ["adam", "adamw"]:
    #     optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    # else:
    #     raise RuntimeError(f"Unsupported optimizer {args.optimizer}")

    # Test dataset
    if len(args.combine_datasets_test) == 0:
        raise RuntimeError("Please provide at least one test dataset")

    Test_all = namedtuple(typename="test_data", field_names=["dataset_name", "dataloader", "base_ds"])

    test_tuples = []
    for dset_name in args.combine_datasets_test:
        dset = build_dataset(dset_name, image_set="fusion_brain", args=args)
        sampler = (
            DistributedSampler(dset, shuffle=False) if args.distributed else torch.utils.data.SequentialSampler(dset)
        )
        dataloader = DataLoader(
            dset,
            args.batch_size,
            sampler=sampler,
            drop_last=False,
            collate_fn=partial(utils.collate_fn, False),
            num_workers=args.num_workers,
        )
        base_ds = get_coco_api_from_dataset(dset)
        test_tuples.append(Test_all(dataset_name=dset_name, dataloader=dataloader, base_ds=base_ds))

    # if args.frozen_weights is not None:
    #     if args.resume.startswith("https"):
    #         checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location="cpu", check_hash=True)
    #     else:
    #         checkpoint = torch.load(args.resume, map_location="cpu")
    #     if "model_ema" in checkpoint and checkpoint["model_ema"] is not None:
    #         model_without_ddp.detr.load_state_dict(checkpoint["model_ema"], strict=False)
    #     else:
    #         model_without_ddp.detr.load_state_dict(checkpoint["model"], strict=False)
    #
    #     if args.ema:
    #         model_ema = deepcopy(model_without_ddp)

    # Used for loading weights from another model and starting a training from scratch. Especially useful if
    # loading into a model with different functionality.
    if args.load:
        print("loading from", args.load)
        checkpoint = torch.load(args.load, map_location="cpu")
        if "model_ema" in checkpoint:
            if args.do_qa_with_qa_fine_tuned:
                # Delete mismatching weights:
                del checkpoint["model_ema"]["qa_embed.weight"]
                del checkpoint["model_ema"]["answer_type_head.weight"]
                del checkpoint["model_ema"]["answer_type_head.bias"]
            model_without_ddp.load_state_dict(checkpoint["model_ema"], strict=False)
        else:
            if args.do_qa_with_qa_fine_tuned:
                # Delete mismatching weights:
                del checkpoint["model"]["qa_embed.weight"]
                del checkpoint["model"]["answer_type_head.weight"]
                del checkpoint["model"]["answer_type_head.bias"]

            model_without_ddp.load_state_dict(checkpoint["model"], strict=False)

    # Used for resuming training from the checkpoint of a model. Used when training times-out or is pre-empted.
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location="cpu", check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])

    # Runs testing on the dataset
    for i, item in enumerate(test_tuples):
        print(f"Inference on {item.dataset_name}")
        run_inference(
            model=model,
            data_loader=item.dataloader,
            dset_name=item.dataset_name,
            device=device,
            args=args,
        )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MDETR inference script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
