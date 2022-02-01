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

import numpy as np
import torch
import torch.utils
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

import util.dist as dist
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.clevrref import ClevrRefEvaluator
from datasets.coco_eval import CocoEvaluator
from datasets.flickr_eval import FlickrEvaluator
from datasets.phrasecut_eval import PhrasecutEvaluator
from datasets.refexp import RefExpEvaluator
from fusion_brain_engine import evaluate, train_one_epoch
from models import build_model
from models.postprocessors import build_postprocessors


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--run_name", default="", type=str)

    # Dataset specific
    parser.add_argument("--dataset_config", default=None, required=True)
    parser.add_argument("--do_qa", action="store_true", help="Whether to do question answering")
    parser.add_argument(
        "--predict_final",
        action="store_true",
        help="If true, will predict if a given box is in the actual referred set. Useful for CLEVR-Ref+ only currently.",
    )
    parser.add_argument("--no_detection", action="store_true", help="Whether to train the detector")
    parser.add_argument(
        "--split_qa_heads", action="store_true", help="Whether to use a separate head per question type in vqa"
    )
    parser.add_argument(
        "--combine_datasets", nargs="+", help="List of datasets to combine for training", default=["flickr"]
    )
    parser.add_argument(
        "--combine_datasets_val", nargs="+", help="List of datasets to combine for eval", default=["flickr"]
    )
    parser.add_argument(
        "--co_training", action="store_true", help="Whether to train the model on both datasets simultaneously"
    )

    parser.add_argument("--coco_path", type=str, default="")
    parser.add_argument("--vg_img_path", type=str, default="")
    parser.add_argument("--vg_ann_path", type=str, default="")
    parser.add_argument("--clevr_img_path", type=str, default="")
    parser.add_argument("--clevr_ann_path", type=str, default="")
    parser.add_argument("--phrasecut_ann_path", type=str, default="")
    parser.add_argument(
        "--phrasecut_orig_ann_path",
        type=str,
        default="",
    )
    parser.add_argument("--modulated_lvis_ann_path", type=str, default="")

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
    parser.add_argument("--optimizer", default="adam", type=str)
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
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9998)
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
    parser.add_argument("--finetune_decoder", action="store_true",
                        help="If set, loads LVIS checkpoint and finetunes decoder for the VQA task")

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
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )
    parser.add_argument(
        "--set_loss",
        default="hungarian",
        type=str,
        choices=("sequential", "hungarian", "lexicographical"),
        help="Type of matching to perform in the loss",
    )

    parser.add_argument("--contrastive_loss", action="store_true", help="Whether to add contrastive loss")
    parser.add_argument(
        "--no_contrastive_align_loss",
        dest="contrastive_align_loss",
        action="store_false",
        help="Whether to add contrastive alignment loss",
    )

    parser.add_argument(
        "--contrastive_loss_hdim",
        type=int,
        default=64,
        help="Projection head output size before computing normalized temperature-scaled cross entropy loss",
    )

    parser.add_argument(
        "--temperature_NCE", type=float, default=0.07, help="Temperature in the  temperature-scaled cross entropy loss"
    )

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
    parser.add_argument("--ce_loss_coef", default=1, type=float)
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument("--qa_loss_coef", default=1, type=float)
    parser.add_argument(
        "--eos_coef",
        default=0.1,
        type=float,
        help="Relative classification weight of the no-object class",
    )
    parser.add_argument("--contrastive_loss_coef", default=0.1, type=float)
    parser.add_argument("--contrastive_align_loss_coef", default=1, type=float)

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
    parser.add_argument("--do_qa_with_qa_fine_tuned", action="store_true", help="Have the model been already fine-tuned on other QA dataset?")

    # Distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    return parser


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
    model, criterion, contrastive_criterion, qa_criterion, weight_dict = build_model(args)
    model.to(device)

    assert (
        criterion is not None or qa_criterion is not None
    ), "Error: should train either detection or question answering (or both)"

    # Get a copy of the model for exponential moving averaged version of the model
    model_ema = deepcopy(model) if args.ema else None
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    # Set up optimizers
    param_dicts = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and "text_encoder" not in n and p.requires_grad
            ]
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "text_encoder" in n and p.requires_grad],
            "lr": args.text_encoder_lr,
        },
    ]
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer in ["adam", "adamw"]:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Unsupported optimizer {args.optimizer}")

    # Train dataset
    if len(args.combine_datasets) == 0 and not args.eval:
        raise RuntimeError("Please provide at least one training dataset")

    dataset_train, sampler_train, data_loader_train = None, None, None
    if not args.eval and not args.co_training:
        dataset_train = ConcatDataset(
            [build_dataset(name, image_set="train", args=args) for name in args.combine_datasets]
        )

        # To handle very big datasets, we chunk it into smaller parts.
        if args.epoch_chunks > 0:
            print(
                f"Splitting the training set into {args.epoch_chunks} of size approximately",
                f"{len(dataset_train) // args.epoch_chunks}"
            )
            chunks = torch.chunk(torch.arange(len(dataset_train)), args.epoch_chunks)
            datasets = [torch.utils.data.Subset(dataset_train, chunk.tolist()) for chunk in chunks]
            if args.distributed:
                samplers_train = [DistributedSampler(ds) for ds in datasets]
            else:
                samplers_train = [torch.utils.data.RandomSampler(ds) for ds in datasets]

            batch_samplers_train = [
                torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
                for sampler_train in samplers_train
            ]
            assert len(batch_samplers_train) == len(datasets)
            data_loaders_train = [
                DataLoader(
                    ds,
                    batch_sampler=batch_sampler_train,
                    collate_fn=partial(utils.collate_fn, False),
                    num_workers=args.num_workers,
                )
                for ds, batch_sampler_train in zip(datasets, batch_samplers_train)
            ]
        else:
            if args.distributed:
                sampler_train = DistributedSampler(dataset_train)
            else:
                sampler_train = torch.utils.data.RandomSampler(dataset_train)

            batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
            data_loader_train = DataLoader(
                dataset_train,
                batch_sampler=batch_sampler_train,
                collate_fn=partial(utils.collate_fn, False),
                num_workers=args.num_workers,
            )
    elif args.co_training:
        dataset_vqa_train = build_dataset("vqa2", image_set="train", args=args)
        dataset_lvis_train = build_dataset("modulated_lvis", image_set="train", args=args)

        if args.epoch_chunks > 0:
            print(
                f"Splitting the training set into {args.epoch_chunks} of size approximately"
                f" {max(len(dataset_vqa_train), len(dataset_lvis_train)) // args.epoch_chunks}"
            )
            chunks_vqa = torch.chunk(torch.arange(len(dataset_vqa_train)), args.epoch_chunks)
            datasets_vqa = [torch.utils.data.Subset(dataset_vqa_train, chunk.tolist()) for chunk in chunks_vqa]
            chunks_lvis = torch.chunk(torch.arange(len(dataset_lvis_train)), args.epoch_chunks)
            datasets_lvis = [torch.utils.data.Subset(dataset_lvis_train, chunk.tolist()) for chunk in chunks_lvis]

            if args.distributed:
                samplers_vqa_train = [DistributedSampler(ds) for ds in datasets_vqa]
                samplers_lvis_train = [DistributedSampler(ds) for ds in datasets_lvis]
            else:
                samplers_vqa_train = [torch.utils.data.RandomSampler(ds) for ds in datasets_vqa]
                samplers_lvis_train = [torch.utils.data.RandomSampler(ds) for ds in datasets_lvis]

            # print(f"SAMPLERS_VQA_TRIAN: {samplers_vqa_train}")

            batch_samplers_vqa_train = [
                torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
                for sampler_train in samplers_vqa_train
            ]
            batch_samplers_lvis_train = [
                torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
                for sampler_train in samplers_lvis_train
            ]
            assert len(batch_samplers_vqa_train) == len(datasets_vqa)
            assert len(batch_samplers_lvis_train) == len(datasets_lvis)

            data_loaders_vqa_train = [
                DataLoader(
                    ds,
                    batch_sampler=batch_sampler_train,
                    collate_fn=partial(utils.collate_fn, False),
                    num_workers=args.num_workers,
                )
                for ds, batch_sampler_train in zip(datasets_vqa, batch_samplers_vqa_train)
            ]
            data_loaders_lvis_train = [
                DataLoader(
                    ds,
                    batch_sampler=batch_sampler_train,
                    collate_fn=partial(utils.collate_fn, False),
                    num_workers=args.num_workers,
                )
                for ds, batch_sampler_train in zip(datasets_lvis, batch_samplers_lvis_train)
            ]
        else:
            if args.distributed:
                sampler_vqa_train = DistributedSampler(dataset_vqa_train)
                sampler_lvis_train = DistributedSampler(dataset_lvis_train)
            else:
                sampler_vqa_train = torch.utils.data.RandomSampler(dataset_vqa_train)
                sampler_lvis_train = torch.utils.data.RandomSampler(dataset_lvis_train)

            batch_sampler_vqa_train = torch.utils.data.BatchSampler(sampler_vqa_train, args.batch_size, drop_last=True)
            batch_sampler_lvis_train = torch.utils.data.BatchSampler(sampler_lvis_train, args.batch_size,
                                                                     drop_last=True)
            data_loader_vqa_train = DataLoader(
                dataset_vqa_train,
                batch_sampler=batch_sampler_vqa_train,
                collate_fn=partial(utils.collate_fn, False),
                num_workers=args.num_workers,
            )
            data_loader_lvis_train = DataLoader(
                dataset_lvis_train,
                batch_sampler=batch_sampler_lvis_train,
                collate_fn=partial(utils.collate_fn, False),
                num_workers=args.num_workers
            )

    # Val dataset
    if len(args.combine_datasets_val) == 0:
        raise RuntimeError("Please provide at leas one validation dataset")

    Val_all = namedtuple(typename="val_data", field_names=["dataset_name", "dataloader", "base_ds", "evaluator_list"])

    val_tuples = []
    for dset_name in args.combine_datasets_val:
        dset = build_dataset(dset_name, image_set="val", args=args)
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
        val_tuples.append(Val_all(dataset_name=dset_name, dataloader=dataloader, base_ds=base_ds, evaluator_list=None))

    if args.frozen_weights is not None:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location="cpu", check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        if "model_ema" in checkpoint and checkpoint["model_ema"] is not None:
            model_without_ddp.detr.load_state_dict(checkpoint["model_ema"], strict=False)
        else:
            model_without_ddp.detr.load_state_dict(checkpoint["model"], strict=False)

        if args.ema:
            model_ema = deepcopy(model_without_ddp)

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

        if args.ema:
            model_ema = deepcopy(model_without_ddp)

    # Used for resuming training from the checkpoint of a model. Used when training times-out or is pre-empted.
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(args.resume, map_location="cpu", check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.eval and "optimizer" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1
        if args.ema:
            if "model_ema" not in checkpoint:
                print("WARNING: ema model not found in checkpoint, resetting to current model")
                model_ema = deepcopy(model_without_ddp)
            else:
                model_ema.load_state_dict(checkpoint["model_ema"])

    # TODO: make this more general, model_ema????????
    if args.finetune_decoder:
        desired_state_dict = model_without_ddp.state_dict()
        zsOD_tuned_state_dict = torch.load("/home/pchelintsev/MDETR/mdetr/lvis100_checkpoint.pth", map_location="cpu")
        vqa2_tuned_state_dict = torch.load("/home/pchelintsev/MDETR/mdetr/input/VQA/3875/BEST_checkpoint.pth",
                                           map_location="cpu")

        model_type = "model_ema" if "model_ema" in zsOD_tuned_state_dict else "model"
        zsOD_tuned_state_dict = zsOD_tuned_state_dict[model_type]
        vqa2_tuned_state_dict = vqa2_tuned_state_dict["model"]

        # Загрузили всё
        print("Loading QA related weights from ~/MDETR/mdetr/input/VQA/3875/BEST_checkpoint.pth")
        model_without_ddp.load_state_dict(vqa2_tuned_state_dict, strict=True)
        # Перезагрузили всё, что не связано с QA и декодером
        print("Loading zsOD related weights from ~/MDETR/mdetr/lvis100_checkpoint.pth")
        model_without_ddp.load_state_dict(zsOD_tuned_state_dict, strict=False)
        # Загрузка VQA декодера, так как он перезатёрся выше
        print("Loading decoder from ~/MDETR/mdetr/input/VQA/3875/BEST_checkpoint.pth")
        with torch.no_grad():
            for name, param in desired_state_dict.items():
                if "decoder" in name:
                    param.copy_(vqa2_tuned_state_dict[name])
        # Так как тестироваться будет именно ema версия, нужно её тоже проинициализировать
        if args.ema:
            model_ema = deepcopy(model_without_ddp)

        # dofinetune = torch.load(
        #     "/home/pchelintsev/MDETR/mdetr/checkpoint/pchelintsev/experiments/495/checkpoint0023.pth",
        #     map_location="cpu"
        # )
        # model_without_ddp.load_state_dict(dofinetune["model"])
        # optimizer.load_state_dict(dofinetune["optimizer"])
        # args.start_epoch = dofinetune["epoch"] + 1
        # model_ema.load_state_dict(dofinetune["model_ema"])

        print("Freezing not VQA parameters...")
        # Заморозка LVIS энкодера, text_encoder, backbone, feature_resizer, input_proj,
        # (bbox_embed, class_embed, contrastive_align оставим на всякий случай)
        for name, param in model_without_ddp.named_parameters():
            if "text_encoder" in name or "encoder" in name or "backbone" in name \
                    or "feature_resizer" in name or "input_proj" in name:
                param.requires_grad = False

        n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
        print("Updated number of params:", n_parameters)

    def build_evaluator_list(base_ds, dataset_name):
        """Helper function to build the list of evaluators for a given dataset"""
        evaluator_list = []
        if args.no_detection:
            return evaluator_list
        iou_types = ["bbox"]
        if args.masks:
            iou_types.append("segm")

        evaluator_list.append(CocoEvaluator(base_ds, tuple(iou_types), useCats=False))
        if "refexp" in dataset_name:
            evaluator_list.append(RefExpEvaluator(base_ds, ("bbox")))
        if "clevrref" in dataset_name:
            evaluator_list.append(ClevrRefEvaluator(base_ds, ("bbox")))
        if "flickr" in dataset_name:
            evaluator_list.append(
                FlickrEvaluator(
                    args.flickr_dataset_path,
                    subset="test" if args.test else "val",
                    merge_boxes=args.GT_type == "merged",
                )
            )
        if "phrasecut" in dataset_name:
            evaluator_list.append(
                PhrasecutEvaluator(
                    "test" if args.test else "miniv",
                    ann_folder=args.phrasecut_orig_ann_path,
                    output_dir=os.path.join(output_dir, "phrasecut_eval"),
                    eval_mask=args.masks,
                )
            )
        return evaluator_list

    # Runs only evaluation, by default on the validation set unless --test is passed.
    if args.eval and not args.co_training:  # TODO: eval mode для co_training
        test_stats = {}
        test_model = model_ema if model_ema is not None else model
        for i, item in enumerate(val_tuples):
            evaluator_list = build_evaluator_list(item.base_ds, item.dataset_name)
            postprocessors = build_postprocessors(args, item.dataset_name)
            item = item._replace(evaluator_list=evaluator_list)
            print(f"Evaluating {item.dataset_name}")
            curr_test_stats = evaluate(
                model=test_model,
                criterion=criterion,
                contrastive_criterion=contrastive_criterion,
                qa_criterion=qa_criterion,
                postprocessors=postprocessors,
                weight_dict=weight_dict,
                data_loader=item.dataloader,
                evaluator_list=item.evaluator_list,
                device=device,
                args=args,
            )
            test_stats.update({item.dataset_name + "_" + k: v for k, v in curr_test_stats.items()})

        log_stats = {
            **{f"test_{k}": v for k, v in test_stats.items()},
            "n_parameters": n_parameters,
        }
        print(log_stats)
        return

    # Runs training and evaluates after every --eval_skip epochs
    print("Start training")
    start_time = time.time()
    best_metric = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.epoch_chunks > 0:
            assert len(samplers_vqa_train) == len(samplers_lvis_train), "Number of chunks must be equal!"
            sampler_vqa_train = samplers_vqa_train[epoch % len(samplers_vqa_train)]
            data_loader_vqa_train = data_loaders_vqa_train[epoch % len(data_loaders_vqa_train)]
            sampler_lvis_train = samplers_lvis_train[epoch % len(samplers_lvis_train)]
            data_loader_lvis_train = data_loaders_lvis_train[epoch % len(data_loaders_lvis_train)]
            print(f"Starting epoch {epoch // len(data_loaders_vqa_train)}, sub_epoch {epoch % len(data_loaders_vqa_train)}")
        else:
            print(f"Starting epoch {epoch}")
        if args.distributed:
            sampler_vqa_train.set_epoch(epoch)
            sampler_lvis_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            contrastive_criterion=contrastive_criterion,
            qa_criterion=qa_criterion,
            data_loader_vqa=data_loader_vqa_train,
            data_loader_zsOD=data_loader_lvis_train,
            weight_dict=weight_dict,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
            max_norm=args.clip_max_norm,
            model_ema=model_ema,
        )

        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            # extra checkpoint before LR drop and every 2 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 2 == 0:
                checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                dist.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "model_ema": model_ema.state_dict() if args.ema else None,
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )

        if epoch % args.eval_skip == 0:
            test_stats = {}
            test_model = model_ema if model_ema is not None else model

            if args.co_training:
                vqa_val_dataset = val_tuples[0]
                lvis_val_dataset = val_tuples[1]
                # Список оценщиков и построцессоров нужен только для lvis
                evaluator_list = build_evaluator_list(lvis_val_dataset.base_ds, lvis_val_dataset.dataset_name)
                postprocessors = build_postprocessors(args, lvis_val_dataset.dataset_name)

                print("Evaluating vqa2 and modulated_lvis simultaneously")
                test_stats = evaluate(
                    model=test_model,
                    criterion=criterion,
                    contrastive_criterion=contrastive_criterion,
                    qa_criterion=qa_criterion,
                    postprocessors=postprocessors,
                    weight_dict=weight_dict,
                    data_loader_vqa=data_loader_vqa_train,
                    data_loader_zsOD=data_loader_lvis_train,
                    evaluator_list=evaluator_list,
                    device=device,
                    args=args
                )
            else:
                for i, item in enumerate(val_tuples):
                    evaluator_list = build_evaluator_list(item.base_ds, item.dataset_name)
                    item = item._replace(evaluator_list=evaluator_list)
                    postprocessors = build_postprocessors(args, item.dataset_name)
                    print(f"Evaluating {item.dataset_name}")
                    curr_test_stats = evaluate(
                        model=test_model,
                        criterion=criterion,
                        contrastive_criterion=contrastive_criterion,
                        qa_criterion=qa_criterion,
                        postprocessors=postprocessors,
                        weight_dict=weight_dict,
                        data_loader=item.dataloader,
                        evaluator_list=item.evaluator_list,
                        device=device,
                        args=args,
                    )
                    test_stats.update({item.dataset_name + "_" + k: v for k, v in curr_test_stats.items()})
        else:
            test_stats = {}

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and dist.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if epoch % args.eval_skip == 0:
            if args.do_qa:  # TODO: make this more general: dataset name before key is needed if not co_training
                if "gqa" in args.combine_datasets:
                    if args.co_training:
                        metric = test_stats["accuracy_answer_total_unscaled"]
                    else:
                        metric = test_stats["gqa_accuracy_answer_total_unscaled"]
                else:
                    if args.co_training:
                        metric = test_stats["accuracy_answer_total_unscaled"]
                    else:
                        metric = test_stats["vqa2_accuracy_answer_total_unscaled"]
            else:
                metric = np.mean([v[1] for k, v in test_stats.items() if "coco_eval_bbox" in k])

            if args.output_dir and metric > best_metric:
                best_metric = metric
                checkpoint_paths = [output_dir / "BEST_checkpoint.pth"]
                # extra checkpoint before LR drop and every 100 epochs
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(
                        {
                            "model": model_without_ddp.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "args": args,
                        },
                        checkpoint_path,
                    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DETR training and evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
