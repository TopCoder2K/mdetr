# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MDETR model and criterion classes.
"""
from typing import Dict, Optional

import torch
import torch.distributed
import torch.nn.functional as F
from torch import nn

import util.dist as dist
from util import box_ops
from util.metrics import accuracy
from util.misc import NestedTensor, interpolate

from .backbone import build_backbone
from .postprocessors import build_postprocessors
from .transformer import build_transformer
from .transformer_decoder_fusing import build_transformer_with_decoder_fusing


class MDETRDecoderFusing(nn.Module):
    """ This is the MDETR module that performs modulated object detection """

    def __init__(
        self,
        backbone,
        transformer,
        num_classes,
        num_queries,
        contrastive_hdim=64,
        qa_dataset: Optional[str] = None,
        split_qa_heads=True,
        predict_final=False,
    ):
        """Initializes the model.

        Args:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         MDETR can detect in a single image. For COCO, we recommend 100 queries.
            contrastive_hdim: dimension used for projecting the embeddings before computing contrastive loss
            qa_dataset: If not None, train a QA head for the target dataset (CLEVR or GQA or VQA2)
            split_qa_heads: If true, use several head for each question type
            predict_final: If true, will predict if a given box is in the actual referred set.
                           Useful for CLEVR-Ref+ only currently.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.isfinal_embed = nn.Linear(hidden_dim, 1) if predict_final else None
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if qa_dataset is not None:
            nb_heads = 6 if qa_dataset == "gqa" else 4
            self.qa_embed = nn.Embedding(nb_heads if split_qa_heads else 1, hidden_dim)

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

        self.qa_dataset = qa_dataset
        self.split_qa_heads = split_qa_heads
        if qa_dataset is not None:
            if split_qa_heads:
                self.answer_type_head = nn.Linear(hidden_dim, 5)
                # TODO: make this more general
                if qa_dataset == "gqa":
                    self.answer_rel_head = nn.Linear(hidden_dim, 1594)
                    self.answer_obj_head = nn.Linear(hidden_dim, 3)
                    self.answer_global_head = nn.Linear(hidden_dim, 111)
                    self.answer_attr_head = nn.Linear(hidden_dim, 403)
                    self.answer_cat_head = nn.Linear(hidden_dim, 678)
                elif qa_dataset == "clevr":
                    self.answer_type_head = nn.Linear(hidden_dim, 3)
                    self.answer_binary_head = nn.Linear(hidden_dim, 1)
                    self.answer_attr_head = nn.Linear(hidden_dim, 15)
                    self.answer_reg_head = MLP(hidden_dim, hidden_dim, 20, 3)
                elif qa_dataset == "vqa2":
                    # All values were taken from the appropriate json file.
                    self.answer_type_head = nn.Linear(hidden_dim, 3)
                    self.answer_yesno_head = nn.Linear(hidden_dim, 3)
                    self.answer_number_head = nn.Linear(hidden_dim, 221)
                    self.answer_other_head = nn.Linear(hidden_dim, 1742)
                else:
                    assert False, f"Invalid qa dataset {qa_dataset}"
            else:
                # TODO: make this more general
                assert qa_dataset == "gqa" or qa_dataset == "vqa2", "Clevr QA is not supported with unified head"
                if qa_dataset == "gqa":
                    self.answer_head = nn.Linear(hidden_dim, 1853)
                elif qa_dataset == "vqa2":
                    self.answer_head = nn.Linear(hidden_dim, 1944)

    def forward(self, samples: NestedTensor, captions, encode_and_save=True, memory_cache=None, dset_name=None):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        It returns a dict with the following elements:
           - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
           - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                           (center_x, center_y, height, width). These values are normalized in [0, 1],
                           relative to the size of each individual image (disregarding possible padding).
                           See PostProcess for information on how to retrieve the unnormalized bounding box.
           - "aux_outputs": Optional, only returned when auxiliary losses are activated. It is a list of
                            dictionaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = NestedTensor.from_tensor_list(samples)

        if encode_and_save:
            assert memory_cache is None
            # print(f'NestedTensor.tensors for backbone: {samples.tensors.shape}')
            features, pos = self.backbone(samples)
            src, mask = features[-1].decompose()
            query_embed = self.query_embed.weight
            if self.qa_dataset is not None:
                query_embed = torch.cat([query_embed, self.qa_embed.weight], 0)
            memory_cache = self.transformer(
                self.input_proj(src),
                mask,
                query_embed,
                pos[-1],
                captions,
                encode_and_save=True,
                text_memory=None,
                img_memory=None,
                text_attention_mask=None,
                dset_name=None  # dataset is not important
            )

            return memory_cache

        else:
            assert memory_cache is not None
            # vqa_hs.shape = (num_layers=6, b, num_queries + nb_heads, 256)
            # zsOD_hs.shape = (num_layers=6, b, num_queries, 256)
            hs = self.transformer(
                mask=memory_cache["mask"],
                query_embed=memory_cache["query_embed"],
                pos_embed=memory_cache["pos_embed"],
                encode_and_save=False,
                text_memory=memory_cache["text_memory_resized"],
                img_memory=memory_cache["img_memory"],
                text_attention_mask=memory_cache["text_attention_mask"],
                dset_name=dset_name
            )

            vqa_hs = None
            zsOD_hs = None
            if dset_name == "vqa2":
                vqa_hs = hs
            elif dset_name == "zsOD":
                zsOD_hs = hs

            out = {}

            if dset_name == "vqa2":
                if self.qa_dataset is not None:
                    if self.split_qa_heads:
                        if self.qa_dataset == "gqa":
                            answer_embeds = vqa_hs[0, :, -6:]
                            vqa_hs = vqa_hs[:, :, :-6]
                            out["pred_answer_type"] = self.answer_type_head(answer_embeds[:, 0])
                            out["pred_answer_obj"] = self.answer_obj_head(answer_embeds[:, 1])
                            out["pred_answer_rel"] = self.answer_rel_head(answer_embeds[:, 2])
                            out["pred_answer_attr"] = self.answer_attr_head(answer_embeds[:, 3])
                            out["pred_answer_cat"] = self.answer_cat_head(answer_embeds[:, 4])
                            out["pred_answer_global"] = self.answer_global_head(answer_embeds[:, 5])
                        elif self.qa_dataset == "clevr":
                            answer_embeds = vqa_hs[0, :, -4:]
                            vqa_hs = vqa_hs[:, :, :-4]
                            out["pred_answer_type"] = self.answer_type_head(answer_embeds[:, 0])
                            out["pred_answer_binary"] = self.answer_binary_head(answer_embeds[:, 1]).squeeze(-1)
                            out["pred_answer_reg"] = self.answer_reg_head(answer_embeds[:, 2])
                            out["pred_answer_attr"] = self.answer_attr_head(answer_embeds[:, 3])
                        elif self.qa_dataset == "vqa2":
                            answer_embeds = vqa_hs[0, :, -4:]
                            vqa_hs = vqa_hs[:, :, :-4]
                            out["pred_answer_type"] = self.answer_type_head(answer_embeds[:, 0])
                            out["pred_answer_yes/no"] = self.answer_yesno_head(answer_embeds[:, 1])
                            out["pred_answer_number"] = self.answer_number_head(answer_embeds[:, 2])
                            out["pred_answer_other"] = self.answer_other_head(answer_embeds[:, 3])
                        else:
                            assert False, f"Invalid qa dataset {self.qa_dataset}"

                    else:
                        answer_embeds = vqa_hs[0, :, -1]
                        vqa_hs = vqa_hs[:, :, :-1]
                        out["pred_answer"] = self.answer_head(answer_embeds)

                return out  # Только ответы на вопросы нужны, детекция не нужна

            else:
                # class_embed.weight.shape = (d_model=256, num_classes+1=255+1)
                # output_class.shape = (num_layers=6, b, num_queries=100, num_classes+1=256)
                outputs_class = self.class_embed(zsOD_hs)
                # bbox_embed(hs).shape = (num_layers=6, b, num_queries=100, 4)
                outputs_coord = self.bbox_embed(zsOD_hs).sigmoid()

                out.update(
                    {
                        "pred_logits": outputs_class[-1],
                        "pred_boxes": outputs_coord[-1],
                    }
                )

                # outputs_isfinal = None
                # if self.isfinal_embed is not None:
                #     outputs_isfinal = self.isfinal_embed(hs)
                #     out["pred_isfinal"] = outputs_isfinal[-1]

                return out  # Аналогично, нужна только детекция


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN) """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_for_fb_with_fusing(args):
    num_classes = 255
    # device = torch.device(args.device)

    assert not args.masks or args.mask_model != "none"

    backbone = build_backbone(args)
    transformer = build_transformer_with_decoder_fusing(args)
    model = MDETRDecoderFusing(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        qa_dataset="vqa2",  # TODO: make this more general
        split_qa_heads=args.split_qa_heads,
        # predict_final=args.predict_final,
    )
    # if args.mask_model != "none":
    #     model = DETRsegm(
    #         model,
    #         mask_head=args.mask_model,
    #         freeze_detr=(args.frozen_weights is not None),
    #     )
    # matcher = build_matcher(args)

    return model
