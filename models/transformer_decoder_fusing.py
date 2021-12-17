# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import RobertaModel, RobertaTokenizerFast
from .transformer import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer, FeatureResizer


class TransformerDecoderFusing(nn.Module):
    def __init__(
            self,
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=False,
            pass_pos_and_query=True,
            text_encoder_type="roberta-base",
            freeze_text_encoder=False,
            contrastive_loss=False,
    ):
        super().__init__()

        self.pass_pos_and_query = pass_pos_and_query
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder1 = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec
        )  # For the VQA2 setting
        self.decoder2 = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec
        )  # For the zsOD setting

        self.CLS = nn.Embedding(1, d_model) if contrastive_loss else None

        self._reset_parameters()

        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            text_encoder_type,
            cache_dir="./checkpoints/"
        )
        self.text_encoder = RobertaModel.from_pretrained(
            text_encoder_type,
            cache_dir="./checkpoints/"
        )

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        self.expander_dropout = 0.1
        config = self.text_encoder.config
        self.resizer = FeatureResizer(
            input_feat_size=config.hidden_size,
            output_feat_size=d_model,
            dropout=self.expander_dropout,
        )

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
            self,
            src=None,
            mask=None,
            query_embed=None,
            pos_embed=None,
            text=None,
            encode_and_save=True,
            text_memory=None,
            img_memory=None,
            text_attention_mask=None,
            dset_name=None
    ):
        if encode_and_save:
            # flatten NxCxHxW to HWxNxC
            bs, c, h, w = src.shape
            # src.shape = (h * w, bs, d_model=256)
            src = src.flatten(2).permute(2, 0, 1)
            device = src.device
            # pos_embed.shape = (h * w, bs, 2*num_pos_feats=256)
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
            # query_embed.shape = (num_queries + nb_heads, bs, 256)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            # mask.shape = (b, h * w)
            mask = mask.flatten(1)

            if self.CLS is not None:
                # We add a CLS token to the image, to be used for contrastive loss

                CLS = self.CLS.weight.view(1, 1, -1).repeat(1, bs, 1)
                # Add the CLS token to the incoming features
                src = torch.cat((CLS, src))

                # Adding zeros as the first token in the sequence to be compatible with the CLS token
                pos_embed = torch.cat((torch.zeros(1, bs, self.d_model, device=device), pos_embed))

                # Adding one mask item to the beginning of the mask to be compatible with CLS token
                cls_pad = torch.zeros(bs, 1).bool().to(device)
                mask = torch.cat((cls_pad, mask), dim=1)

            if self.pass_pos_and_query:
                tgt = torch.zeros_like(query_embed)
            else:
                src, tgt, query_embed, pos_embed = src + 0.1 * pos_embed, query_embed, None, None

            device = src.device
            if isinstance(text[0], str):
                # Encode the text
                # tokenized --- dict, у которого два поля: input_inds, shape = (b, longest + 2);
                # attention_mask, shape = (b, longest + 2) (зануляет padded)
                tokenized = self.tokenizer.batch_encode_plus(text, padding="longest", return_tensors="pt").to(device)
                # encoded_text --- BaseModelOutputWithPoolingAndCrossAttentions,
                # у которого в данном случае два поля: last_hidden_state, pooler_output
                encoded_text = self.text_encoder(**tokenized)
                # print(f'Text features shape: {encoded_text.last_hidden_state.shape}')

                # Transpose memory because pytorch's attention expects sequence first
                text_memory = encoded_text.last_hidden_state.transpose(0, 1)
                # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
                # text_attention_mask.shape = (b, seq_length=max_seq_len+2)
                text_attention_mask = tokenized.attention_mask.ne(1).bool()

                # Resize the encoder hidden states to be of the same d_model as the decoder
                # text_memory_resized.shape = (seq_length, bs, d_model=256)
                text_memory_resized = self.resizer(text_memory)
            else:
                # The text is already encoded, use as is.
                text_attention_mask, text_memory_resized, tokenized = text

            # Concat on the sequence dimension
            # src.shape = (h * w + seq_length, b, d_model=256)
            src = torch.cat([src, text_memory_resized], dim=0)
            # For mask, sequence dimension is second
            # mask.shape = (b, h * w + seq_length)
            mask = torch.cat([mask, text_attention_mask], dim=1)
            # Pad the pos_embed with 0 so that the addition will be a no-op for the text tokens
            # pos_embed.shape = (h * w + seq_length, b, 2*num_pos_feats=256)
            pos_embed = torch.cat([pos_embed, torch.zeros_like(text_memory_resized)], dim=0)

            img_memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

            text_memory = img_memory[-len(text_memory_resized):]

            assert img_memory.shape[1] == text_memory.shape[1] == tgt.shape[1]
            memory_cache = {
                "text_memory_resized": text_memory_resized,
                "text_memory": text_memory,
                "img_memory": img_memory,
                "text_pooled_op": encoded_text.pooler_output if self.CLS is not None else None,
                "img_pooled_op": img_memory[0] if self.CLS is not None else None,  # Return the CLS token
                "mask": mask,
                "text_attention_mask": text_attention_mask,
                "pos_embed": pos_embed,
                "query_embed": query_embed,
                "tokenized": tokenized,
            }
            return memory_cache

        else:
            if self.pass_pos_and_query:
                # tgt.shape = (num_queries + nb_heads, b, 256)
                tgt = torch.zeros_like(query_embed)
            else:
                src, tgt, query_embed, pos_embed = src + 0.1 * pos_embed, query_embed, None, None

            assert img_memory.shape[1] == text_memory.shape[1] == tgt.shape[1]

            # pos_embed.shape = (h*w + seq_length, b, 2*num_pos_feats=256)
            # query_embed.shape = (num_queries + nb_heads, b, 256)
            if dset_name == "vqa2":
                hs = self.decoder1(
                    tgt,
                    img_memory,
                    text_memory,
                    memory_key_padding_mask=mask,
                    text_memory_key_padding_mask=text_attention_mask,
                    pos=pos_embed,
                    query_pos=query_embed,
                )
            elif dset_name == "zsOD":
                hs = self.decoder2(
                    tgt,
                    img_memory,
                    text_memory,
                    memory_key_padding_mask=mask,
                    text_memory_key_padding_mask=text_attention_mask,
                    pos=pos_embed,
                    query_pos=query_embed,
                )
            else:
                assert False, f"Decoders can be fused only for vqa2 and zsOD, but got {dset_name}"

            # hs.shape = (num_layers=6, num_queries + nb_heads, b, 256)
            # hs.transpose(1, 2).shape = (num_layers=6, b, num_queries + nb_heads, 256)
            return hs.transpose(1, 2)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def build_transformer_with_decoder_fusing(args):
    # In case of inference no loss is needed
    return TransformerDecoderFusing(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        pass_pos_and_query=args.pass_pos_and_query,
        text_encoder_type=args.text_encoder_type,
        freeze_text_encoder=args.freeze_text_encoder,
    )

    # return Transformer(
    #     d_model=args.hidden_dim,
    #     dropout=args.dropout,
    #     nhead=args.nheads,
    #     dim_feedforward=args.dim_feedforward,
    #     num_encoder_layers=args.enc_layers,
    #     num_decoder_layers=args.dec_layers,
    #     normalize_before=args.pre_norm,
    #     return_intermediate_dec=True,
    #     pass_pos_and_query=args.pass_pos_and_query,
    #     text_encoder_type=args.text_encoder_type,
    #     freeze_text_encoder=args.freeze_text_encoder,
    #     contrastive_loss=args.contrastive_loss,
    # )
