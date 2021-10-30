# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .mdetr import build
from .mdetr import build_for_fb


def build_model(args):
    if args.inference:
        return build_for_fb(args)

    return build(args)
