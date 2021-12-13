import argparse
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser("Conversion script")

    parser.add_argument(
        "--data_path",
        default="./input/zsOD",
        type=str,
        help="Path to the Fusion Brain zsOD dataset.",
    )

    parser.add_argument(
        "--out_path",
        default="./input/zsOD",
        type=str,
        help="Path where to export the resulting dataset.",
    )

    parser.add_argument(
        "--use_translated",
        action="store_true",
        help="Whether to use translated versions of the questions. Works only with --fusion_brain option"
    )

    return parser.parse_args()


def convert(data_path, output_path, use_translated=False):
    translated_requests_path = data_path / "translated_requests.json"
    translated_requests = None

    with open(data_path / "requests.json") as f:
        requests = json.load(f)

    # If questions have been already translated and were enabled, use them
    if use_translated and translated_requests_path.is_file():
        print("Using translated version of the requests")
        with open(translated_requests_path) as f:
            translated_requests = json.load(f)
    else:
        print("Using NOT translated version of the requests!")

    target_requests = translated_requests if not translated_requests is None else requests

    categories = [{"supercategory": "object", "id": 1, "name": "object"}]
    annotations = []
    images = []
    d_name = "zsOD"
    next_img_id = 0

    for filename, img_requests in target_requests.items():
        for i, obj_to_detect in enumerate(img_requests):

            if use_translated and translated_requests_path.is_file():
                # Тогда obj_to_detect --- список из двух элементов: объект и язык. Кстати, такое кодирование, кажется,
                # избыточно, так как язык далее не используем и при переводе обратно тоже (версии tricky)
                cur_img = {
                    "file_name": filename,
                    "height": None,  # В output вносится orig_size и size, эти не используются
                    "width": None,  # В output вносится orig_size и size, эти не используются
                    "id": next_img_id,
                    "original_id": None,  # Кажется, это не используется
                    "coco_url": None,  # Кажется, не нужно. Это использовалось для загрузки и получения имени coco imgs
                    "caption": obj_to_detect[0],
                    "tokens_negative": [(0, len(obj_to_detect[0]))],
                    "dataset_name": d_name,
                    "orig_caption": requests[filename][i]
                }
            else:
                # Тогда obj_to_detect строка с объектом
                cur_img = {
                    "file_name": filename,
                    "height": None,  # В output вносится orig_size и size, эти не используются
                    "width": None,  # В output вносится orig_size и size, эти не используются
                    "id": next_img_id,
                    "original_id": None,  # Кажется, это не используется
                    "coco_url": None,  # Кажется, не нужно.
                    "caption": obj_to_detect,
                    "tokens_negative": [(0, len(obj_to_detect))],
                    "dataset_name": d_name,
                    "orig_caption": obj_to_detect,
                }

            next_img_id += 1
            images.append(cur_img)

    ds = {"info": [], "licenses": [], "images": images, "annotations": annotations, "categories": categories}
    with open(output_path / f"inference_zsOD_fusion_brain.json", "w") as j_file:
        json.dump(ds, j_file)

    print("zsOD: COCO format was formed successfully!")
    return next_img_id, 0


def main(args):
    data_path = Path(args.data_path)
    output_path = Path(args.out_path)

    os.makedirs(str(output_path), exist_ok=True)

    convert(data_path, output_path, args.use_translated)


if __name__ == "__main__":
    main(parse_args())
