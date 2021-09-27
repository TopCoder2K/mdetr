import json
from pathlib import Path
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser('Conversion script')

    parser.add_argument(
        '--fine-tune_path',
        default="",
        type=str,
        help='Path to the processed annotations for the VQA v2 dataset',
    )

    return parser.parse_args()

# Makes answer2id.json and answer2id_by_type.json analogously to gqa.
def main(args):
    finetune_path = Path(args.fine_tune_path)
    ans2id = {}
    ans2id_by_type = {}
    indexes = {'global': 0, 'yes/no': 0, 'number': 0, 'other': 0}

    for split_type in tqdm(['train', 'minival', 'trainval']):
        with open(finetune_path / f'finetune_vqa2_{split_type}.json') as f:
            split_info =  json.load(f)
            images = split_info['images']
            for image_info in images:
                answer = image_info['answer']
                if answer not in ans2id:
                    ans2id[answer] = indexes['global']
                    indexes['global'] += 1

                answer_type = image_info['answer_type']
                if answer_type not in ans2id_by_type:
                    ans2id_by_type[answer_type] = {}
                if answer not in ans2id_by_type[answer_type]:
                    ans2id_by_type[answer_type][answer] = indexes[answer_type]
                    indexes[answer_type] += 1

    with open(finetune_path / 'vqa2_answer2id.json', 'w') as f:
        json.dump(ans2id, f)
    with open(finetune_path / 'vqa2_answer2id_by_type.json', 'w') as f:
        json.dump(ans2id_by_type, f)

if __name__ == "__main__":
    main(parse_args())
