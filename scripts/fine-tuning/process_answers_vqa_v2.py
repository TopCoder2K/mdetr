import json
from pathlib import Path
import argparse
from tqdm import tqdm
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser('Conversion script')

    parser.add_argument(
        '--fine-tune_path',
        default="",
        type=str,
        help='Path to the processed annotations for the VQA v2 dataset',
    )

    parser.add_argument(
        '--thresh_answerword',
        default=3,
        type=int,
        help='A threshold for answers when building vocabulary',
    )

    return parser.parse_args()

# Makes answer2id.json and answer2id_by_type.json analogously to gqa.
def main(args):
    finetune_path = Path(args.fine_tune_path)

    # As there are a lot of answers, we should keep only the most popular
    ans_freqs = Counter()
    for split_type in tqdm(['train', 'minival', 'trainval']):
        with open(finetune_path / f'finetune_vqa2_{split_type}.json') as f:
            split_info =  json.load(f)
            images = split_info['images']
            for image_info in images:
                answer = image_info['answer']
                ans_freqs[answer] += 1
    print('Answers frequencies were calculated!')

    # Building the mapping from an answer to the index
    ans2id = {}
    ans2id_by_type = {}
    indexes = {'global': 0, 'yes/no': 0, 'number': 0, 'other': 0}
    for split_type in tqdm(['train', 'minival', 'trainval']):
        with open(finetune_path / f'finetune_vqa2_{split_type}.json') as f:
            split_info = json.load(f)
            images = split_info['images']
            for image_info in images:
                answer = image_info['answer']
                if ans_freqs[answer] >= args.thresh_answerword:
                    if answer not in ans2id:
                        ans2id[answer] = indexes['global']
                        indexes['global'] += 1
                    answer_type = image_info['answer_type']
                    if answer_type not in ans2id_by_type:
                        ans2id_by_type[answer_type] = {}
                    if answer not in ans2id_by_type[answer_type]:
                        ans2id_by_type[answer_type][answer] = indexes[answer_type]
                        indexes[answer_type] += 1

    # Add 'unknown' answer
    ans2id['unknown'] = indexes['global']
    ans2id_by_type['yes/no']['unknown'] = indexes['yes/no']
    ans2id_by_type['number']['unknown'] = indexes['number']
    ans2id_by_type['other']['unknown'] = indexes['other']

    with open(finetune_path / 'vqa2_answer2id.json', 'w') as f:
        json.dump(dict(sorted(ans2id.items(), key=lambda item: item[1])), f)
    with open(finetune_path / 'vqa2_answer2id_by_type.json', 'w') as f:
        for key in ans2id_by_type.keys():
            ans2id_by_type[key] = dict(sorted(ans2id_by_type[key].items(),
                                              key=lambda item: item[1]))
        json.dump(ans2id_by_type, f)

    print('vqa2_answer2id and vqa2_answer2id_by_type files have been created!')

if __name__ == "__main__":
    main(parse_args())