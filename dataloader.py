import os
import json
import random
import datasets
import pandas as pd
from datasets import load_dataset

from transformers import BertForMaskedLM, BertConfig, BertTokenizer


TASK_CLASSES = {
    'agnews': {
        'metric_prompt': [lambda text1, text2: [text1] + ['[SEP]', 'A', 'news', 'of', '[MASK]', 'topic', ':'] + [text2]],
    },
    'yahoo_answers_topics': {
        'metric_prompt': [lambda text1, text2: [text1] + ['[SEP]', 'A', 'news', 'of', '[MASK]', 'topic', ':'] + [text2]],
    },
    'dbpedia': {
        'metric_prompt': [lambda text1, text2: [text1] + ['[SEP]', 'A', 'news', 'of', '[MASK]', 'topic', ':'] + [text2]],
    },
}

MODEL_CLASSES = {
    'bert': {
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        'mlm': BertForMaskedLM
    },
}


def load_true_few_shot_dataset(args):

    labels = []
    with open(os.path.join(args.data_path, 'classes', args.dataset, 'classes.txt')) as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) > 0:
                labels.append(line)

    os.makedirs(os.path.join(args.data_path, 'TextClassification', args.dataset), exist_ok=True)

    if args.dataset == 'agnews':
        dataset = load_dataset('fancyzhx/ag_news')
        test_dataset = dataset.data['test'].table
        test_df = test_dataset.to_pandas()
        test_data = []
        for i in range(test_df.shape[0]):
            line = test_df.loc[i]
            label_id, text = line['label'], line['text']
            text = ' '.join(text.split()[:120])
            item = {
                'raw': text,
                'label': str(label_id),
                'text_len': len(text)
            }
            test_data.append(item)

        if os.path.exists(
                os.path.join(args.data_path, 'TextClassification', args.dataset, str(args.k_shot) + 'shot.txt')):
            episodes = []
            path = os.path.join(args.data_path, 'TextClassification', args.dataset, str(args.k_shot) + 'shot.txt')
            with open(path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    episode = json.loads(line)
                    episode['query_set'] = test_data
                    episode['labels'] = labels
                    episodes.append(episode)

            return episodes

        train_dataset = dataset.data['train'].table
        train_df = train_dataset.to_pandas()
        train_data_dict = {i: [] for i in range(len(labels))}
        for i in range(train_df.shape[0]):
            line = train_df.loc[i]
            label_id, text = line['label'], line['text']
            text = ' '.join(text.split()[:120])
            item = {
                'raw': text,
                'label': str(label_id),
                'text_len': len(text)
            }
            train_data_dict[label_id].append(item)

    elif args.dataset == 'yahoo_answers_topics':
        # path = os.path.join(args.data_path, 'TextClassification', 'yahoo_answers_topics')
        dataset = load_dataset('community-datasets/yahoo_answers_topics')
        dataset = datasets.load_from_disk(dataset_path=path)
        test_dataset = dataset.data['test'].table.columns
        test_data = []
        for i in range(len(test_dataset[0])):
            question_title = str(test_dataset[2][i]).strip()
            question_content = str(test_dataset[3][i]).strip()
            answer = str(test_dataset[4][i]).strip()
            raw = question_title + question_content + answer
            raw = ' '.join(raw.split()[:120])

            label_id = test_dataset[1][i].as_py()

            item = {
                'raw': raw,
                'label': label_id,
                'text_len': len(raw)
            }
            test_data.append(item)

        if os.path.exists(
                os.path.join(args.data_path, 'TextClassification', args.dataset, str(args.k_shot) + 'shot.txt')):
            episodes = []
            with open(os.path.join(args.data_path, 'TextClassification', args.dataset, str(args.k_shot) + 'shot.txt'),
                      'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    episode = json.loads(line)
                    episode['query_set'] = test_data
                    episode['labels'] = labels
                    episodes.append(episode)

            return episodes

        train_dataset = dataset.data['train'].table.columns
        train_data_dict = {i: [] for i in range(len(labels))}
        for i in range(len(train_dataset[0])):
            question_title = str(train_dataset[2][i]).strip()
            question_content = str(train_dataset[3][i]).strip()
            answer = str(train_dataset[4][i]).strip()
            raw = question_title + question_content + answer
            raw = ' '.join(raw.split()[:120])

            label_id = train_dataset[1][i].as_py()

            item = {
                'raw': raw,
                'label': label_id,
                'text_len': len(raw)
            }
            train_data_dict[label_id].append(item)

    else:
        dataset = load_dataset('fancyzhx/dbpedia_14')
        test_dataset = dataset.data['test'].table
        test_df = test_dataset.to_pandas()
        test_data = []
        for i in range(test_df.shape[0]):
            line = test_df.loc[i]
            label_id, text = line['label'], line['content']
            text = ' '.join(text.split()[:120])
            item = {
                'raw': text,
                'label': str(label_id),
                'text_len': len(text)
            }
            test_data.append(item)

        if os.path.exists(
                os.path.join(args.data_path, 'TextClassification', args.dataset, str(args.k_shot) + 'shot.txt')):
            episodes = []
            with open(os.path.join(args.data_path, 'TextClassification', args.dataset, str(args.k_shot) + 'shot.txt'),
                      'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    episode = json.loads(line)
                    episode['query_set'] = test_data
                    episode['labels'] = labels
                    episodes.append(episode)

            return episodes

        train_dataset = dataset.data['train'].table
        train_df = train_dataset.to_pandas()
        train_data_dict = {i: [] for i in range(len(labels))}
        for i in range(train_df.shape[0]):
            line = train_df.loc[i]
            label_id, text = line['label'], line['content']
            text = ' '.join(text.split()[:120])
            item = {
                'raw': text,
                'label': str(label_id),
                'text_len': len(text)
            }
            train_data_dict[label_id].append(item)

    episodes = []

    for _ in range(10):
        train_data = []
        for i in range(len(labels)):
            this_samples = random.sample(train_data_dict[i], args.k_shot)
            train_data = train_data + this_samples

        episode_to_save = {
            'support_set': train_data,
        }
        with open(os.path.join(args.data_path, 'TextClassification', args.dataset, str(args.k_shot) + 'shot.txt'), 'w') as f:
            string = json.dumps(episode_to_save)
            f.write(string)
            f.write('\n')

        episode = {
            'support_set': train_data,
            'query_set': test_data,
            'labels': labels
        }
        episodes.append(episode)


    return episodes
