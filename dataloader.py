import os
import json
import random
import datasets
import pandas as pd

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


def convert(data):
    for i, item in enumerate(data):
        new_item = {}
        new_item['raw'] = item['title'] + item['text']
        new_item['label'] = item['label']
        new_item['text_len'] = item['text_len']
        data[i] = new_item


def load_true_few_shot_dataset(args):

    labels = []
    with open(os.path.join(args.data_path, 'TextClassification', args.dataset, 'classes.txt')) as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) > 0:
                labels.append(line)

    if args.dataset == 'agnews':
        test_df = pd.read_csv(os.path.join(args.data_path, 'TextClassification', args.dataset, 'test.csv'), header=None)
        test_data = []
        for i in range(test_df.shape[0]):
            line = test_df.loc[i]
            label_id, title, text = list(line)
            label_id -= 1
            # raw = title + '. ' + text
            # text = text[:200]
            text = ' '.join(text.split()[:120])
            item = {
                'title': title,
                'text': text,
                'label': str(label_id),
                'text_len': len(text)
            }
            test_data.append(item)

        convert(test_data)

        if os.path.exists(
                os.path.join(args.data_path, 'TextClassification', args.dataset, str(args.k_shot) + 'shot.txt')):
            episodes = []
            path = os.path.join(args.data_path, 'TextClassification', args.dataset, str(args.k_shot) + 'shot.txt')
            with open(path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    episode = json.loads(line)
                    convert(episode['support_set'])
                    episode['query_set'] = test_data
                    episode['labels'] = labels
                    episodes.append(episode)

            return episodes

        train_df = pd.read_csv(os.path.join(args.data_path, 'TextClassification', args.dataset, 'train.csv'), header=None)
        train_data_dict = {i: [] for i in range(len(labels))}
        for i in range(train_df.shape[0]):
            line = train_df.loc[i]
            label_id, title, text = list(line)
            label_id -= 1
            # raw = title + '. ' + text
            # text = text[:200]
            text = ' '.join(text.split()[:120])
            item = {
                'title': title,
                'text': text,
                'label': str(label_id),
                'text_len': len(text)
            }
            train_data_dict[label_id].append(item)

    elif args.dataset == 'yahoo_answers_topics':
        path = os.path.join(args.data_path, 'TextClassification', 'yahoo_answers_topics')
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
        test_data = []
        with open(os.path.join(args.data_path, 'TextClassification', args.dataset, 'test.txt')) as sample_file:
            with open(os.path.join(args.data_path, 'TextClassification', args.dataset, 'test_labels.txt')) as label_file:
                for raw in sample_file.readlines():
                    raw, label_id = raw.strip(), int(label_file.readline().strip())
                    # raw = raw[:200]
                    raw = ' '.join(raw.split()[:120])
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

        train_data_dict = {i: [] for i in range(len(labels))}
        with open(os.path.join(args.data_path, 'TextClassification', args.dataset, 'train.txt')) as sample_file:
            with open(os.path.join(args.data_path, 'TextClassification', args.dataset, 'train_labels.txt')) as label_file:
                for raw in sample_file.readlines():
                    raw, label_id = raw.strip(), int(label_file.readline().strip())
                    # raw = raw[:200]
                    raw = ' '.join(raw.split()[:120])
                    item = {
                        'raw': raw,
                        'label': label_id,
                        'text_len': len(raw)
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
        with open(os.path.join(args.data_path, 'TextClassification', args.dataset, str(args.k_shot) + 'shot.txt'), 'a') as f:
            string = json.dumps(episode_to_save)
            f.write(string)
            f.write('\n')

        if args.dataset == 'agnews':
            convert(train_data)
        episode = {
            'support_set': train_data,
            'query_set': test_data,
            'labels': labels
        }
        episodes.append(episode)


    return episodes
