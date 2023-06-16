import json
import jsonpickle
import os
from time import sleep
from typing import List, Dict, Optional
import copy
import random
import math

import torch
# torch.multiprocessing.set_start_method('spawn')
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score
from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing as mp
import pathos.multiprocessing as pathosmp
import multiprocess.context as ctx
ctx._force_start_method('spawn')
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler, Dataset
from tqdm import trange, tqdm
from transformers import InputExample, AdamW, get_linear_schedule_with_warmup
from transformers.data.metrics import simple_accuracy

from utils import tprint, exact_match
from dataloader import TASK_CLASSES, MODEL_CLASSES


class DictDataset(Dataset):
    """A dataset of tensors that uses a dictionary for key-value mappings"""

    def __init__(self, tensors):
        self.data = tensors

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.data.items()}

    def __len__(self):
        key0 = list(self.data.keys())[0]
        return len(self.data[key0])


class ContinuousPrompt(torch.nn.Module):
    def __init__(self, args, tokenizer):
        super(ContinuousPrompt, self).__init__()
        self.config = args
        self.tokenizer = tokenizer

        config_class = MODEL_CLASSES[self.config.pretrained_model]['config']
        pretrained_model_name_or_path = args.model_type
        model_config = config_class.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            use_cache=False)
        model_class = MODEL_CLASSES[self.config.pretrained_model]['mlm']
        self.model = model_class.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            config=model_config)


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None,
                output_hidden_states=False):
        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels,
                          token_type_ids=token_type_ids,
                          output_hidden_states=output_hidden_states)


class TransformerModelWrapper:
    """A wrapper around a Transformer-based language model."""

    def __init__(self, args):
        self.config = args

        tokenizer_class = MODEL_CLASSES[self.config.pretrained_model]['tokenizer']
        self.tokenizer = tokenizer_class.from_pretrained(
            pretrained_model_name_or_path=args.model_type)

        self.model = ContinuousPrompt(args, self.tokenizer)
        self.prompt_template = args.prompt_template


    @classmethod
    def generate_verbalizer(cls, label_map):
        verbalizer = {}
        except_words = ['the', '&', 'and', ]
        convert_words = {}
        for label in label_map.keys():
            answers = label.split()
            for word in except_words:
                if word in answers:
                    answers.remove(word)
            for word in convert_words.keys():
                if word in answers:
                    answers.remove(word)
                    answers.append(convert_words[word])
            answers = [' ' + answer.lower() for answer in answers]
            verbalizer[label] = answers
        return verbalizer


    def refresh_label_map(self, label_map):
        self.label_map = label_map
        self.verbalizer = self.generate_verbalizer(self.label_map)
        self.metric_label_map = {'relevant consistent similar': 0, 'irrelevant inconsistent different': 1}
        self.metric_verbalizer = self.generate_verbalizer(self.metric_label_map)
        self.metric_mlm_logits_to_answer_logits_tensor = self._build_mlm_logits_to_cls_logits_tensor()


    def save(self, path: str):
        tprint("Saving models.")
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

        model_to_save.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self._save_config(path)


    @classmethod
    def from_pretrained(cls, path: str):
        """Load a pretrained wrapper from a given path."""

        wrapper = TransformerModelWrapper.__new__(TransformerModelWrapper)
        wrapper.config = wrapper._load_config(path)

        tokenizer_class = MODEL_CLASSES[wrapper.config.pretrained_model]['tokenizer']
        wrapper.tokenizer = tokenizer_class.from_pretrained(path)

        wrapper.model = ContinuousPrompt(wrapper.config, wrapper.tokenizer)
        model_class = MODEL_CLASSES[wrapper.config.pretrained_model]['mlm']
        wrapper.model.model = model_class.from_pretrained(path)

        wrapper.model.cuda()
        return wrapper


    def _save_config(self, path: str):
        with open(os.path.join(path, 'wrapper_config.json'), 'w') as f:
            f.write(jsonpickle.encode(self.config))

    @staticmethod
    def _load_config(path: str):
        with open(os.path.join(path, 'wrapper_config.json'), 'r') as f:
            return jsonpickle.decode(f.read())


    def run_single_episode(self,
                            episode,
                            num,
                            k_shot: int = 5,
                            batch_size: int = 8,
                            n_adapt_epochs: int = 3,
                            gradient_accumulation_steps: int = 1,
                            weight_decay: float = 0.0,  #
                            lm_learning_rate: float = 1e-5,
                            adam_epsilon: float = 1e-8,
                            max_grad_norm: float = 1,
                            ):
        task_wrapper = self.__class__(self.config)
        label_map = {label: i for i, label in enumerate(episode['labels'])}
        task_wrapper.refresh_label_map(label_map)

        train_dataset, eval_dataset_paths, pivot_dataset = task_wrapper.build_dataset(episode, num)

        # Check if there exists a model trained under the same setting.
        output_dir = task_wrapper.config.output_dir + str(num) + 'set'
        os.makedirs(output_dir, exist_ok=True)
        pretrained_model_path = os.path.join(output_dir, 'wrapper_config.json')

        if os.path.exists(pretrained_model_path):
            task_wrapper = task_wrapper.from_pretrained(output_dir)
            task_wrapper.refresh_label_map(label_map)
            task_wrapper.config.pivot = self.config.pivot
        else:
            train_batch_size = batch_size
            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size, collate_fn=task_wrapper.collate_fn)

            t_total = n_adapt_epochs * len(train_dataloader)

            warmup_steps = int(t_total / 30)
            cur_model = task_wrapper.model.module if hasattr(task_wrapper.model, 'module') else task_wrapper.model
            optimizer, scheduler = task_wrapper.prepare_optimizer_scheduler(
                cur_model, t_total, weight_decay, lm_learning_rate, adam_epsilon, warmup_steps)

            task_wrapper.model.cuda()

            # Accelerated training with mixed precision
            scaler = torch.cuda.amp.GradScaler(init_scale=256)
            task_wrapper.model.zero_grad()
            for i in trange(n_adapt_epochs):
                for step, batch in enumerate(train_dataloader):
                    task_wrapper.model.train()
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        batch = {k: t.cuda() for k, t in batch.items()}
                        loss = task_wrapper.mlm_train_step(batch)

                        if gradient_accumulation_steps > 1:
                            loss = loss / gradient_accumulation_steps

                    try:
                        scaler.scale(loss).backward()
                    except Exception as e:
                        print(e)

                    if (step + 1) % gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(task_wrapper.model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        task_wrapper.model.zero_grad()
            task_wrapper.save(output_dir)

        task_wrapper.model.cuda()
        task_wrapper.model.eval()

        # Compute the representativeness score for each sample
        pivot_batch_size = batch_size
        pivot_sampler = RandomSampler(pivot_dataset)
        pivot_dataloader = DataLoader(pivot_dataset, sampler=pivot_sampler, batch_size=pivot_batch_size,
                                      collate_fn=task_wrapper.collate_fn)
        preds = None
        for step, batch in enumerate(pivot_dataloader):
            with torch.no_grad(), torch.inference_mode(), torch.cuda.amp.autocast():
                batch = {k: t.cuda() for k, t in batch.items()}
                logits = task_wrapper.mlm_eval_step(batch)
            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

        # positive_scores stands for intra-class relevance scores, while negative_scores is inter-class relevance scores
        representativeness = []
        for i in range(len(episode['support_set'])):
            offset = i * len(episode['support_set'])
            positive_scores = [preds[offset+j, 0] - preds[offset+j, 1] for j in range(self.config.k_shot)]
            negative_scores = [preds[offset+j, 0] - preds[offset+j, 1] for j in range(self.config.k_shot, len(episode['support_set']))]
            positive_score = sum(positive_scores) / len(positive_scores)
            negative_score = sum(negative_scores) / len(negative_scores)
            representativeness.append(positive_score - negative_score)

        # Select representative samples for each label
        pivot_data = []
        for i in range(len(episode['support_set']) // self.config.k_shot):
            this_label_representativeness = representativeness[i*self.config.k_shot: (i+1)*self.config.k_shot]
            this_label_representativeness = [(score, i) for i, score in enumerate(this_label_representativeness)]
            this_label_representativeness.sort(reverse=True)
            for j in range(self.config.pivot):
                pivot_data.append(episode['support_set'][i * self.config.k_shot + this_label_representativeness[j][1]])

        n_way = len(episode['support_set']) // k_shot
        if self.config.pivot > 0:
            pivot_test_dataset_paths = task_wrapper.build_pivot_dataset(pivot_data, episode['query_set'], num)
            eval_dataset_paths = pivot_test_dataset_paths
            episode['support_set'] = pivot_data

        # Accelerate with kernl
        if task_wrapper.config.kernl_accerleration != 0:
            from kernl.model_optimization import optimize_model
            optimize_model(task_wrapper.model.model)

        # Extract samples for the inference stage which are stored in separate files
        def extract_chunk_num(path):
            chunk_num_string = path[path.find('set_') + 4: path.find('chunk')]
            return int(chunk_num_string)

        eval_dataset_paths = [[extract_chunk_num(path), path] for path in eval_dataset_paths]
        eval_dataset_paths.sort()

        preds = None
        scores = []
        query_num_offset = 0
        for chunk_num, eval_dataset_path in eval_dataset_paths:
            with open(eval_dataset_path) as f:
                data_json_string = f.read().strip()
                feature_dict = json.loads(data_json_string)
                feature_dict = {k: [torch.tensor(item) for item in v] for k, v in feature_dict.items()}
                eval_dataset = DictDataset(feature_dict)

            eval_batch_size = batch_size
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size, collate_fn=task_wrapper.collate_fn)

            eval_dataloader = tqdm(eval_dataloader, desc="Chunk" + str(chunk_num))
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad(), torch.inference_mode(), torch.cuda.amp.autocast():
                    batch = {k: t.cuda() for k, t in batch.items()}
                    logits = task_wrapper.mlm_eval_step(batch)

                if preds is None:
                    preds = logits.detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

            final_preds, out_label_ids = [], []
            i = 0       # i stands for query sample index

            # label_convert maps training sample index to its corresponding label index
            label_convert = {}
            for ii, sample in enumerate(episode['support_set']):
                label_convert[ii] = int(sample['label'])

            # Pool relevance scores with mean pooling
            if self.config.pooling == 'mean':
                while i <= preds.shape[0] - len(episode['support_set']):
                    this_pred = [[] for _ in range(n_way)]
                    for j in range(len(episode['support_set'])):
                        try:
                            this_pred[label_convert[j]].append(preds[i + j][0] - preds[i + j][1])
                        except Exception as e:
                            print(e)

                    this_pred = [sum(scores)/len(scores) for scores in this_pred]
                    final_preds.append(this_pred)
                    out_label_ids.append(int(episode['query_set'][query_num_offset+i//len(episode['support_set'])]['label']))
                    i += len(episode['support_set'])

                final_preds = np.array(final_preds)
                predictions = np.argmax(final_preds, axis=1)
                out_label_ids = np.array(out_label_ids)
                scores.append([simple_accuracy(predictions, out_label_ids), i//len(episode['support_set'])])
                preds = preds[i:]
                query_num_offset += i//len(episode['support_set'])

            # Pool relevance scores with max pooling
            elif self.config.pooling == 'max':
                while i <= preds.shape[0] - len(episode['support_set']):
                    this_pred = [-1e5] * n_way
                    for j in range(len(episode['support_set'])):
                        # this_pred[j // k_shot] = max(this_pred[j // k_shot], preds[i+j][0] - preds[i+j][1])
                        this_pred[label_convert[j]] = max(this_pred[label_convert[j]], preds[i+j][0] - preds[i+j][1])
                    final_preds.append(this_pred)
                    out_label_ids.append(int(episode['query_set'][query_num_offset+i//len(episode['support_set'])]['label']))
                    i += len(episode['support_set'])

                final_preds = np.array(final_preds)
                predictions = np.argmax(final_preds, axis=1)
                out_label_ids = np.array(out_label_ids)
                scores.append([simple_accuracy(predictions, out_label_ids), i//len(episode['support_set'])])
                preds = preds[i:]
                query_num_offset += i//len(episode['support_set'])

            # Pool relevance scores with KNN pooling
            else:
                while i <= preds.shape[0] - len(episode['support_set']):
                    similarities = []
                    for j in range(len(episode['support_set'])):
                        similarities.append((preds[i+j][0] - preds[i+j][1], j))
                    similarities.sort(reverse=True)

                    knn_pool = [0] * n_way
                    for j in range(len(similarities)//2):
                        knn_pool[label_convert[similarities[j][1]]] += 1

                    # If several labels take up the same proportion of topk relevant samples, select the training
                    # sample achieving the highest relevance score and classify this query sample to its class
                    sorted_knn_pool = sorted(knn_pool, reverse=True)
                    if sorted_knn_pool[0] == sorted_knn_pool[1]:
                        for item in similarities:
                            if knn_pool[label_convert[item[1]]] == sorted_knn_pool[0]:
                                knn_pool[label_convert[item[1]]] += 1
                                break

                    final_preds.append(knn_pool)
                    out_label_ids.append(int(episode['query_set'][query_num_offset+i//len(episode['support_set'])]['label']))
                    i += len(episode['support_set'])

                final_preds = np.array(final_preds)
                predictions = np.argmax(final_preds, axis=1)
                out_label_ids = np.array(out_label_ids)
                scores.append([simple_accuracy(predictions, out_label_ids), i//len(episode['support_set'])])
                preds = preds[i:]
                query_num_offset += i//len(episode['support_set'])

        scores = {'acc': sum([score*num for score, num in scores]) / sum([num for score, num in scores])}
        print(scores)
        return scores


    def run(self,
             data: List[InputExample],
             start_episode=0,
             num_episodes=0,
             batch_size: int = 8,
             n_adapt_epochs: int = 3,
             gradient_accumulation_steps: int = 1,
             weight_decay: float = 0.1,  #
             lm_learning_rate: float = 1e-5,
             adam_epsilon: float = 1e-8,
             max_grad_norm: float = 1,
             ):

        scores = []
        episode_iter = tqdm(data[start_episode: start_episode+num_episodes], desc="Episode")
        for num, episode in enumerate(episode_iter):
            episode_score = self.run_single_episode(episode=episode,
                                                     num=num,
                                                     k_shot=self.config.k_shot,
                                                     batch_size=batch_size,
                                                     n_adapt_epochs=n_adapt_epochs,
                                                     lm_learning_rate=lm_learning_rate,
                                                     gradient_accumulation_steps=gradient_accumulation_steps,
                                                     weight_decay=weight_decay,  #
                                                     adam_epsilon=adam_epsilon,
                                                     max_grad_norm=max_grad_norm,
                                                     )
            scores.append(episode_score)
            print(f"{start_episode} set score: {episode_score}")
            torch.cuda.empty_cache()

        def mean(nums):
            return sum(nums) / len(nums)

        average_scores = {}

        for key in scores[0].keys():
            average_scores[key] = mean([episode_score[key] for episode_score in scores])
        return average_scores, scores


    def save_features(self, feature_dict, path):
        serializable_feature_dict = {k: [item.numpy().tolist() for item in v] for k, v in feature_dict.items()}
        feature_dict_json_string = json.dumps(serializable_feature_dict)
        with open(path, 'w') as f:
            f.write(feature_dict_json_string)


    def build_dataset(self, episode, set_num):
        base_dir = os.path.join(self.config.data_path, 'TextClassification', self.config.dataset,
                                       str(self.config.k_shot) + 'shot' + str(self.config.prompt_template) + 'template')
        os.makedirs(base_dir, exist_ok=True)

        # Construct training data
        encoded_train_data_path = os.path.join(base_dir, 'encoded_train_data_'+str(set_num)+'set.json')
        if os.path.exists(encoded_train_data_path):
            with open(encoded_train_data_path) as f:
                data_json_string = f.read().strip()
                feature_dict = json.loads(data_json_string)
                feature_dict = {k: [torch.tensor(item) for item in v] for k, v in feature_dict.items()}
                train_dataset = DictDataset(feature_dict)
        else:
            data = []
            for target_item in episode['support_set']:
                for source_item in episode['support_set']:
                    metric_keys = list(self.metric_verbalizer.keys())
                    label = metric_keys[0] if source_item['label'] == target_item['label'] else metric_keys[1]
                    item = {
                        'text1': source_item['raw'],
                        'text2': target_item['raw'],
                        'label': label
                    }
                    data.append(item)

            train_dataset, train_feature_dict = self._generate_dataset(data=data)
            self.save_features(train_feature_dict, encoded_train_data_path)

        # Construct training data for pivot sample selection
        encoded_pivot_data_path = os.path.join(base_dir, 'encoded_' + str(self.config.pivot) + 'pivot_data_' + str(set_num) + 'set.json')
        if os.path.exists(encoded_pivot_data_path):
            with open(encoded_pivot_data_path) as f:
                data_json_string = f.read().strip()
                feature_dict = json.loads(data_json_string)
                feature_dict = {k: [torch.tensor(item) for item in v] for k, v in feature_dict.items()}
                pivot_dataset = DictDataset(feature_dict)
        else:
            data = []
            for target_item in episode['support_set']:
                metric_keys = list(self.metric_verbalizer.keys())
                for source_item in episode['support_set']:
                    label = metric_keys[0] if source_item['label'] == target_item['label'] else metric_keys[1]
                    item = {
                        'text1': source_item['raw'],
                        'text2': target_item['raw'],
                        'label': label
                    }
                    data.append(item)

                # Move intra-class text pairs to the start position, and left inter-class text pairs behind
                cur = len(data) - len(episode['support_set'])
                for i in range(len(data) - len(episode['support_set']), len(data)):
                    if data[i]['label'] == metric_keys[0]:
                        data[cur], data[i] = data[i], data[cur]
                        cur += 1

            pivot_dataset, pivot_feature_dict = self._generate_dataset(data=data)
            self.save_features(pivot_feature_dict, encoded_pivot_data_path)

        # Construct test data
        encoded_test_data_path = os.path.join(base_dir, 'encoded_test_data_' + str(set_num) + 'set')
        data = []
        for target_item in episode['query_set']:
            for source_item in episode['support_set']:
                metric_keys = list(self.metric_verbalizer.keys())
                label = metric_keys[0] if source_item['label'] == target_item['label'] else metric_keys[1]
                item = {
                    'text1': source_item['raw'],
                    'text2': target_item['raw'],
                    'label': label
                }
                data.append(item)

        test_dataset_paths = []
        if len(data) > 100000:
            # Save 100,000 samples' encoded features as a chunk
            for i in range(0, len(data), 100000):
                this_encoded_test_data_path = encoded_test_data_path + '_' + str(i // 100000) + 'chunk.json'
                if not os.path.exists(this_encoded_test_data_path):
                    this_data = data[i: i + 100000]
                    # If the number of samples to be encoded exceeds 50,000, activate multiprocessing encoding
                    if len(this_data) < 50000:
                        test_dataset, test_feature_dict = self._generate_dataset(data=this_data)
                    else:
                        test_dataset, test_feature_dict = self._generate_dataset(data=this_data, multiprocessing=True)
                    self.save_features(test_feature_dict, this_encoded_test_data_path)
                test_dataset_paths.append(this_encoded_test_data_path)
        else:
            this_encoded_test_data_path = encoded_test_data_path + '_0chunk.json'
            if not os.path.exists(this_encoded_test_data_path):
                test_dataset, test_feature_dict = self._generate_dataset(data=data)
                self.save_features(test_feature_dict, this_encoded_test_data_path)
            test_dataset_paths.append(this_encoded_test_data_path)

        return train_dataset, test_dataset_paths, pivot_dataset


    def build_pivot_dataset(self, pivot_data, query_data, set_num):
        """
            Construct test data with pivot samples
        """
        base_dir = os.path.join(self.config.data_path, 'TextClassification', self.config.dataset,
                                str(self.config.k_shot) + 'shot' + str(self.config.prompt_template) + 'template')
        encoded_test_data_path = os.path.join(base_dir, 'encoded_' + str(self.config.pivot) + 'pivot_test_data_' + str(set_num) + 'set')
        data = []
        for target_item in query_data:
            for source_item in pivot_data:
                metric_keys = list(self.metric_verbalizer.keys())
                label = metric_keys[0] if source_item['label'] == target_item['label'] else metric_keys[1]
                item = {
                    'text1': source_item['raw'],
                    'text2': target_item['raw'],
                    'label': label
                }
                data.append(item)

        test_dataset_paths = []
        if len(data) > 100000:
            for i in range(0, len(data), 100000):
                this_encoded_test_data_path = encoded_test_data_path + '_' + str(i // 100000) + 'chunk.json'
                if not os.path.exists(this_encoded_test_data_path):
                    this_data = data[i: i + 100000]
                    if len(this_data) < 50000:
                        test_dataset, test_feature_dict = self._generate_dataset(data=this_data)
                    else:
                        test_dataset, test_feature_dict = self._generate_dataset(data=this_data, multiprocessing=True)
                    self.save_features(test_feature_dict, this_encoded_test_data_path)
                test_dataset_paths.append(this_encoded_test_data_path)
        else:
            this_encoded_test_data_path = encoded_test_data_path + '_0chunk.json'
            if not os.path.exists(this_encoded_test_data_path):
                test_dataset, test_feature_dict = self._generate_dataset(data=data)
                self.save_features(test_feature_dict, this_encoded_test_data_path)
            test_dataset_paths.append(this_encoded_test_data_path)

        return test_dataset_paths


    def collate_fn(self, batch):
        """
            Pad input features to a fixed length to be processed in batches
        """
        return_batch = {}
        tokenizer = self.tokenizer

        for key in batch[0].keys():
            if len(batch[0][key].shape) == 0:
                return_batch[key] = torch.stack([batch[i][key] for i in range(len(batch))])
                continue

            # max_seq_length = max([batch[i][key].shape[0] for i in range(len(batch))])
            max_seq_length = 256        # Model accelerated with kernl requires each batch to have the same feature shape
            this_tensors = torch.full((len(batch), max_seq_length), tokenizer.pad_token_id, dtype=torch.long)
            for i in range(len(batch)):
                this_tensors[i, :batch[i][key].shape[0]] = batch[i][key]
            return_batch[key] = this_tensors

        return return_batch


    def _generate_dataset(self, data: list, labelled: bool = True, multiprocessing: bool = False):
        features = self._convert_examples_to_features(data, labelled, multiprocessing)
        feature_dict = {
            'input_ids': [torch.tensor(f["input_ids"], dtype=torch.long) for this_features in features for f in this_features],
            'attention_mask': [torch.tensor(f["attention_mask"], dtype=torch.long) for this_features in features for f in this_features],
            'token_type_ids': [torch.tensor(f["token_type_ids"], dtype=torch.long) for this_features in features for f in this_features],
            'labels': [torch.tensor(f["label"], dtype=torch.long) for this_features in features for f in this_features],
            'mlm_labels': [torch.tensor(f["mlm_labels"], dtype=torch.long) for this_features in features for f in this_features],
        }
        return DictDataset(feature_dict), feature_dict


    def _convert_examples_to_features(self, examples: list, labelled: bool = True, multiprocessing: bool = False):
        tprint("start generating dataset")

        if not multiprocessing:
            features = []
            for (ex_index, example) in enumerate(examples):
                if ex_index > 0 and ex_index % 10000 == 0:
                    tprint("Writing example {}".format(ex_index))
                input_features = self.get_input_features(example, labelled=labelled)

                features.append(input_features)
            features = [features]
        else:
            features = []
            def func(examples):
                features = []
                for (ex_index, example) in enumerate(examples):
                    if ex_index > 0 and ex_index % 100000 == 0:
                        tprint("Writing example {}".format(ex_index))
                    input_features = self.get_input_features(example, labelled=labelled)

                    features.append(input_features)
                return features

            # pool = Pool(os.cpu_count())
            worker_num = 20
            pool = Pool(worker_num)
            group_size = math.ceil(len(examples)/worker_num)
            example_groups = [examples[i*group_size: (i+1)*group_size] for i in range(worker_num)]
            worker = pool.imap(func, example_groups)
            for this_features in worker:
                features.append(this_features)
                pass

        tprint("finished generating dataset")
        return features


    def generate_item(self, example: dict, task: dict):
        if hasattr(self.config, 'prompt_template'):
            prompt_template = self.config.prompt_template
        else:
            prompt_template = 0
        prompted_str = task['metric_prompt'][prompt_template](example['text1'], example['text2'])

        parts = [self.tokenizer.encode(string, add_special_tokens=False) for string in prompted_str]
        parts = [part[:120] for part in parts]
        token_ids = [id for ids in parts for id in ids]

        input_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids)
        sep_token_position = token_ids.index(102)
        token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(
            token_ids_0=token_ids[: sep_token_position], token_ids_1=token_ids[sep_token_position+1:])

        assert len(input_ids) == len(token_type_ids)
        return input_ids, token_type_ids


    def get_input_features(self, example: dict, labelled: bool):
        input_ids, token_type_ids = self.generate_item(example,
                                                       TASK_CLASSES[self.config.dataset])

        attention_mask = [1] * len(input_ids)
        example_label = example['label']
        label = self.metric_label_map[example_label] if example_label is not None else -100

        if labelled:
            labels = [-1] * len(input_ids)
            for label_idx, input_id in enumerate(input_ids):
                if input_id == self.tokenizer.mask_token_id:
                    labels[label_idx] = 1
            mlm_labels = labels
        else:
            mlm_labels = [-1] * 512

        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "label": label,
                "mlm_labels": mlm_labels}



    def get_mask_positions(self, input_ids: List[int]):
        labels = [-1] * len(input_ids)
        for label_idx, input_id in enumerate(input_ids):
            if input_id == self.tokenizer.mask_token_id:
                labels[label_idx] = 1
        return labels


    def generate_default_inputs(self, batch: Dict[str, torch.Tensor], M=None):

        inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
        if self.config.pretrained_model in ['bert']:
            inputs['token_type_ids'] = batch['token_type_ids']

        return inputs


    def mlm_train_step(self, labeled_batch: Dict[str, torch.Tensor]):
        inputs = self.generate_default_inputs(labeled_batch)
        mlm_labels, labels = labeled_batch['mlm_labels'], labeled_batch['labels']
        outputs = self.model(**inputs, output_hidden_states=True)
        prediction_scores = self.convert_mlm_logits_to_cls_logits(mlm_labels, outputs[0])

        loss = nn.CrossEntropyLoss()(prediction_scores.view(-1, len(self.metric_verbalizer.keys())), labels.view(-1))
        return loss


    def mlm_eval_step(self, batch: Dict[str, torch.Tensor]):
        inputs = self.generate_default_inputs(batch)
        outputs = self.model(**inputs)
        return self.convert_mlm_logits_to_cls_logits(batch['mlm_labels'], outputs[0])


    def convert_mlm_logits_to_cls_logits(self, mlm_labels: torch.Tensor, logits: torch.Tensor):
        masked_logits = logits[mlm_labels > 0]
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(ml) for ml in masked_logits])
        return cls_logits


    def _convert_single_mlm_logits_to_cls_logits(self, logits: torch.Tensor):
        m2c = self.metric_mlm_logits_to_answer_logits_tensor.to(logits.device)
        filler_len = torch.tensor([len(self.metric_verbalizer[label]) for label in self.metric_verbalizer.keys()],
                                  dtype=torch.float)
        filler_len = filler_len.to(logits.device)

        cls_logits = logits[torch.max(torch.zeros_like(m2c), m2c)]
        cls_logits = cls_logits * (m2c > 0).float()

        cls_logits = cls_logits.sum(axis=1) / filler_len
        return cls_logits


    def _build_mlm_logits_to_cls_logits_tensor(self):
        metric_label_list = self.metric_verbalizer.keys()
        max_num_answers = max([len(self.metric_verbalizer[label]) for label in metric_label_list])
        metric_m2c_tensor = torch.ones([len(metric_label_list), max_num_answers],
                                dtype=torch.long,
                                requires_grad=False) * -1

        for label_idx, label in enumerate(metric_label_list):
            answers = self.metric_verbalizer[label]
            for answer_id, answer in enumerate(answers):
                verbalizer_id = self.tokenizer.encode(answer, add_special_tokens=False)[0]
                # verbalizer_id = get_verbalization_ids(answer, self.wrapper.tokenizer, force_single_token=True)
                assert verbalizer_id != self.tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
                metric_m2c_tensor[label_idx, answer_id] = verbalizer_id

        return metric_m2c_tensor


    def prepare_optimizer_scheduler(
            self,
            cur_model: torch.nn.Module,
            t_total: int,
            weight_decay: float = 0.0,
            lm_learning_rate: float = 1e-5,
            adam_epsilon: float = 1e-8,
            warmup_steps: int = 0,
    ):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in cur_model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in cur_model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=lm_learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=t_total)

        return optimizer, scheduler






