import os
import argparse
import random

import torch
import numpy as np

import dataloader as loader
from model import TransformerModelWrapper
import wandb
from utils import tprint

def parse_args():
    parser = argparse.ArgumentParser(
        description="Few Shot Text Classification with P-tuning")

    # data configuration
    parser.add_argument("--data_path", type=str,
                        default="data",
                        help="path to dataset")
    parser.add_argument("--dataset", type=str, default="agnews",
                        help="name of the dataset. "
                             "Options: [agnews, dbpedia, yahoo_answers_topics]")
    parser.add_argument("--output_dir", default="./output_dir/",
                        type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written")

    # backbone model configuration
    parser.add_argument("--pretrained_model", default="bert",
                        help="use PLM embedding (only available for sent-level datasets: huffpost, fewrel")
    parser.add_argument("--model_type", default="bert-base-uncased",
                        help="[bert-base-uncased, llm]")

    # task configuration
    parser.add_argument("--k_shot", type=int, default=2,
                        help="#support examples for each class for each task")
    parser.add_argument("--start_episode", type=int, default=0,
                        help="The index of the training set to start with")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="#training sets")

    # training options
    parser.add_argument("--seed", type=int, default=1999, help="seed")
    parser.add_argument("--prompt_template", type=int, default=0)
    parser.add_argument("--n_adapt_epochs", type=int, default=120, help="#Training epochs")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lm_learning_rate", type=float, default=1e-5)
    parser.add_argument("--kernl_accerleration", type=int, default=0,
                        help="0 for plain inference, 1 for accelerated inference")

    # MetricPrompt options
    parser.add_argument("--pooling", type=str, default='mean',
                        help=("Options: [mean, max, knn]"))
    parser.add_argument("--pivot", type=int, default=2,
                        help=("Options: [0, 1, 2]"))

    return parser.parse_args()


def set_seed(seed):
    """
        Setting random seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    torch.multiprocessing.set_start_method('spawn')

    args = parse_args()
    tprint(args)
    file_path = os.path.abspath(__file__)
    dir_path = file_path[:file_path.rfind('/')]

    config_str = str(args.k_shot) + 'shot_' + str(args.n_adapt_epochs) + 'ada' + \
                 str(args.seed) + 'seed' + str(args.prompt_template) + 'template'
    args.output_dir = os.path.join(dir_path,
                                   args.output_dir,
                                   args.dataset,
                                   config_str + '')
    args.data_path = os.path.join(dir_path, args.data_path)
    set_seed(args.seed)
    episodes = loader.load_true_few_shot_dataset(args)

    wrapper = TransformerModelWrapper(args)

    scores, average_scores = wrapper.run(data=episodes,
                                         start_episode=args.start_episode,
                                         num_episodes=args.num_episodes,
                                         batch_size=args.batch_size,
                                         n_adapt_epochs=args.n_adapt_epochs,
                                         lm_learning_rate=args.lm_learning_rate,
                                         )
    print(scores)
    print(average_scores)


if __name__ == "__main__":
    main()
