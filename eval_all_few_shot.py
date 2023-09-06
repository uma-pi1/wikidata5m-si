import os
import torch
import argparse
from eval_few_shot import run


def eval_all_few_shot(config_path, model_path, split="test"):
    torch.set_float32_matmul_precision('medium')

    for cs in ["most_common", "least_common", "random"]:
        for k in [0, 1, 3, 5, 10]:
            print("runnning few shot with", k, cs)
            run(checkpoint_path=model_path, config_path=config_path, num_shots=k, split=split, context_selection=cs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test kgt5 model')
    parser.add_argument('-c', '--config', help='Path to config',
                        required=True)
    parser.add_argument('-m', '--model', help='Path to checkpoint',
                        required=True)
    parser.add_argument('-s', '--split', help='Split to evaluate on',
                        default="test")

    eval_all_few_shot(args["config"], args["model"], split=args["split"]) 
