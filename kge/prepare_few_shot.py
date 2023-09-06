import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


class FewShotSetCreator:
    def __init__(
            self,
            dataset_name="wikidata-semi-inductive",
            split="valid",
            use_inverse=True,
            context_selection="most_common",
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.use_inverse = use_inverse
        self.context_selection = context_selection
        if self.use_inverse:
            train_triples = self._load_train_triples()
            self.num_relations = len(np.unique(train_triples[:, 1]))
        self.triple_pool = self._load_triple_pool()

    def _load_train_triples(self):
        triples = pd.read_csv(
            os.path.join("data", self.dataset_name, "train.del"),
            delimiter="\t",
            header=None
        ).to_numpy()
        return triples

    def _load_triple_pool(self):
        triple_pool = pd.read_csv(
            os.path.join("data", self.dataset_name, f"{self.split}_pool.del"),
            delimiter="\t",
            header=None
        ).to_numpy()
        if self.use_inverse:
            triple_inverse_pool = np.copy(triple_pool)
            triple_inverse_pool[:, 3] += self.num_relations
            triple_inverse_pool[:, 2] = np.copy(triple_pool[:, 4])
            triple_inverse_pool[:, 4] = np.copy(triple_pool[:, 2])

            relevant_triple_pool = triple_pool[triple_pool[:, 1] == 0]
            relevant_inverse_triple_pool = triple_inverse_pool[triple_inverse_pool[:, 1] == 2]
            relevant_inverse_triple_pool[:, 1] = 0
            triple_pool = np.concatenate((relevant_triple_pool, relevant_inverse_triple_pool))

        return triple_pool

    def create_few_shot_dataset(self, num_shots):
        print(f"create few shot set for {self.split} with {num_shots} shots")
        # convert to torch as split behaves differently in torch compared to numpy
        triples_per_entity = [t.numpy() for t in torch.from_numpy(self.triple_pool).split(11)]
        eval_list = list()
        for tpe in tqdm(triples_per_entity):
            for i in range(len(tpe)):
                mask = np.ones((len(tpe,)), dtype=bool)
                mask[i] = False
                triple = tpe[i]
                context = tpe[mask]
                if self.context_selection == "most_common":
                    context = context[:num_shots]
                elif self.context_selection == "least_common":
                    context = context[-num_shots:]
                elif self.context_selection == "random":
                    # numpy shuffles in place
                    np.random.shuffle(context)
                    context = context[:num_shots]
                eval_dict = {
                    "unseen_entity": triple[0].item(),
                    "unseen_slot": triple[1].item(),
                    "triple": triple[2:],
                    "context": context
                }
                eval_list.append(eval_dict)
        return eval_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="wikidata5m-semi-inductive")
    parser.add_argument("--split", "-s", type=str, default="valid")
    parser.add_argument("--num_shots", "-k", type=int, default=10)
    args = parser.parse_args()

    few_shot_set_creator = FewShotSetCreator(
        dataset_name=args.dataset,
        split=args.split,
    )

    few_shot_set_creator.create_few_shot_dataset(args.num_shots)




