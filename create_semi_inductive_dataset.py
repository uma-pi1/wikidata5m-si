import os
import yaml
import torch
import argparse
import numpy as np
import pandas as pd


"""
This script creates a dataset to evaluate KGC models in the semi-inductive setting.
"""


class Dataset:

    def __init__(self, folder_name: str):
        self.folder_name = folder_name
        self.split = dict()
        self.split["train"] = torch.from_numpy(pd.read_csv(
            os.path.join("data", folder_name, "train.del"), header=None, sep="\t"
        ).to_numpy())
        self.split["valid"] = torch.from_numpy(pd.read_csv(
            os.path.join("data", folder_name, "valid.del"), header=None, sep="\t"
        ).to_numpy())
        self.split["test"] = torch.from_numpy(pd.read_csv(
            os.path.join("data", folder_name, "test.del"), header=None, sep="\t"
        ).to_numpy())

        self.entity_ids = self._load_ids("entity_ids.del")
        self.relation_ids = self._load_ids("relation_ids.del")

    def _load_ids(self, file_name):
        return_list = []
        with open(os.path.join("data", self.folder_name, file_name), 'r') as fp:
            for line in fp.readlines():
                idx, value = line.strip().split("\t")
                return_list.append(value)
        return return_list

    def num_entities(self):
        return len(self.entity_ids)

    def num_relations(self):
        return len(self.relation_ids)


def sample_unseen_entities_stratified(dataset: Dataset, lower_limit: int, upper_limit: int, num_ents_per_split: int, remove_test_val_ents=True):
    ent_counts = torch.from_numpy(
        np.bincount(
            dataset.split["train"][:, [0, 2]].view(-1), minlength=dataset.num_entities()
        )
    )

    # remove entities occurring in test or valid set
    if remove_test_val_ents:
        val_ents = np.unique(dataset.split["valid"][:, [0, 2]].view(-1))
        test_ents = np.unique(dataset.split["test"][:, [0, 2]].view(-1))

    # get counts per possible occurrence group
    group_counts = list()
    group_masks = list()
    stratified_counts = list()
    for i in range(lower_limit, upper_limit+1):
        group_mask = (ent_counts == i)
        group_count = group_mask.sum().item()
        if remove_test_val_ents:
            group_mask[val_ents] = False
            group_mask[test_ents] = False
        group_counts.append(group_count)
        group_masks.append(group_mask)

    total_group_count = sum(group_counts)
    selected_unseen_entities = list()
    all_entities = np.arange(dataset.num_entities())
    for group_count, group_mask in zip(group_counts, group_masks):
        stratified_count = int(round((group_count/total_group_count)*num_ents_per_split*2))
        stratified_counts.append(stratified_count)
        group_entities = all_entities[group_mask]
        shuffler = torch.randperm(len(group_entities))
        selected_group_entities = group_entities[shuffler][:stratified_count]
        selected_unseen_entities.append(selected_group_entities)

    selected_unseen_entities = np.concatenate(selected_unseen_entities)
    # shuffle again so that valid and test have same distribution of groups
    selected_unseen_entities = selected_unseen_entities[torch.randperm(len(selected_unseen_entities))]

    # let's get the set of seen entities
    seen_mask = torch.ones(dataset.num_entities(), dtype=torch.bool)
    seen_mask[selected_unseen_entities] = False
    entities_seen = torch.arange(dataset.num_entities())[seen_mask]

    entities_unseen_valid, entities_unseen_test = torch.from_numpy(selected_unseen_entities).chunk(2)
    return entities_seen, entities_unseen_valid, entities_unseen_test


def select_triple_by_relation_frequency(split_data, split_entities, train_data, num_relations, num_triples_to_select):
    relation_frequency = np.bincount(train_data[:, 1], minlength=num_relations)
    sorted_split_pool = []
    for eu in split_entities:
        relevant_triples = split_data[np.logical_or(split_data[:, 0] == eu, split_data[:, 2] == eu)]
        relevant_relation_frequency = relation_frequency[relevant_triples[:, 1]]
        relation_frequ_sorter = np.argsort(-relevant_relation_frequency)
        relevant_triples = relevant_triples[relation_frequ_sorter]
        relevant_triples = relevant_triples[:num_triples_to_select]

        # add id of unseen entity in first column
        prepended_triples = np.full((len(relevant_triples), 5), eu)
        prepended_triples[:, 2:] = relevant_triples
        direction_mask = relevant_triples[:, 0] == eu

        # indicator for unseen entity slot in second colum
        prepended_triples[direction_mask, 1] = 0
        prepended_triples[~direction_mask, 1] = 2
        sorted_split_pool.append(prepended_triples)
    sorted_split_pool = np.concatenate(sorted_split_pool, axis=0)
    return sorted_split_pool


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument('-mm', '--map_mentions', action='store_true')
    parser.set_defaults(map_mentions=True)
    args = parser.parse_args()

    map_mentions = args.map_mentions
    dataset_name = args.dataset
    lower_limit = 11
    upper_limit = 20
    num_ents_per_split = 500
    num_triples_per_entity = lower_limit
    dataset = Dataset(folder_name=dataset_name)

    set_seeds(444)

    entities_seen, entities_unseen_valid, entities_unseen_test = sample_unseen_entities_stratified(dataset, lower_limit, upper_limit, num_ents_per_split)

    # let's assign the triples to their corresponding sets
    s_in_valid_unseen_mask = torch.from_numpy(np.isin(dataset.split["train"][:, 0], entities_unseen_valid))
    o_in_valid_unseen_mask = torch.from_numpy(np.isin(dataset.split["train"][:, 2], entities_unseen_valid))
    s_in_test_unseen_mask = torch.from_numpy(np.isin(dataset.split["train"][:, 0], entities_unseen_test))
    o_in_test_unseen_mask = torch.from_numpy(np.isin(dataset.split["train"][:, 2], entities_unseen_test))

    # first find the triples to remove from train set
    # remove when either subject and object are unseen
    train_split_new = dataset.split["train"][~(s_in_valid_unseen_mask | o_in_valid_unseen_mask | s_in_test_unseen_mask | o_in_test_unseen_mask)]

    possible_valid_data = dataset.split["train"][torch.logical_and(
        torch.logical_xor(s_in_valid_unseen_mask, o_in_valid_unseen_mask),
        ~torch.logical_or(s_in_test_unseen_mask, o_in_test_unseen_mask)
    )]
    possible_test_data = dataset.split["train"][torch.logical_and(
        torch.logical_xor(s_in_test_unseen_mask, o_in_test_unseen_mask),
        ~torch.logical_or(s_in_valid_unseen_mask, o_in_valid_unseen_mask)
    )]

    # filter original valid and test
    s_in_valid_unseen_mask_valid = np.isin(dataset.split["valid"][:, 0], entities_unseen_valid)
    o_in_valid_unseen_mask_valid = np.isin(dataset.split["valid"][:, 2], entities_unseen_valid)
    s_in_test_unseen_mask_valid = np.isin(dataset.split["valid"][:, 0], entities_unseen_test)
    o_in_test_unseen_mask_valid = np.isin(dataset.split["valid"][:, 2], entities_unseen_test)
    valid_split_seen = dataset.split["valid"][~(
            s_in_valid_unseen_mask_valid |
            o_in_valid_unseen_mask_valid |
            s_in_test_unseen_mask_valid |
            o_in_test_unseen_mask_valid
    )]
    s_in_valid_unseen_mask_test = np.isin(dataset.split["test"][:, 0], entities_unseen_valid)
    o_in_valid_unseen_mask_test = np.isin(dataset.split["test"][:, 2], entities_unseen_valid)
    s_in_test_unseen_mask_test = np.isin(dataset.split["test"][:, 0], entities_unseen_test)
    o_in_test_unseen_mask_test = np.isin(dataset.split["test"][:, 2], entities_unseen_test)
    test_split_seen = dataset.split["test"][~(
            s_in_valid_unseen_mask_test |
            o_in_valid_unseen_mask_test |
            s_in_test_unseen_mask_test |
            o_in_test_unseen_mask_test
    )]

    # finally, after removing the entities we need to remap the ids,
    # so that 1-n in train n-m in valid and m-o in test
    all_entities = np.concatenate([entities_seen, entities_unseen_valid, entities_unseen_test])
    id_mapper = np.full(dataset.num_entities(), 100000000, dtype=np.int64)
    id_mapper[all_entities] = np.arange(len(all_entities))
    train_split_new = train_split_new.numpy()
    train_split_new[:, 0] = id_mapper[train_split_new[:, 0]]
    train_split_new[:, 2] = id_mapper[train_split_new[:, 2]]

    # map original valid data
    valid_split_seen = valid_split_seen.numpy()
    valid_split_seen[:, 0] = id_mapper[valid_split_seen[:, 0]]
    valid_split_seen[:, 2] = id_mapper[valid_split_seen[:, 2]]

    # map valid data few shot pool
    possible_valid_data = possible_valid_data.numpy()
    possible_valid_data[:, 0] = id_mapper[possible_valid_data[:, 0]]
    possible_valid_data[:, 2] = id_mapper[possible_valid_data[:, 2]]

    # map original test data
    test_split_seen = test_split_seen.numpy()
    test_split_seen[:, 0] = id_mapper[test_split_seen[:, 0]]
    test_split_seen[:, 2] = id_mapper[test_split_seen[:, 2]]

    # map test data few shot pool
    possible_test_data = possible_test_data.numpy()
    possible_test_data[:, 0] = id_mapper[possible_test_data[:, 0]]
    possible_test_data[:, 2] = id_mapper[possible_test_data[:, 2]]

    entities_seen = id_mapper[entities_seen]
    entities_unseen_valid = id_mapper[entities_unseen_valid]
    entities_unseen_test = id_mapper[entities_unseen_test]

    sorted_valid_pool = select_triple_by_relation_frequency(
        split_data=possible_valid_data,
        split_entities=entities_unseen_valid,
        train_data=train_split_new,
        num_relations=dataset.num_relations(),
        num_triples_to_select=num_triples_per_entity
    )
    sorted_test_pool = select_triple_by_relation_frequency(
        split_data=possible_test_data,
        split_entities=entities_unseen_test,
        train_data=train_split_new,
        num_relations=dataset.num_relations(),
        num_triples_to_select=num_triples_per_entity
    )

    # print some statistics
    print("entities seen", len(entities_seen))
    print("entities unseen valid", len(entities_unseen_valid))
    print("entities unseen test", len(entities_unseen_test))
    print("relations", dataset.num_relations())
    print("relations in valid", len(np.unique(sorted_valid_pool[:, 3])))
    print("relations in test", len(np.unique(sorted_test_pool[:, 3])))
    print("train", len(train_split_new))
    print("valid pool", len(sorted_valid_pool))
    print("test pool", len(sorted_test_pool))

    # in the next step we map back all ids to their text-ids and write out train.txt, valid.txt, test.txt
    # make new directory
    new_dataset_name = f"{dataset_name}_semi_inductive_test"
    output_folder = os.path.join("data", new_dataset_name)
    os.mkdir(output_folder)

    reverse_mapper = np.argsort(id_mapper)[:len(all_entities)]
    with open(os.path.join(output_folder, "all_entity_ids.del"), "w") as entity_ids_file:
        for new_id, old_id in enumerate(reverse_mapper):
            entity_ids_file.write(f"{new_id}\t{dataset.entity_ids[old_id]}\n")

    with open(os.path.join(output_folder, "entity_ids.del"), "w") as entity_ids_file:
        for new_id, old_id in enumerate(reverse_mapper[:len(entities_seen)]):
            entity_ids_file.write(f"{new_id}\t{dataset.entity_ids[old_id]}\n")

    with open(os.path.join(output_folder, "valid_entity_ids.del"), "w") as entity_ids_file:
        for new_id, old_id in enumerate(reverse_mapper[len(entities_seen):len(entities_seen)+len(entities_unseen_valid)], len(entities_seen)):
            entity_ids_file.write(f"{new_id}\t{dataset.entity_ids[old_id]}\n")

    with open(os.path.join(output_folder, "test_entity_ids.del"), "w") as entity_ids_file:
        for new_id, old_id in enumerate(reverse_mapper[len(entities_seen)+len(entities_unseen_valid):], len(entities_seen)+len(entities_unseen_valid)):
            entity_ids_file.write(f"{new_id}\t{dataset.entity_ids[old_id]}\n")

    # now map entity mentions
    if map_mentions:
        entity_mentions = []
        with open(os.path.join("data", dataset_name, "entity_mentions.del")) as f:
            for line in f:
                entity_mentions.append(line.strip().split("\t", 1)[1])

        with open(os.path.join(output_folder, "all_entity_mentions.del"), "w") as entity_ids_file:
            for new_id, old_id in enumerate(reverse_mapper):
                entity_ids_file.write(f"{new_id}\t{entity_mentions[old_id]}\n")

        with open(os.path.join(output_folder, "entity_mentions.del"), "w") as entity_ids_file:
            for new_id, old_id in enumerate(reverse_mapper[:len(entities_seen)]):
                entity_ids_file.write(f"{new_id}\t{entity_mentions[old_id]}\n")

        with open(os.path.join(output_folder, "valid_entity_mentions.del"), "w") as entity_ids_file:
            for new_id, old_id in enumerate(reverse_mapper[len(entities_seen):len(entities_seen)+len(entities_unseen_valid)], len(entities_seen)):
                entity_ids_file.write(f"{new_id}\t{entity_mentions[old_id]}\n")

        with open(os.path.join(output_folder, "test_entity_mentions.del"), "w") as entity_ids_file:
            for new_id, old_id in enumerate(reverse_mapper[len(entities_seen)+len(entities_unseen_valid):], len(entities_seen)+len(entities_unseen_valid)):
                entity_ids_file.write(f"{new_id}\t{entity_mentions[old_id]}\n")

    # now map relation ids
    with open(os.path.join(output_folder, "relation_ids.del"), "w") as relation_ids_file:
        for new_id, relation in enumerate(dataset.relation_ids):
            relation_ids_file.write(f"{new_id}\t{relation}\n")

    # now map relation mentions
    if map_mentions:
        relation_mentions = []
        with open(os.path.join("data", dataset_name, "relation_mentions.del")) as f:
            for line in f:
                relation_mentions.append(line.strip().split("\t", 1)[1])
        with open(os.path.join(output_folder, "relation_mentions.del"), "w") as relation_ids_file:
            for new_id, relation in enumerate(relation_mentions):
                relation_ids_file.write(f"{new_id}\t{relation}\n")

    np.savetxt(
        os.path.join(output_folder, "train.del"),
        train_split_new,
        delimiter="\t",
        fmt="%d",
    )
    np.savetxt(
        os.path.join(output_folder, "valid.del"),
        valid_split_seen,
        delimiter="\t",
        fmt="%d",
    )
    np.savetxt(
        os.path.join(output_folder, "test.del"),
        test_split_seen,
        delimiter="\t",
        fmt="%d",
    )
    np.savetxt(
        os.path.join(output_folder, "valid_pool.del"),
        sorted_valid_pool,
        delimiter="\t",
        fmt="%d",
    )
    np.savetxt(
        os.path.join(output_folder, "test_pool.del"),
        sorted_test_pool,
        delimiter="\t",
        fmt="%d",
    )

    yaml_config = {
        "dataset": {
            "files.entity_ids.filename": "all_entity_ids.del",
            "files.entity_ids.type": "map",
            "files.seen_entity_ids.filename": "seen_entity_ids.del",
            "files.seen_entity_ids.type": "map",
            "files.test.filename": "test.del",
            "files.test.type": "triples",
            "files.valid.filename": "valid.del",
            "files.valid.type": "triples",
            "files.train.filename": "train.del",
            "files.train.type": "triples",
            "name": new_dataset_name,
        }
    }
    with open(os.path.join(output_folder, "dataset.yaml"), "w") as yaml_file:
        dump = yaml.dump(yaml_config)
        yaml_file.write(dump)


