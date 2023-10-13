import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    input_folder = "data/wikidata5m-si"
    output_folder = "data/wikidata5m_si"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    # create entities.json
    # NOTE: we do not have to care about all entities occurring in entities.json
    # as unseen ones will never be used during training.
    # They also won't occur as negatives due to the batch, pre-batch and self-negative
    # sampling techniques of simkgc

    simkgc_entities = []
    wiki_ids = []
    mentions = []
    descriptions = []
    with open(os.path.join(input_folder, "all_entity_ids.del"), "r") as f:
        for line in f.readlines():
            wiki_ids.append(line.split("\t")[1])
    with open(os.path.join(input_folder, "all_entity_mentions.del"), "r") as f:
        for line in f.readlines():
            mentions.append(line.split("\t")[1])
    with open(os.path.join(input_folder, "all_entity_desc.del"), "r") as f:
        for line in f.readlines():
            descriptions.append(line.split("\t")[1])

    relations = []
    relation_ids = []
    with open(os.path.join(input_folder, "relation_mentions.del"), "r") as f:
        for line in f.readlines():
            relations.append(line.split("\t")[1])
    with open(os.path.join(input_folder, "relation_ids.del"), "r") as f:
        for line in f.readlines():
            relation_ids.append(line.split("\t")[1])
    relation_json = dict()
    for rel_id, rel_mention in zip(relation_ids, relations):
        relation_json[rel_id] = rel_mention
    with open(os.path.join(output_folder, "relations.json"), "w") as f:
        json.dump(relation_json, f, indent=4)


    for wiki_id, mention, description in zip(wiki_ids, mentions, descriptions):
        simkgc_entities.append(
            {
                "entity_id": wiki_id,
                "entity": mention,
                "entity_desc": description
            }
        )

    print("write simkgc entities to file")
    with open(os.path.join(output_folder, "entities.json"), "w") as f:
        json.dump(simkgc_entities, f, indent=4)

    for split in ["train", "valid", "test"]:
        print(f"read transductive {split} file")
        simkgc_triples_json = []
        simkgc_triples_txt = []
        triples = pd.read_csv(os.path.join(input_folder, f"{split}.del"), sep="\t", header=None).to_numpy().tolist()
        print(f"map transductive {split} file")
        for triple in tqdm(triples):
            simkgc_triples_json.append(
                {
                    "head_id": wiki_ids[triple[0]],
                    "head": mentions[triple[0]],
                    "relation": relations[triple[1]],
                    "tail_id": wiki_ids[triple[2]],
                    "tail": mentions[triple[2]],
                }
            )
            simkgc_triples_txt.append(
                f"{wiki_ids[triple[0]]}\t{relation_ids[triple[1]]}\t{wiki_ids[triple[2]]}\n"
            )
        print("write simkgc transductive", split)
        with open(os.path.join(output_folder, f"{split}.txt.json"), "w") as f:
            json.dump(simkgc_triples_json, f, indent=4)

        with open(os.path.join(output_folder, f"{split}.txt"), "w") as f:
            for line in tqdm(simkgc_triples_txt):
                f.write(line)

    print("map semi inductive split")
    print("create one head and one tail prediction file")
    print("resulting mrr needs to be combined by a weighted average after evaluation")

    data = pd.read_csv(os.path.join(input_folder, "test_pool.del"), delimiter="\t", header=None).to_numpy()

    entity_id_dict = dict()
    with open(os.path.join(input_folder, "all_entity_ids.del"), "r") as efile:
        for line in efile.readlines():
            split_line = line.strip().split("\t")
            entity_id_dict[int(split_line[0])] = split_line[1]
    rel_id_dict = dict()
    with open(os.path.join(input_folder, "relation_ids.del"), "r") as efile:
        for line in efile.readlines():
            split_line = line.strip().split("\t")
            rel_id_dict[int(split_line[0])] = split_line[1]

    data0 = data[data[:, 1] == 0]
    data2 = data[data[:, 1] == 2]

    tail_pred_json = []
    with open(os.path.join(output_folder, "test_pool_tail_pred.txt"), "w") as tpfile:
        for d in data0.tolist():
            tpfile.write(f"{entity_id_dict[d[2]]}\t{rel_id_dict[d[3]]}\t{entity_id_dict[d[4]]}\n")
            tail_pred_json.append(
                {
                    "head_id": wiki_ids[d[2]],
                    "head": mentions[d[2]],
                    "relation": relations[d[3]],
                    "tail_id": wiki_ids[d[4]],
                    "tail": mentions[d[4]],
                }
            )
    with open(os.path.join(output_folder, f"test_pool_tail_pred.txt.json"), "w") as f:
        json.dump(tail_pred_json, f, indent=4)

    head_pred_json = []
    with open(os.path.join(output_folder, "test_pool_head_pred.txt"), "w") as tpfile:
        for d in data2.tolist():
            tpfile.write(
                f"{entity_id_dict[d[2]]}\t{rel_id_dict[d[3]]}\t{entity_id_dict[d[4]]}\n")
            head_pred_json.append(
                {
                    "head_id": wiki_ids[d[2]],
                    "head": mentions[d[2]],
                    "relation": relations[d[3]],
                    "tail_id": wiki_ids[d[4]],
                    "tail": mentions[d[4]],
                }
            )
    with open(os.path.join(output_folder, f"test_pool_head_pred.txt.json"), "w") as f:
        json.dump(head_pred_json, f, indent=4)




