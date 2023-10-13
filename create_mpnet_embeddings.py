import argparse
import torch
from transformers import AutoTokenizer, MPNetModel
from tqdm import tqdm
import gc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some data.')
    parser.add_argument('--mentions_only', action='store_true', help='Set if you want to process mentions only')
    args = parser.parse_args()
    mentions_only = args.mentions_only

    dataset_name = "wikidata5m-si"

    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")
        model = MPNetModel.from_pretrained("microsoft/mpnet-base").cuda()


        entity_mentions = []
        entity_descriptions = []
        with open(f"data/{wikidata5m-si}/all_entity_mentions.del") as f:
            for line in f.readlines():
                entity_mentions.append(line.strip().split("\t")[1])
        with open(f"data/{wikidata5m-si}/all_entity_desc.del") as f:
            for line in f.readlines():
                entity_descriptions.append(line.strip().split("\t")[1])

        sentence_embs = []
        batch = []
        for i, (mention, description) in tqdm(enumerate(zip(entity_mentions, entity_descriptions)), total=len(entity_mentions)):
            input_text = f"{mention}, {description}"
            if mentions_only:
                input_text = mention
            batch.append(input_text)
            if i % 128 == 0 and i != 0:
                inputs = tokenizer(batch, return_tensors="pt", add_special_tokens=True, max_length=512, padding=True, truncation=True)
                for key, value in inputs.data.items():
                    inputs.data[key] = value.cuda()
                outputs = model(**inputs)
                last_hidden_states = outputs.last_hidden_state
                sentence_emb = last_hidden_states[:, 0, :]
                sentence_embs.append(sentence_emb.cpu())
                batch = []
        if len(batch) > 0:
            inputs = tokenizer(batch, return_tensors="pt", add_special_tokens=True, max_length=512, padding=True, truncation=True)
            for key, value in inputs.data.items():
                inputs.data[key] = value.cuda()
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            sentence_emb = last_hidden_states[:, 0, :]
            sentence_embs.append(sentence_emb.cpu())

        sentence_embs = torch.cat(sentence_embs, dim=0)
        if mentions_only:
            torch.save(sentence_embs, f"data/{wikidata5m-si}/mentions_embs.pt")
        else:
            torch.save(sentence_embs, f"data/{wikidata5m-si}/descriptions_embs.pt")
