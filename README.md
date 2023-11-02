# A Benchmark for Semi-Inductive Link Prediction in Knowledge Graphs

This is the benchmark, code, and configuration accompanying the paper [A Benchmark for Semi-Inductive Link Prediction in Knowledge Graphs](https://arxiv.org/pdf/2310.11917.pdf).
The main branch holds code/information about the benchmark itself. 
The following branches hold code and configuration for the separate models evaluated in the study.

- [KGT5 \& KGT5-context](https://github.com/uma-pi1/wikidata5m-si/tree/kgt5-context)
- [ComplEx + Bias + FoldIn](https://github.com/uma-pi1/wikidata5m-si/tree/complex_fold_in)
- [DisMult ERAvg](https://github.com/uma-pi1/wikidata5m-si/tree/odistmult)
- [DisMult ERAvg + Mention/Description](https://github.com/uma-pi1/wikidata5m-si/tree/odistmult_descriptions)
- [HittER](https://github.com/uma-pi1/wikidata5m-si/tree/hitter)


## Benchmark

### Download data

```
mkdir data
cd data
curl -O https://web.informatik.uni-mannheim.de/pi1/kge-datasets/wikidata5m-si.tar.gz
tar -zxvf wikidata5m-si.tar.gz
```

### Generate Few Shot Tasks

- use the file `prepare_few_shot.py`
- create a `few_shot_set_creator` object
	- `dataset_name`: (str) name of the dataset
      - default: wikidata5m_v3_semi_inductive
	- `use_invese`: (bool) whether to use inverse relations
      - default: False
      - if True: for all triples where the unseen entity is in the object slot, increase relation id by num-relations and invert triple
	- `split`: (str) which split to use
      - default: valid
	- `context_selection`: (str) which context\_selection technique to use
      - default: most\_common
      - options: most\_common, least\_common, random

```
few_shot_set_creator = FewShotSetCreator(
	dataset_name="wikidata5m_v3_semi_inductive",
	use_inverse=True,
	split="test"
)
```

- generate the data using the `few_shot_set_creator`
	- `num_shots`: (int) the number of shots to use (between 0 and 10)

```
data = few_shot_set_creator.create_few_shot_dataset(num_shots=5)
```

- evaluation is performed in direction unseen to seen
- output format looks like this
```
[
{
	"unseen_entity": <id of unseen entity>,
	"unseen_slot": <slot of unseen entity: 0 for head/subject, 2 for tail/object>,
	"triple: <[s, p, o]>,
	"context: <[unseen_entity_id, unseen_entity_slot, s, p, o]>
},
...

]


# Create Benchmarks Based on Other Graphs

- to create similar benchmark based on other graphs use the file `create_semi_inductive_dataset.py`
- this file was used to create wikidata5m-si based on wikidata5m

# How to Cite

- if you use the proposed benchmark, the provided code or insights presented in the paper please cite.

```
@inproceedings{kochsiek2023benchmark,                                                                                                                                                                  
title={A Benchmark for Semi-Inductive Link Prediction in Knowledge Graphs},
author={Kochsiek, Adrian and Gemulla, Rainer},
booktitle={Findings of the Association for Computational Linguistics: EMNLP 2023},
year={2023}
}
```


