## SimKGC

This branch contains the slightly adapted code of the paper
"[SimKGC: Simple Contrastive Knowledge Graph Completion with Pre-trained Language Models](https://aclanthology.org/2022.acl-long.295.pdf)".

## Requirements
* python>=3.7
* torch>=1.6 (for mixed precision training)
* transformers>=4.15
* pandas
* tqdm

### Preprocess

```
python convert_data_to_simkgc_format.py
```


### Train

```
OUTPUT_DIR=./checkpoint/wikidata5m-si/ bash scripts/train_wikidata5m_si.sh
```

### Evaluate Transductive

```
bash scripts/eval_wikidata5m_si_transductive.sh ./checkpoint/wikidata5m_si/model_last.mdl
```

### Evaluate Semi-Inductive

We evaluate separately for head- and tail-prediction.
SimKGC outputs automatically performance for both directions separately.
For calculation of final MRR calculate the weighted average of the corresponding head and tail results.
(Weight in terms of triples per test file.)

```
bash scripts/eval_wikidata5m_si_head_pred.sh ./checkpoint/wikidata5m_si/model_last.mdl
```

```
bash scripts/eval_wikidata5m_si_tail_pred.sh ./checkpoint/wikidata5m_si/model_last.mdl
```
