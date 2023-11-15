# A Benchmark for Semi-Inductive Link Prediction in Knowledge Graphs

This is the benchmark, code, and configuration accompanying the paper [A Benchmark for Semi-Inductive Link Prediction in Knowledge Graphs](https://arxiv.org/pdf/2310.11917.pdf).

## DistMult + ERAvg

This branch holds the code for the model DistMult + ERAvg + Mentions/Descriptions.
It is an extension of [LibKGE](https://github.com/uma-pi1/kge),

### Setup

```
git clone https://github.com/uma-pi1/wikidata5m-si.git
cd wikidata5m-si
git checkout odistmult_descriptions
pip install -e .
```

#### Download data

```
mkdir data
cd data
curl -O https://madata.bib.uni-mannheim.de/424/2/wikidata5m-si.tar.gz
tar -zxvf wikidata5m-si.tar.gz
```

#### Create mention and description embeddings

```
python create_mpnet_embeddings.py --mentions_only
python create_mpnet_embeddings.py
```

### Training

```
python -m kge start config_distmult_eravg_mentions.yaml
```

```
python -m kge start config_distmult_eravg_descriptions.yaml
```

### Evaluation

#### Transductive

```
python -m kge test <path to config in output train folder>
```


#### Semi-Inductive

```
bash eval_all_semi_inductive.sh <path to config in output train folder>
```


