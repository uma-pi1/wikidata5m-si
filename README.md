# A Benchmark for Semi-Inductive Link Prediction in Knowledge Graphs

This is the benchmark, code, and configuration accompanying the paper [A Benchmark for Semi-Inductive Link Prediction in Knowledge Graphs]().

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
curl -O https://web.informatik.uni-mannheim.de/pi1/kge-datasets/wikidata5m_v3_semi_inductive.tar.gz
tar -zxvf wikidata5m_v3_semi_inductive.tar.gz
```

#### Download mention and description embedding

```
cd data/wikidata5m_v3_semi_inductive
curl -O https://web.informatik.uni-mannheim.de/pi1/kge-datasets/wikidata5m-si/description_embs.pt
curl -O https://web.informatik.uni-mannheim.de/pi1/kge-datasets/wikidata5m-si/mention_embs.pt
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


