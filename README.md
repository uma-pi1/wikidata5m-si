# A Benchmark for Semi-Inductive Link Prediction in Knowledge Graphs

This is the benchmark, code, and configuration accompanying the paper [A Benchmark for Semi-Inductive Link Prediction in Knowledge Graphs]().

## Hitter

This branch holds the code for the model Hitter.
It is an extension of [LibKGE](https://github.com/uma-pi1/kge),

### Setup

```
git clone https://github.com/uma-pi1/wikidata5m-si.git
cd wikidata5m-si
git checkout hitter
pip install -e .
```

#### Download data

```
mkdir data
cd data
curl -O https://web.informatik.uni-mannheim.de/pi1/kge-datasets/wikidata5m-si.tar.gz
tar -zxvf wikidata5m-si.tar.gz
```

### Training

```
python -m kge start config_hitter.yaml
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


