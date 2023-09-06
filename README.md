# KGT5-context

This is an extension of the model [KGT5-context]() for semi-inductive link prediction.


## Getting Started

```
git clone git@github.com:uma-pi1/wikidata5m-si.git
cd wikidata5m-si
git checkout kgt5-context
conda create -n kgt5 python==3.10
conda activate kgt5
pip install -r requirements.txt
```

### Download Data

```
mkdir data
cd data
curl -O https://web.informatik.uni-mannheim.de/pi1/kge-datasets/wikidata5m_v3_semi_inductive.tar.gz
tar -zxvf wikidata5m_v3_semi_inductive.tar.gz
```


## Reproduction

### Training

To train the KGT5-context on Wikidata5M-SI, run the following command.
Note, this library will automatically use all available GPUs.
You can control the GPUs used with the environment variable `CUDA_VISIBLE_DEVICES=0,1,2,3`

```
python main_kgt5.py dataset.name=wikidata5m_v3_semi_inductive train.max_epochs=6
```

If you want to utilize descriptions (provided with the dataset), run

```
python main.py dataset.name=wikidata5m_v3 train.max_epochs=6 descriptions.use=True
```

If you want to use context-hiding during training, run

```
python main_kgt5.py dataset.name=wikidata5m_v3_semi_inductive train.max_epochs=6 train.context.dropout.percentage=0.5
```

If you want to train the original KGT5 without context use

```
python main_kgt5.py dataset=wikidata5m dataset.v1=True
```

### Evaluation

#### Transductive

To evaluate the model in a transductive setting run

```
python eval.py --config <path to config> --model <path to trained model>
```

#### Semi-Inductive

To evaluate the model in a semi-inductive setting run

```
python eval_few_shot.py --config <path to config> --model <path to trained model> --num_shots <num shots> --context selection <context selection>
```

To evaluate all semi-inductive settings run

```
python eval_all_few_shot.py --config <path to config> --model <path to trained model> --num_shots <num shots> --context selection <context selection>
```

## Wandb
This library supports logging via wandb.
If you want to use it use the option `use_wandb=True`

Note, output directory and wandb project name are defined in the file conf/config.yaml.


