import torch
import argparse
import pytorch_lightning as pl
from main_kgt5 import process_deprecated
from pytorch_lightning.loggers import WandbLogger
from custom_data import NbhoodDataModule
from kgt5_model import KGT5_Model
#from kgt5_scoring_model import KGT5_Model
from omegaconf import DictConfig, OmegaConf, open_dict


def run(checkpoint_path: str, config_path: str, split: str) -> None:
    config = OmegaConf.load(config_path)
    print(OmegaConf.to_yaml(config))
    if ".hydra" in config_path:
        config.output_dir = config_path.split(".hydra")[0]
    else:
        config.output_dir = config_path.split("config.yaml")[0]
    print("output written to", config.output_dir)
    config = process_deprecated(config)
    config.eval.num_predictions = 500

    # dm = Wikidata5MModule(batch_size=256)
    dm = NbhoodDataModule(config=config)

    # model = KGT5_Model(tokenizer=tokenizer)
    model = KGT5_Model.load_from_checkpoint(
        checkpoint_path, config=config, data_module=dm
    )

    train_options = {
        'accelerator': config.train.accelerator,
        'devices': 1,
        'max_epochs': 1,
        'default_root_dir': config.output_dir,
        'strategy': config.train.strategy,
        'precision': config.train.precision,
        'check_val_every_n_epoch': config.valid.every,
    }

    trainer = pl.Trainer(**train_options, )  # limit_train_batches=1)

    if split=="test":
        trainer.test(model, ckpt_path=checkpoint_path, datamodule=dm)
    else:
        trainer.validate(model, ckpt_path=checkpoint_path, datamodule=dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test kgt5 model')
    parser.add_argument('-c', '--config', help='Path to config',
                        required=True)
    parser.add_argument('-m', '--model', help='Path to checkpoint',
                        required=True)
    parser.add_argument('-s', '--split', help='Split to evaluate on',
                        default="test")
    args = vars(parser.parse_args())
    torch.set_float32_matmul_precision('medium')
    run(args["model"], args["config"], split=args["split"])



