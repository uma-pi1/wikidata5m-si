import torch
import argparse
import pytorch_lightning as pl
from main_kgt5 import process_deprecated
from custom_data import NbhoodDataModule
from kgt5_model import KGT5_Model
from omegaconf import OmegaConf


def run(checkpoint_path: str, config_path: str, num_shots: int, split: str, context_selection: str) -> None:
    config = OmegaConf.load(config_path)
    if ".hydra" in config_path:
        config.output_dir = config_path.split(".hydra")[0]
    else:
        config.output_dir = config_path.split("config.yaml")[0]
    print("output written to", config.output_dir)
    config = process_deprecated(config)
    if not "few_shot" in config:
        config.eval["few_shot"] = {
            "use": False,
            "num_shots": 10,
        }
    config.eval.few_shot.use = True
    config.eval.few_shot.num_shots = num_shots
    config.eval.few_shot.context_selection = context_selection
    config.eval.num_predictions = 500
    config.context.dropout.percentage = 0.0
    print(OmegaConf.to_yaml(config))

    # dm = Wikidata5MModule(batch_size=256)
    dm = NbhoodDataModule(config=config, split=split)

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

    trainer = pl.Trainer(**train_options)

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
    parser.add_argument('-k', '--num_shots', help='num shots to use',
                        default=10, type=int)
    parser.add_argument('-cs', '--context_selection', help='[most_common, least_common, random]',
                        default='most_common', type=str)
    args = vars(parser.parse_args())
    torch.set_float32_matmul_precision('medium')
    run(args["model"], args["config"], num_shots=args["num_shots"], split=args["split"], context_selection=args["context_selection"])



