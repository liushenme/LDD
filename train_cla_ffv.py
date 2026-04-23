import argparse

import toml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset.ffv import LAVDFDataModule
from model import *
from utils import LrLogger, EarlyStoppingLR
import os 
import json

parser = argparse.ArgumentParser(description="BATFD training")
parser.add_argument("--config", type=str)
parser.add_argument("--data_root", type=str)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--precision", default=32)
parser.add_argument("--num_train", type=int, default=None)
parser.add_argument("--num_val", type=int, default=1000)
parser.add_argument("--max_epochs", type=int, default=25)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--exp_dir", type=str, default=None)
parser.add_argument("--model", type=str, default=None)

if __name__ == '__main__':
    args = parser.parse_args()
    config = toml.load(args.config)

    print(args.model)
    print(args.batch_size)
    model = eval(args.model)

    dm = LAVDFDataModule(
        root=args.data_root,
        batch_size=args.batch_size, num_workers=args.num_workers,
        take_train=args.num_train, take_dev=args.num_val,
        get_meta_attr=model.get_meta_attr
    )


    model = model(
        v_encoder=config["model"]["video_encoder"]["type"],
        a_encoder=config["model"]["audio_encoder"]["type"],
        weight_decay=config["optimizer"]["weight_decay"],
        learning_rate=config["optimizer"]["learning_rate"],
        distributed=args.gpus > 1
    )

    try:
        precision = int(args.precision)
    except ValueError:
        pass
    
    checkpoint_dir = os.path.join(args.exp_dir, "checkpoints/")    

    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir, save_last=True, filename="{epoch}-{val_loss:.3f}",
        monitor="val_loss", mode="min", save_top_k=10, verbose=True
    )    


    trainer = Trainer(log_every_n_steps=50, precision=precision, max_epochs=args.max_epochs,
        callbacks=[
            checkpoint,
            #LrLogger(),
            #EarlyStoppingLR(lr_threshold=1e-7)
        ], enable_checkpointing=True,
        accelerator="auto",
        devices=args.gpus,
        strategy=None if args.gpus < 2 else "ddp",
        resume_from_checkpoint=args.resume,
        default_root_dir=args.exp_dir
    )

    trainer.fit(model, dm)
    
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(args.exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)


