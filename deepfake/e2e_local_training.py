import sys
sys.path.insert(1, '/home/ubuntu/KagglePlaygrounds/deepfake/code')

from utils import *

config = {"lr": 0.0001,
          "batch_size": 16,
          "num_workers": 8,
          "milestone": 5,
          "epochs": 20,
          "backbone": "x3d_xs",
          "unfreeze_top_layers": 3,
          "resize": False,}

model = DeepFakeModel(backbone=config["backbone"], 
                      lr=config["lr"], 
                      milestone=config["milestone"])

data = DeepFakeDataModule(data_path="/home/ubuntu/KagglePlaygrounds/deepfake", 
                          backbone=config["backbone"], 
                          batch_size=config["batch_size"], 
                          num_workers=config["num_workers"], 
                          resize=config["resize"])

finetuning_callback = MilestonesFinetuning(milestone=config["milestone"], 
                                           unfreeze_top_layers=config["unfreeze_top_layers"],
                                           train_bn=False)

wandb_logger = WandbLogger(project='deepfake', 
                           offline=False, 
                           name=config["backbone"], 
                           config=config)

trainer = pl.Trainer(gpus=1, 
                     max_epochs=config["epochs"], 
                     callbacks=[finetuning_callback],
                     logger=wandb_logger,
                     log_every_n_steps=25,
                     amp_level='O2', 
                     precision=16,
                    )

trainer.fit(model, data)