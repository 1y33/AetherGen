import src.trainer as trainer
import src.unet as unet
import torchvision
from torch.utils.data import DataLoader

import os
os.environ['MIOPEN_FIND_MODE'] = '2'

model_args = unet.UNetConfig(in_channels=3,time_dim=512,features=(64,256,512,1024))
trainer_args = trainer.TrainerConfig(epochs=1000,batch_size=128,learning_rate=5e-6,sample_epoch=200)


flowers_dataset = torchvision.datasets.Flowers102(root="data/", 
                                                  download=True,
                                                  transform=trainer_args.transform)

train_dataloader = DataLoader(
    flowers_dataset,
    batch_size=trainer_args.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=8,
)

trainer.train(trainer_args, model_args, train_dataloader)