import src.trainer as trainer
import src.unet as unet
import torchvision
from torch.utils.data import DataLoader
import torch
import src.diffuser as diffuser
import os
import src.utils as utils

os.environ['MIOPEN_FIND_MODE'] = '2'

model_args = unet.UNetConfig(in_channels=3,time_dim=512,features=(64,256,512,1024,2048))
trainer_args = trainer.TrainerConfig(epochs=301,batch_size=512,learning_rate=1e-6,sample_epoch=50)

 
 
model = unet.ModifiedUNet(model_args).to(trainer_args.device)
model.load_state_dict(torch.load("model_epoch_100.pth"))
model.eval()
diffusion_model = diffuser.DiffusionModel(noise_steps=1000, img_size=trainer_args.image_size, device=trainer_args.device)
utils.plot_images(diffusion_model.sample(model, 81), nrow=9, save_path="sample.png")



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

trainer.train(trainer_args, model_args, train_dataloader,"model_epoch_800.pth")
