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
trainer_args = trainer.TrainerConfig(epochs=1000,batch_size=128,learning_rate=1e-4,sample_epoch=200)

model = unet.ModifiedUNet(model_args)
model.load_state_dict(torch.load("model_epoch_800.pth"))
model.to("cuda")
diffusion_model = diffuser.DiffusionModel()


utils.plot_images(
    diffusion_model.sample(model, trainer_args.num_samples),
    display=False,
    save_path="generated_samples.png"
)

# flowers_dataset = torchvision.datasets.Flowers102(root="data/", 
#                                                   download=True,
#                                                   transform=trainer_args.transform)

# train_dataloader = DataLoader(
#     flowers_dataset,
#     batch_size=trainer_args.batch_size,
#     shuffle=True,
#     pin_memory=True,
#     num_workers=8,
# )

# trainer.train(trainer_args, model_args, train_dataloader)
