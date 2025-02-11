import src.trainer as trainer
import src.unet as unet
import torchvision
from torch.utils.data import DataLoader


model_args = unet.UNetConfig(in_channels=3,time_dim=512,features=(64, 128, 256, 512))
trainer_args = trainer.TrainerConfig(learning_rate=1e-3,sample_epoch=50)


flowers_dataset = torchvision.datasets.Flowers102(root="data/", 
                                                  download=True,
                                                  transform=trainer_args.transform)
train_dataloader = DataLoader(
    flowers_dataset,
    batch_size=trainer_args.batch_size,
    shuffle=True
)
trainer.train(trainer_args, model_args, train_dataloader)


