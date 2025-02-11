from .unet import ModifiedUNet
from .diffuser import DiffusionModel
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from .utils import plot_images, model_params

class TrainerConfig:
    def __init__(
        self,
        device="cuda" if torch.cuda.is_available() else "cpu",
        epochs=500,
        batch_size=64,
        learning_rate=1e-3,
        sample_epoch=100,
        num_samples=4,
        image_size=64,
    ):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.sample_epoch = sample_epoch
        self.num_samples = num_samples
        self.image_size = image_size
        
        # Define the transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1,0.1), scale=(0.5,1.5), shear=10),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))], p=0.1),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)


def train(config: TrainerConfig,
          model_config: ModifiedUNet,
          train_dataloader: DataLoader,
          ):
    
    device = config.device
    model= ModifiedUNet(model_config).to(device)
    model_params(model)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    mse = nn.MSELoss()
    diffusion = DiffusionModel(device=device)
    
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0
        
        pbar = tqdm(train_dataloader,
                   desc=f'Epoch {epoch}',
                   leave=True,
                   position=0)
        
        for images, _ in pbar:
            images = images.to(device)
            
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({'loss': f'{running_loss/(pbar.n+1):.4f}'})
        
        pbar.close()
        
        print(f"Epoch {epoch} completed | Average Loss: {running_loss/len(train_dataloader):.4f}")

        if epoch % config.sample_epoch == 0:
            sampled_images = diffusion.sample(model, n=config.num_samples)
            plot_images(sampled_images)

