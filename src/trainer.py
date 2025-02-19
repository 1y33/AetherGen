from .unet import ModifiedUNet
from .diffuser import DiffusionModel
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from .utils import plot_images, model_params
from torch.optim.lr_scheduler import CosineAnnealingLR

class TrainerConfig:
    def __init__(
        self,
        device="cuda" if torch.cuda.is_available() else "cpu",
        epochs=500,
        batch_size=64,
        learning_rate=1e-4,       # Updated learning rate
        sample_epoch=50,          # More frequent sampling
        num_samples=4,
        image_size=64,
        name="model"
    ):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.weight_decay = learning_rate * 0.01  # 1e-6 weight decay
        self.sample_epoch = sample_epoch
        self.num_samples = num_samples
        self.image_size = image_size
        self.eta_min = 1e-6  # Minimum learning rate for cosine annealing
        self.name=name
        # Define the transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),  # Mild spatial augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
            # transforms.RandomApply([
            #     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01)
            #     ], p=0.3)
])


    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

def model_save(model, epoch,name):
    """
    Save the model's state dictionary to a file.
    """
    torch.save(model.state_dict(), f"{name}_epoch_{epoch}.pth")
    print(f"Model saved at epoch {epoch}")

def train(config: TrainerConfig,
          model_config: ModifiedUNet,
          train_dataloader: DataLoader,
          weights_path: str = None):
    
    device = config.device
    model = ModifiedUNet(model_config).to(device)
    model_params(model)
    
    model.load_state_dict(torch.load(weights_path)) if weights_path else None
    
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)    
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.eta_min)

    mse = nn.MSELoss()
    diffusion = DiffusionModel(device=device)
    
    losses = []
    
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
        
        scheduler.step()
        print(f"Epoch {epoch} completed | Average Loss: {running_loss/len(train_dataloader):.4f}")

        losses.append(running_loss/len(train_dataloader))
        
        if epoch % config.sample_epoch == 0:
            model_save(model, epoch,config.name)
            print(f"Model saved at epoch {epoch}")
            sampled_images = diffusion.sample(model, n=config.num_samples)
            plot_images(sampled_images)

    return losses