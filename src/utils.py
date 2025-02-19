import matplotlib.pyplot as plt
import numpy as np

def plot_images(images, nrow=None, display=True, save_path=None):
    images = images.detach().cpu().numpy()
    
    batch_size = images.shape[0]
    rows = int(np.sqrt(batch_size))
    cols = int(np.ceil(batch_size / rows))
    
    plt.figure(figsize=(15, 15))
    
    for i in range(batch_size):
        plt.subplot(rows, cols, i + 1)
        img = np.transpose(images[i], (1, 2, 0))
        plt.imshow(img)
        plt.axis('off')  
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if display:
        plt.show()
    else:
        plt.close()
    
    
def model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")