import yaml
import os
import sys
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from models.unet_cond import Unet
from models.vqvae import VQVAE
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
import torch
from modules.utils import *
from modules.dataset import split_subject_data, read_json_file, ImageDataset, get_phantom_data
import matplotlib.pyplot as plt
from pytorch_fid import fid_score
from skimage.metrics import structural_similarity as ssim
from torch.nn.functional import mse_loss
import torchvision
from torchvision.utils import make_grid


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def eval(model, scheduler, train_config, diffusion_model_config,
         autoencoder_model_config, diffusion_config, dataset_config, vae, test_dataloader):
    """
    Evaluate the model by generating multiple images (4) for each test condition and computing metrics.
    Saves original and generated images, and computes FID, SSIM, and NMSE.
    """    
    # Create output directories
    output_dir = os.path.join(train_config['task_name'], 'evaluation_results')
    os.makedirs(output_dir, exist_ok=True)
    real_images_dir = os.path.join(output_dir, 'real_images')
    generated_images_dir = os.path.join(output_dir, 'generated_images')
    os.makedirs(real_images_dir, exist_ok=True)
    os.makedirs(generated_images_dir, exist_ok=True)
    
    # Lists to store metrics
    ssim_scores = []
    nmse_scores = []
    
    # Calculate image size
    im_size = dataset_config['im_size'] // (2 ** sum(autoencoder_model_config['down_sample']))
    cf_guidance_scale = get_config_value(train_config, 'cf_guidance_scale', 1.0)
    num_generations = 4  # Number of images to generate per input
    
    # Process each batch in test_dataloader
    for batch_idx, data in enumerate(test_dataloader):
        with torch.no_grad():
            if len(data) == 3:  # If data contains conditions
                real_images, edge, mask = data
                edge = edge.unsqueeze(1)
                conditions = torch.cat([edge, mask], dim=1).float().to(device)
                cond_input = {'image': conditions}
            else:
                real_images = data
                cond_input = {'image': None}
            
            real_images = real_images.float().to(device)
            
            # Generate multiple images for each input
            all_generated_images = []
            for gen_idx in range(num_generations):
                # Generate images with different random seeds
                xt = torch.randn((1,  # batch size is 1
                                autoencoder_model_config['z_channels'],
                                im_size,
                                im_size)).to(device)
                
                # Sampling loop
                for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
                    t = torch.ones((1,)).long().to(device) * i  # batch size is 1
                    noise_pred = model(xt, t, cond_input)
                    
                    if cf_guidance_scale > 1:
                        uncond_input = {'image': torch.zeros_like(cond_input['image'])}
                        noise_pred_uncond = model(xt, t, uncond_input)
                        noise_pred = noise_pred_uncond + cf_guidance_scale * (noise_pred - noise_pred_uncond)
                    
                    xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
                
                # Decode final images
                generated_images = vae.decode(xt)
                generated_images = torch.clamp(generated_images, -1., 1.)
                generated_images = (generated_images + 1.) / 2.  # scale to [0,1]
                all_generated_images.append(generated_images)
            
            # Stack all generated images
            all_generated_images = torch.cat(all_generated_images, dim=0)  # [num_generations, C, H, W]
            real_images = (real_images + 1.) / 2.  # scale to [0,1]
            
            # Calculate metrics for each generated image
            real_np = real_images[0].cpu().numpy()  # Take first (and only) image
            for gen_idx in range(num_generations):
                gen_np = all_generated_images[gen_idx].cpu().numpy()
                
                # SSIM calculation
                try:
                    r_img = real_np.transpose(1, 2, 0)
                    g_img = gen_np.transpose(1, 2, 0)
                    
                    ssim_score = ssim(r_img, g_img,
                                    win_size=min(7, r_img.shape[0]-1, r_img.shape[1]-1),
                                    channel_axis=2,
                                    data_range=1.0)
                    ssim_scores.append(ssim_score)
                except Exception as e:
                    print(f"SSIM calculation failed: {e}")
                    continue
                
                # NMSE calculation
                nmse = mse_loss(torch.tensor(real_np), torch.tensor(gen_np)).item() / torch.var(torch.tensor(real_np)).item()
                nmse_scores.append(nmse)
            
            # Create figure for visualization
            if len(data) == 3:
                # Row 1: Original + edge map + semantic map
                # Row 2: 4 generated images
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                axes = axes.flatten()
                
                # First row: Original and condition maps
                # Plot original image
                real_img = real_images[0].cpu()
                axes[0].imshow(real_img.squeeze() * 255., cmap='gray')
                axes[0].set_title('Original')
                axes[0].axis('off')
                
                # Plot edge map
                edge_map = edge[0].cpu()
                axes[1].imshow(edge_map[0] * 255, cmap='gray')
                axes[1].set_title('Edge Map')
                axes[1].axis('off')
                
                # Plot semantic mask
                sem_map = torch.argmax(mask[0].cpu(), dim=0)
                axes[2].imshow(sem_map, cmap='nipy_spectral')
                axes[2].set_title('Semantic Map')
                axes[2].axis('off')
                
                # Turn off the unused subplot in first row
                axes[3].axis('off')
                
                # Second row: Generated images
                for gen_idx in range(num_generations):
                    gen_img = all_generated_images[gen_idx].cpu()
                    axes[gen_idx + 4].imshow(gen_img.squeeze() * 255., cmap='gray')
                    axes[gen_idx + 4].set_title(f'Generated {gen_idx + 1}')
                    axes[gen_idx + 4].axis('off')
                
            else:
                # For cases without conditions: single row with 5 images
                fig, axes = plt.subplots(1, 5, figsize=(20, 4))
                
                # Plot original image
                real_img = real_images[0].cpu()
                axes[0].imshow(real_img.squeeze() * 255., cmap='gray')
                axes[0].set_title('Original')
                axes[0].axis('off')
                
                # Plot generated images
                for gen_idx in range(num_generations):
                    gen_img = all_generated_images[gen_idx].cpu()
                    axes[gen_idx + 1].imshow(gen_img.squeeze() * 255., cmap='gray')
                    axes[gen_idx + 1].set_title(f'Generated {gen_idx + 1}')
                    axes[gen_idx + 1].axis('off')
            
            plt.tight_layout()
            os.makedirs(os.path.join(train_config['task_name'], 'phantom_evaluation_results'), exist_ok=True)
            plt.savefig(os.path.join(train_config['task_name'], 'phantom_evaluation_results',
                                   f'comparison_batch_{batch_idx}.pdf'))
            plt.close(fig)
            
            # Save individual images for FID calculation
            # Save real image
            r_i = torchvision.transforms.ToPILImage()(real_images[0])
            r_i.save(os.path.join(real_images_dir, f'real_{batch_idx}.png'))
            r_i.close()
            
            # Save all generated images
            for gen_idx in range(num_generations):
                g_i = torchvision.transforms.ToPILImage()(all_generated_images[gen_idx])
                g_i.save(os.path.join(generated_images_dir, f'gen_{batch_idx}_sample_{gen_idx}.png'))
                g_i.close()
    
    # Calculate FID score
    fid = fid_score.calculate_fid_given_paths([real_images_dir, generated_images_dir],
                                            batch_size=50,
                                            device=device,
                                            dims=2048)
    
    # Calculate average metrics
    avg_ssim = np.mean(ssim_scores)
    avg_nmse = np.mean(nmse_scores)
    
    # Save metrics
    metrics = {
        'FID': fid,
        'Average SSIM': avg_ssim,
        'Average NMSE': avg_nmse
    }
    
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        for metric_name, value in metrics.items():
            f.write(f'{metric_name}: {value}\n')
    
    print("Evaluation Results:")
    print(f"FID Score: {fid}")
    print(f"Average SSIM: {avg_ssim}")
    print(f"Average NMSE: {avg_nmse}")
    
    return metrics

def infer(config_path):
    # Read the config file #
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']

    # Set the desired seed value #
    seed = train_config['seed']
    
    ########## Create the noise scheduler #############
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    ###############################################
    
    ############# Validate the config #################
    # Instantiate Condition related components
    condition_types = []
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    if condition_config is not None:
        assert 'condition_types' in condition_config, \
            "This test script only supports image conditioning"
        condition_types = condition_config['condition_types']

    # Create the dataset
    im_path = 'processed_data/diffusion'

    data_indices_json = read_json_file(os.path.join(im_path, 'dataset_index_mapping.json'))

    test_indices = get_phantom_data(data_indices_json)

    test_dataset = ImageDataset(test_indices, im_path, data_indices_json, 'diffusion')

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    ###############################################
    
    ########## Load Unet #############
    model = Unet(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)
    model.eval()

    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['ldm_cond_ckpt_name'])):
        print('Loaded unet checkpoint')
        model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                      train_config['ldm_cond_ckpt_name']),
                                         map_location=device))
    else:
        raise Exception('Model checkpoint {} not found'.format(os.path.join(train_config['task_name'],
                                                                   train_config['ldm_cond_ckpt_name'])))
    #####################################
    ########## Load VQVAE #############
    vae = VQVAE(im_channels=dataset_config['im_channels'],
                model_config=autoencoder_model_config).to(device)
    vae.eval()
    
    # Load vae if found
    if os.path.exists(os.path.join(train_config['task_name'],
                                   train_config['vqvae_autoencoder_ckpt_name'])):
        print('Loaded vae checkpoint')
        vae.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                    train_config['vqvae_autoencoder_ckpt_name']),
                                       map_location=device))
    else:
        raise Exception('VAE checkpoint {} not found'.format(os.path.join(train_config['task_name'],
                                                                          train_config['vqvae_autoencoder_ckpt_name'])))
    #####################################
    
    with torch.no_grad():
        eval(model, scheduler, train_config, diffusion_model_config,
               autoencoder_model_config, diffusion_config, dataset_config, vae, test_dataloader)


if __name__ == '__main__':
    infer('config/config.yaml')
