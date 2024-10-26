import yaml
import torch
import random
import torchvision
import os
import numpy as np
from tqdm import tqdm
from models.vqvae import VQVAE
from models.lpips import LPIPS
from models.discriminator import Discriminator
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from torchvision.utils import make_grid
from modules.dataset import split_subject_data, read_json_file, ImageDataset
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(config_path):
    # Read the config file #
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    
    dataset_config = config['dataset_params']
    autoencoder_config = config['autoencoder_params']
    train_config = config['train_params']
    
    # Set the desired seed value #
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    #############################
    
    # Create the model and dataset #
    model = VQVAE(im_channels=dataset_config['im_channels'],
                  model_config=autoencoder_config).to(device)
    
    # Create the dataset
    data_indices_json = read_json_file(os.path.join(dataset_config['im_path'], 'dataset_index_mapping.json'))

    train_indices, test_indices = split_subject_data(data_indices_json, num_test_subjects=train_config['leave_out_subjects'], random_seed=seed)

    train_dataset = ImageDataset(train_indices, dataset_config['im_path'], data_indices_json, 'vqvae')

    train_dataloader = DataLoader(train_dataset, batch_size=train_config['autoencoder_batch_size'], shuffle=True)
    
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
        
    num_epochs = train_config['autoencoder_epochs']

    # L1/L2 loss for Reconstruction
    recon_criterion = torch.nn.MSELoss()
    # Disc Loss can even be BCEWithLogits
    disc_criterion = torch.nn.MSELoss()
    
    # No need to freeze lpips as lpips.py takes care of that
    lpips_model = LPIPS().eval().to(device)
    discriminator = Discriminator(im_channels=dataset_config['im_channels']).to(device)
    
    optimizer_d = Adam(discriminator.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    optimizer_g = Adam(model.parameters(), lr=train_config['autoencoder_lr'], betas=(0.5, 0.999))
    
    disc_step_start = train_config['disc_start']
    step_count = 0
    
    # This is for accumulating gradients incase the images are huge
    # And one cant afford higher batch sizes
    acc_steps = train_config['autoencoder_acc_steps']
    image_save_steps = train_config['autoencoder_img_save_steps']
    img_save_count = 0
    
    for epoch_idx in range(num_epochs):
        recon_losses = []
        codebook_losses = []
        #commitment_losses = []
        perceptual_losses = []
        disc_losses = []
        gen_losses = []
        losses = []
        
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        
        for im in tqdm(train_dataloader):
            step_count += 1
            im = im.float().to(device)
            
            # Fetch autoencoders output(reconstructions)
            model_output = model(im)
            output, z, quantize_losses = model_output
            
            # # Image Saving Logic
            # if step_count % image_save_steps == 0 or step_count == 1:
            #     sample_size = min(8, im.shape[0])
            #     save_output = torch.clamp(output[:sample_size], -1., 1.).detach().cpu()
            #     save_output = ((save_output + 1) / 2)
            #     save_input = ((im[:sample_size] + 1) / 2).detach().cpu()
                
            #     grid = make_grid(torch.cat([save_input, save_output], dim=0), nrow=sample_size)
            #     img = torchvision.transforms.ToPILImage()(grid)
            #     if not os.path.exists(os.path.join(train_config['task_name'],'vqvae_autoencoder_samples')):
            #         os.mkdir(os.path.join(train_config['task_name'], 'vqvae_autoencoder_samples'))
            #     img.save(os.path.join(train_config['task_name'],'vqvae_autoencoder_samples',
            #                           'current_autoencoder_sample_{}.png'.format(img_save_count)))
            #     img_save_count += 1
            #     img.close()

            # Image Saving Logic
            if step_count % image_save_steps == 0 or step_count == 1:
                sample_size = min(8, im.shape[0])
                # Just detach and move to CPU without normalization
                save_output = output[:sample_size].detach().cpu()
                save_input = im[:sample_size].detach().cpu()
                
                # Create a figure with subplots for each pair of images
                n_samples = sample_size
                fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 3, 6))
                
                # Plot original images on top row
                for i in range(n_samples):
                    axes[0, i].imshow(save_input[i].squeeze(), cmap='gray')
                    axes[0, i].set_title(f'Original {i+1}')
                    axes[0, i].axis('off')
                
                # Plot reconstructed images on bottom row
                for i in range(n_samples):
                    axes[1, i].imshow(save_output[i].squeeze(), cmap='gray')
                    axes[1, i].set_title(f'Reconstructed {i+1}')
                    axes[1, i].axis('off')
                
                plt.tight_layout()
                
                # Create directory if it doesn't exist
                if not os.path.exists(os.path.join(train_config['task_name'],'vqvae_autoencoder_samples')):
                    os.mkdir(os.path.join(train_config['task_name'], 'vqvae_autoencoder_samples'))
                
                # Save and close figure
                plt.savefig(os.path.join(train_config['task_name'],'vqvae_autoencoder_samples',
                                      'current_autoencoder_sample_{}.png'.format(img_save_count)))
                plt.close(fig)  # Important to prevent memory leaks
                img_save_count += 1
            
            ######### Optimize Generator ##########
            # L2 Loss
            recon_loss = recon_criterion(output, im) 
            recon_losses.append(recon_loss.item())
            recon_loss = recon_loss / acc_steps
            g_loss = (recon_loss +
                      (train_config['codebook_weight'] * quantize_losses['codebook_loss'] / acc_steps) +
                      (train_config['commitment_beta'] * quantize_losses['commitment_loss'] / acc_steps))
            codebook_losses.append(train_config['codebook_weight'] * quantize_losses['codebook_loss'].item())
            # Adversarial loss only if disc_step_start steps passed
            if step_count > disc_step_start:
                disc_fake_pred = discriminator(model_output[0])
                disc_fake_loss = disc_criterion(disc_fake_pred,
                                                torch.ones(disc_fake_pred.shape,
                                                           device=disc_fake_pred.device))
                gen_losses.append(train_config['disc_weight'] * disc_fake_loss.item())
                g_loss += train_config['disc_weight'] * disc_fake_loss / acc_steps
            lpips_loss = torch.mean(lpips_model(output, im)) / acc_steps
            perceptual_losses.append(train_config['perceptual_weight'] * lpips_loss.item())
            g_loss += train_config['perceptual_weight']*lpips_loss / acc_steps
            losses.append(g_loss.item())
            g_loss.backward()
            #####################################
            
            ######### Optimize Discriminator #######
            if step_count > disc_step_start:
                fake = output
                disc_fake_pred = discriminator(fake.detach())
                disc_real_pred = discriminator(im)
                disc_fake_loss = disc_criterion(disc_fake_pred,
                                                torch.zeros(disc_fake_pred.shape,
                                                            device=disc_fake_pred.device))
                disc_real_loss = disc_criterion(disc_real_pred,
                                                torch.ones(disc_real_pred.shape,
                                                           device=disc_real_pred.device))
                disc_loss = train_config['disc_weight'] * (disc_fake_loss + disc_real_loss) / 2
                disc_losses.append(disc_loss.item())
                disc_loss = disc_loss / acc_steps
                disc_loss.backward()
                if step_count % acc_steps == 0:
                    optimizer_d.step()
                    optimizer_d.zero_grad()
            #####################################
            
            if step_count % acc_steps == 0:
                optimizer_g.step()
                optimizer_g.zero_grad()
        optimizer_d.step()
        optimizer_d.zero_grad()
        optimizer_g.step()
        optimizer_g.zero_grad()
        if len(disc_losses) > 0:
            print(
                'Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | '
                'Codebook : {:.4f} | G Loss : {:.4f} | D Loss {:.4f}'.
                format(epoch_idx + 1,
                       np.mean(recon_losses),
                       np.mean(perceptual_losses),
                       np.mean(codebook_losses),
                       np.mean(gen_losses),
                       np.mean(disc_losses)))
        else:
            print('Finished epoch: {} | Recon Loss : {:.4f} | Perceptual Loss : {:.4f} | Codebook : {:.4f}'.
                  format(epoch_idx + 1,
                         np.mean(recon_losses),
                         np.mean(perceptual_losses),
                         np.mean(codebook_losses)))
        
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['vqvae_autoencoder_ckpt_name']))
        torch.save(discriminator.state_dict(), os.path.join(train_config['task_name'],
                                                            train_config['vqvae_discriminator_ckpt_name']))
    print('Done Training...')


if __name__ == '__main__':
    train('config/config.yaml')
