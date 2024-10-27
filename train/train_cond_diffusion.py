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
from modules.dataset import split_subject_data, read_json_file, ImageDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
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
    
    # Instantiate Condition related components
    condition_types = []
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    if condition_config is not None:
        assert 'condition_types' in condition_config, \
            "condition type missing in conditioning config"
        condition_types = condition_config['condition_types']
            
    # Create the dataset
    data_indices_json = read_json_file(os.path.join(dataset_config['im_path'], 'dataset_index_mapping.json'))

    train_indices, test_indices = split_subject_data(data_indices_json, num_test_subjects=train_config['leave_out_subjects'], random_seed=seed)

    train_dataset = ImageDataset(train_indices, dataset_config['im_path'], data_indices_json, 'vqvae')

    train_dataloader = DataLoader(train_dataset, batch_size=train_config['ldm_batch_size'], shuffle=True)
    
    # Instantiate the model
    model = Unet(im_channels=autoencoder_model_config['z_channels'],
                 model_config=diffusion_model_config).to(device)
    model.train()
    
    # Load VQVAE
    print('Loading vqvae model')
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
        sys.exit('No vae checkpoint found')
    
    # Specify training parameters
    num_epochs = train_config['ldm_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['ldm_lr'])
    criterion = torch.nn.MSELoss()
    
    # Freeze vae parameters
    for param in vae.parameters():
        param.requires_grad = False
    
    # Run training
    for epoch_idx in range(num_epochs):
        losses = []
        for data in tqdm(train_dataloader):
            cond_input = None
            if condition_config is not None:
                im, edge, map = data
                edge = edge
                map = map
                cond_input = torch.cat([edge, map], dim=1)
                cond_input = cond_input.float().to(device)
            else:
                im = data
            optimizer.zero_grad()
            im = im.float().to(device)

            with torch.no_grad():
                im, _ = vae.encode(im)
                    
            ########### Handling Conditional Input ###########
            if 'image' in condition_types:
                assert 'image' in cond_input, 'Conditioning Type Image but no image conditioning input present'
                validate_image_config(condition_config)
                cond_input_image = cond_input['image'].to(device)
                # Drop condition
                im_drop_prob = get_config_value(condition_config['image_condition_config'],
                                                      'cond_drop_prob', 0.)
                cond_input['image'] = drop_image_condition(cond_input_image, im, im_drop_prob)
            ################################################
            
            # Sample random noise
            noise = torch.randn_like(im).to(device)
            
            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            
            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t, cond_input=cond_input)
            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses)))
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['ldm_ckpt_name']))
    
    print('Done Training ...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/celebhq_text_cond_clip.yaml', type=str)
    args = parser.parse_args()
    train(args)
