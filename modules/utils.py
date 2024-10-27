import torch


def validate_image_conditional_input(cond_input, x):
    assert 'image' in cond_input, \
        "Model initialized with image conditioning but cond_input has no image information"
    assert cond_input['image'].shape[0] == x.shape[0], \
        "Batch size mismatch of image condition and input"
    assert cond_input['image'].shape[2] % x.shape[2] == 0, \
        "Height/Width of image condition must be divisible by latent input"
    

def get_config_value(config, key, default_value):
    return config[key] if key in config else default_value


def drop_image_condition(image_condition, im, im_drop_prob):
    if im_drop_prob > 0:
        im_drop_mask = torch.zeros((im.shape[0], 1, 1, 1), device=im.device).float().uniform_(0,
                                                                                        1) > im_drop_prob
        return image_condition * im_drop_mask
    else:
        return image_condition
    
    
def validate_image_config(condition_config):
    assert 'image_condition_config' in condition_config, \
        "Image conditioning desired but image condition config missing"
    assert 'image_condition_input_channels' in condition_config['image_condition_config'], \
        "image_condition_input_channels missing in image condition config"
    assert 'image_condition_output_channels' in condition_config['image_condition_config'], \
        "image_condition_output_channels missing in image condition config"