import json
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import os
from typing import List, Tuple, Dict, Optional, Union
import numpy as np

def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
        return None

def split_subject_data(json_data: Dict, num_test_subjects: int, random_seed: int = 42) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Splits subjects into training and testing sets and creates indexed samples for each.
    Subject '361' is always included in the training set if present.
    Total number of subjects must be exactly 360 (excluding '361' if present).
    
    Args:
        json_data (Dict): The JSON data containing subject information
        num_test_subjects (int): Number of subjects to use for testing
        random_seed (int): Seed for random number generator for reproducibility
        
    Returns:
        Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]: Two lists of (subject_id, sample_index) tuples
            for training and testing sets respectively
            
    Raises:
        ValueError: If total number of subjects (excluding '361') is not exactly 360
                   or if num_test_subjects is invalid
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Get all subject IDs
    all_subjects = sorted(list(json_data.keys()))
    
    # Handle subject '361' specially
    has_361 = '361' in all_subjects
    if has_361:
        print("Subject '361' found. Excluding from test set.")
        all_subjects.remove('361')
    
    # Check total subjects is exactly 360
    total_subjects = len(all_subjects)
    if total_subjects != 360:
        raise ValueError(f"Total number of subjects (excluding '361') must be exactly 360, but got {total_subjects}")
    
    if num_test_subjects >= total_subjects:
        raise ValueError(f"Number of test subjects ({num_test_subjects}) must be less than total subjects (360)")
    
    # Create a random permutation of subject indices
    permuted_indices = np.random.permutation(total_subjects)
    
    # Split subjects into training and testing
    test_indices = permuted_indices[:num_test_subjects]
    train_indices = permuted_indices[num_test_subjects:]
    
    # Get subject IDs for each set
    test_subjects = [all_subjects[i] for i in test_indices]
    train_subjects = [all_subjects[i] for i in train_indices]
    
    # Add subject '361' to training set if it exists
    if has_361:
        train_subjects.append('361')
    
    # Create training set tuples
    train_tuples = []
    for subject_id in train_subjects:
        num_samples = json_data[subject_id]['num_samples']
        for sample_idx in range(num_samples):
            train_tuples.append((subject_id, sample_idx))
    
    # Create testing set tuples
    test_tuples = []
    for subject_id in test_subjects:
        num_samples = json_data[subject_id]['num_samples']
        for sample_idx in range(num_samples):
            test_tuples.append((subject_id, sample_idx))
    
    return train_tuples, test_tuples


def get_phantom_data(json_data: Dict) -> List[Tuple[str, int]]:
    """
    Extracts all samples from subject '361' (phantom data).
    
    Args:
        json_data (Dict): The JSON data containing subject information
        
    Returns:
        List[Tuple[str, int]]: List of (subject_id, sample_index) tuples for subject '361'
            
    Raises:
        KeyError: If subject '361' is not present in the data
    """
    if '361' not in json_data:
        raise KeyError("Subject '361' not found in the data")
    
    phantom_tuples = []
    num_samples = json_data['361']['num_samples']
    for sample_idx in range(num_samples):
        phantom_tuples.append(('361', sample_idx))
    
    return phantom_tuples


class ImageDataset(Dataset):
    """A dataset class for loading and managing image data from HDF5 files.
    
    Attributes:
        data_tuples: List of (subject_id, sample_idx) tuples
        data_dir: Directory containing the HDF5 files
        json_data: Dictionary containing metadata for each subject
        stage: Processing stage ('vqvae' or 'diffusion')
    """
    
    VALID_STAGES = {'vqvae', 'diffusion'}
    
    def __init__(
        self,
        data_tuples: List[Tuple[str, int]],
        data_dir: str,
        json_data: Dict,
        stage: str = 'vqvae'
    ) -> None:
        
        if not os.path.isdir(data_dir):
            raise ValueError(f"Directory not found: {data_dir}")
        
        if stage not in self.VALID_STAGES:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {self.VALID_STAGES}")
            
        self.data_tuples = data_tuples
        self.data_dir = data_dir
        self.json_data = json_data
        self.stage = stage
    
    def _get_file_path(self, subject_id: str) -> str:
        """Gets the full file path for a subject."""
        if subject_id not in self.json_data:
            raise KeyError(f"Subject ID not found in json_data: {subject_id}")
        return os.path.join(self.data_dir, self.json_data[subject_id]['file_name'])
    
    def __len__(self) -> int:
        return len(self.data_tuples)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        subject_id, sample_idx = self.data_tuples[idx]
        
        try:
            file_path = self._get_file_path(subject_id)
            with h5py.File(file_path, 'r') as h5_file:

                if self.stage == 'vqvae':
                    image = torch.from_numpy(h5_file['images'][sample_idx]).unsqueeze(0)
                    return image
                
                else:  # diffusion stage
                    image = torch.from_numpy(h5_file['images'][sample_idx]).unsqueeze(0)
                    edge = torch.from_numpy(h5_file['edges'][sample_idx])
                    semantic_map = h5_file['semantic_maps'][sample_idx]
                    semantic_map_one_hot = np.eye(4)[semantic_map]
                    semantic_map_one_hot = torch.from_numpy(semantic_map_one_hot).permute(2, 0, 1)
                    return image, edge, semantic_map_one_hot
                    
        except Exception as e:
            print(f"Error details for subject {subject_id}, sample {sample_idx}: {str(e)}")
            raise RuntimeError(f"Error loading data for subject {subject_id}, sample {sample_idx}") from e
        