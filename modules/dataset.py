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
    
    Args:
        json_data (Dict): The JSON data containing subject information
        num_test_subjects (int): Number of subjects to use for testing
        random_seed (int): Seed for random number generator for reproducibility
        
    Returns:
        Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]: Two lists of (subject_id, sample_index) tuples
            for training and testing sets respectively
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Get all subject IDs
    all_subjects = sorted(list(json_data.keys()))
    total_subjects = len(all_subjects)
    
    if num_test_subjects >= total_subjects:
        raise ValueError(f"Number of test subjects ({num_test_subjects}) must be less than total subjects ({total_subjects})")
    
    # Create a deterministic random permutation of subject indices
    permuted_indices = np.random.permutation(total_subjects)
    
    # Split subjects into training and testing
    test_indices = permuted_indices[:num_test_subjects]
    train_indices = permuted_indices[num_test_subjects:]
    
    # Get subject IDs for each set
    test_subjects = [all_subjects[i] for i in test_indices]
    train_subjects = [all_subjects[i] for i in train_indices]
    
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
                    semantic_map = torch.from_numpy(h5_file['semantic_maps'][sample_idx])
                    return image, edge, semantic_map
                    
        except Exception as e:
            print(f"Error details for subject {subject_id}, sample {sample_idx}: {str(e)}")
            raise RuntimeError(f"Error loading data for subject {subject_id}, sample {sample_idx}") from e


# class ImageDataset(Dataset):
#     """A dataset class for loading and managing image data from HDF5 files.
    
#     Attributes:
#         data_tuples: List of (subject_id, sample_idx) tuples
#         data_dir: Directory containing the HDF5 files
#         json_data: Dictionary containing metadata for each subject
#         preload_subjects: Whether to preload all files into memory
#         stage: Processing stage ('vqvae' or 'diffusion')
#     """
    
#     VALID_STAGES = {'vqvae', 'diffusion'}
    
#     def __init__(
#         self,
#         data_tuples: List[Tuple[str, int]],
#         data_dir: str,
#         json_data: Dict,
#         preload_subjects: bool = False,
#         stage: str = 'vqvae'
#     ) -> None:
        
#         if not os.path.isdir(data_dir):
#             raise ValueError(f"Directory not found: {data_dir}")
        
#         if stage not in self.VALID_STAGES:
#             raise ValueError(f"Invalid stage: {stage}. Must be one of {self.VALID_STAGES}")
            
#         self.data_tuples = data_tuples
#         self.data_dir = data_dir
#         self.json_data = json_data
#         self.preload_subjects = preload_subjects
#         self.stage = stage
        
#         # Initialize file handlers dictionary if preloading
#         self.file_handlers: Dict[str, h5py.File] = {}
#         if preload_subjects:
#             self._preload_files()
    
#     def _preload_files(self) -> None:
#         """Preloads all unique subject files into memory."""
#         unique_subjects = {t[0] for t in self.data_tuples}
#         for subject_id in unique_subjects:
#             try:
#                 file_path = self._get_file_path(subject_id)
#                 self.file_handlers[subject_id] = h5py.File(file_path, 'r')
#             except Exception as e:
#                 self.close_files()  # Clean up on error
#                 raise RuntimeError(f"Failed to preload file for subject {subject_id}") from e
    
#     def _get_file_path(self, subject_id: str) -> str:
#         """Gets the full file path for a subject."""
#         if subject_id not in self.json_data:
#             raise KeyError(f"Subject ID not found in json_data: {subject_id}")
#         return os.path.join(self.data_dir, self.json_data[subject_id]['file_name'])
    
#     def _file_handler(self, subject_id: str) -> h5py.File:
#         """Context manager for handling HDF5 files."""
#         if self.preload_subjects:
#             return self.file_handlers[subject_id]
#         else:
#             file_path = self._get_file_path(subject_id)
#             with h5py.File(file_path, 'r') as f:
#                 return f
    
#     def __len__(self) -> int:
#         return len(self.data_tuples)
    
#     def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
#         subject_id, sample_idx = self.data_tuples[idx]
        
#         try:
#             with self._file_handler(subject_id) as h5_file:
#                 if self.stage == 'vqvae':
#                     image = torch.from_numpy(h5_file['images'][sample_idx]).unsqueeze(0) 
#                     # print(image.size())
#                     return image
                
#                 else:  # diffusion stage
#                     image = torch.from_numpy(h5_file['images'][sample_idx]).unsqueeze(0)
#                     edge = torch.from_numpy(h5_file['edges'][sample_idx])
#                     semantic_map = torch.from_numpy(h5_file['semantic_maps'][sample_idx])
#                     return image, edge, semantic_map
                    
#         except Exception as e:
#             raise RuntimeError(f"Error loading data for subject {subject_id}, sample {sample_idx}") from e
    
#     def close_files(self) -> None:
#         """Closes all open file handlers."""
#         for handler in self.file_handlers.values():
#             handler.close()
#         self.file_handlers.clear()
    
#     def __del__(self) -> None:
#         """Ensures proper cleanup when the dataset object is deleted."""
#         self.close_files()