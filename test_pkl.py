import pickle
import os
from pathlib import Path
import torch
import numpy as np

def test_single_pkl(pkl_path):
    """测试单个pkl文件"""
    print(f"\nTesting pkl file: {pkl_path}")
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print("Successfully loaded pkl file!")
        print(f"Data type: {type(data)}")
        
        # 如果是字典，打印键
        if isinstance(data, dict):
            print(f"Keys in dict: {data.keys()}")
            for k, v in data.items():
                print(f"\nKey: {k}")
                print(f"Value type: {type(v)}")
                if isinstance(v, (torch.Tensor, np.ndarray)):
                    print(f"Shape: {v.shape}")
                elif isinstance(v, list):
                    print(f"List length: {len(v)}")
                    if len(v) > 0:
                        print(f"First element type: {type(v[0])}")
        
        # 如果是元组或列表，打印长度和元素类型
        elif isinstance(data, (tuple, list)):
            print(f"Length: {len(data)}")
            for i, item in enumerate(data):
                print(f"\nItem {i}:")
                print(f"Type: {type(item)}")
                if isinstance(item, (torch.Tensor, np.ndarray)):
                    print(f"Shape: {item.shape}")
        
        return True
    
    except Exception as e:
        print(f"Error loading pkl file: {str(e)}")
        return False

def test_pkl_directory(directory):
    """测试目录下所有pkl文件"""
    directory = Path(directory)
    pkl_files = list(directory.glob("*.pkl"))
    
    print(f"Found {len(pkl_files)} pkl files in {directory}")
    
    success = 0
    failed = 0
    
    for pkl_file in pkl_files:
        if test_single_pkl(pkl_file):
            success += 1
        else:
            failed += 1
            
    print(f"\nSummary:")
    print(f"Successfully loaded: {success} files")
    print(f"Failed to load: {failed} files")

if __name__ == "__main__":
    # 测试单个文件
    single_file = "datasets/clip_proposals/neudet_props/clip_distilfeats/crazing_1.pkl"
    print("Testing single file...")
    test_single_pkl(single_file)
    
    # 测试整个目录
    # print("\nTesting entire directory...")
    # directory = "datasets/clip_proposals/neudet_props/clip_distilfeats"
    # test_pkl_directory(directory) 