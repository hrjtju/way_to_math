#!/usr/bin/env python3
"""
MSD Lung数据集批量转换为HDF5格式
支持断点续传、多进程并行、压缩优化
"""

import os
import h5py
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import json
import argparse
import logging
from multiprocessing import Pool, cpu_count
from functools import partial

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_single_file(args: Tuple[str, str, Dict]) -> bool:
    """
    转换单个文件（支持多进程）
    args: (input_path, output_path, compression_config)
    """
    input_path, output_path, config = args
    try:
        # 检查是否已存在（断点续传）
        if os.path.exists(output_path) and not config['overwrite']:
            return True
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 加载nii.gz
        nib_img = nib.load(input_path)
        data = nib_img.get_fdata()
        
        # 根据数据类型优化存储
        if 'label' in input_path:
            # 标签：使用int64，压缩率更高
            data = data.astype(np.int64)
            compression_opts = config['label_compression']
        else:
            # 图像：使用float32，保持精度
            data = data.astype(np.float32)
            compression_opts = config['image_compression']
        
        # 写入HDF5
        with h5py.File(output_path, 'w') as f:
            dataset = f.create_dataset(
                'data',  # 统一键名为'data'
                data=data,
                compression='gzip',
                compression_opts=compression_opts,
                chunks=True,  # 关键：支持切片读取
                shuffle=True,  # 提高压缩率
            )
            
            # 保存元数据
            dataset.attrs['affine'] = nib_img.affine.tolist()
            dataset.attrs['original_shape'] = data.shape
            dataset.attrs['original_dtype'] = str(data.dtype)
            dataset.attrs['nii_header'] = str(nib_img.header)
        
        return True
        
    except Exception as e:
        logger.error(f"转换失败 {input_path}: {str(e)}")
        # 清理不完整文件
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def build_conversion_list(base_path: str, json_path: str, output_dir: str) -> List[Tuple[str, str, Dict]]:
    """
    构建转换任务列表
    返回: [(input_path, output_path, config), ...]
    """
    with open(json_path, 'r') as f:
        json_info = json.load(f)
    
    # 压缩配置：平衡速度和压缩率
    compression_config = {
        'image_compression': 4,  # 0-9，推荐4（平衡）
        'label_compression': 9,  # 标签可以压缩到最高
        'overwrite': False,
    }
    
    tasks = []
    
    # 处理所有split
    for split in ['train', "val"]:
        if split not in json_info:
            continue
        
        print(split)
        
        logger.info(f"扫描 {split} 集...")
        for item in json_info[split]:
            
            print(item)
            
            # 转换图像
            img_input = os.path.join(base_path, item['image'])
            img_output = os.path.join(output_dir, item['image'].replace('.nii.gz', '.h5'))
            tasks.append((img_input, img_output, compression_config))
            
            # 转换标签（如果有）
            if 'label' in item:
                label_input = os.path.join(base_path, item['label'])
                label_output = os.path.join(output_dir, item['label'].replace('.nii.gz', '.h5'))
                tasks.append((label_input, label_output, compression_config))
    
    # test
    split = "test"
    logger.info(f"扫描 {split} 集...")
    for item in json_info[split]:
        
        print(item)
        
        # 转换图像
        img_input = os.path.join(base_path, item)
        img_output = os.path.join(output_dir, item.replace('.nii.gz', '.h5'))
        tasks.append((img_input, img_output, compression_config))
        
    return tasks


def verify_conversion(output_dir: str, json_path: str) -> Dict[str, List[str]]:
    """
    验证转换完整性
    返回缺失文件列表
    """
    with open(json_path, 'r') as f:
        json_info = json.load(f)
    
    missing_files = {'images': [], 'labels': []}
    
    for split in ['training', 'val']:
        if split not in json_info:
            continue
        
        for item in json_info[split]:
            # 检查图像
            img_output = os.path.join(output_dir, item['image'].replace('.nii.gz', '.h5'))
            if not os.path.exists(img_output):
                missing_files['images'].append(item['image'])
            
            # 检查标签
            if 'label' in item:
                label_output = os.path.join(output_dir, item['label'].replace('.nii.gz', '.h5'))
                if not os.path.exists(label_output):
                    missing_files['labels'].append(item['label'])
    
    split = "test"
    for item in json_info[split]:
        # 检查图像
        img_output = os.path.join(output_dir, item.replace('.nii.gz', '.h5'))
        if not os.path.exists(img_output):
            missing_files['images'].append(item)
        
    
    return missing_files


def main():
    parser = argparse.ArgumentParser(description='批量转换MSD Lung数据集到HDF5')
    parser.add_argument('--base_path', type=str, required=True, help='MSD数据集根目录')
    parser.add_argument('--output_dir', type=str, required=True, help='HDF5输出目录')
    parser.add_argument('--json_path', type=str, default=None, help='dataset.json路径（默认在base_path下）')
    parser.add_argument('--num_workers', type=int, default=0, help='并行进程数（0=自动）')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已存在文件')
    
    args = parser.parse_args()
    
    # 自动查找json文件
    if args.json_path is None:
        args.json_path = os.path.join(args.base_path, 'dataset.json')
    
    assert os.path.exists(args.json_path), f"找不到dataset.json: {args.json_path}"
    
    # 构建任务列表
    logger.info("正在扫描转换任务...")
    tasks = build_conversion_list(args.base_path, args.json_path, args.output_dir)
    
    if not tasks:
        logger.warning("没有找到任何可转换的文件！")
        return
    
    # 估计存储空间
    total_input_size = sum(os.path.getsize(t[0]) for t in tasks if os.path.exists(t[0]))
    estimated_output_size = total_input_size * 0.7  # HDF5压缩后约70%大小
    logger.info(f"找到 {len(tasks)} 个文件")
    logger.info(f"原始数据大小: {total_input_size / 1e9:.2f} GB")
    logger.info(f"预估HDF5大小: {estimated_output_size / 1e9:.2f} GB")
    
    # 配置并行
    if args.num_workers == 0:
        args.num_workers = min(cpu_count(), 8)  # 最多8进程，避免IO争抢
    
    # 执行转换
    logger.info(f"启动 {args.num_workers} 个进程...")
    with Pool(args.num_workers) as pool:
        results = list(tqdm(
            pool.imap(convert_single_file, tasks),
            total=len(tasks),
            desc="转换进度"
        ))
    
    # 统计结果
    success_count = sum(results)
    logger.info(f"成功: {success_count}/{len(tasks)}")
    
    # 验证完整性
    logger.info("验证转换完整性...")
    missing = verify_conversion(args.output_dir, args.json_path)
    
    if missing['images'] or missing['labels']:
        logger.warning("缺失文件:")
        if missing['images']:
            logger.warning(f"  图像: {len(missing['images'])} 个")
        if missing['labels']:
            logger.warning(f"  标签: {len(missing['labels'])} 个")
    else:
        logger.info("✅ 所有文件转换成功！")


if __name__ == '__main__':
    main()