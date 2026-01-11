#!/usr/bin/env python3
"""
适配您修改后的转换代码的验证工具
支持train/val/test三份数据，test集为字符串列表结构
"""

import os
import h5py
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HDF5ValidatorV2:
    """适配您转换逻辑的验证器"""
    
    def __init__(self, h5_base_path: str, original_json_path: str, original_base_path: str):
        self.h5_base_path = Path(h5_base_path)
        self.original_json_path = Path(original_json_path)
        self.original_base_path = Path(original_base_path)
        
        with open(self.original_json_path, 'r') as f:
            self.dataset_info = json.load(f)
    
    def get_expected_files(self) -> List[Tuple[str, str, str]]:
        """
        根据您的转换逻辑生成预期文件列表
        返回: [(split, file_type, relative_path), ...]
        """
        expected_files = []
        
        # 处理 train 和 val (item是字典)
        for split in ['train', 'val']:
            if split not in self.dataset_info:
                continue
            for item in self.dataset_info[split]:
                # 图像文件
                img_path = item['image'].replace('.nii.gz', '.h5')
                expected_files.append((split, 'image', img_path))
                
                # 标签文件
                if 'label' in item:
                    label_path = item['label'].replace('.nii.gz', '.h5')
                    expected_files.append((split, 'label', label_path))
        
        # 处理 test (item是字符串)
        split = 'test'
        if split in self.dataset_info:
            for item in self.dataset_info[split]:
                # item 直接是文件名
                img_path = item.replace('.nii.gz', '.h5')
                expected_files.append((split, 'image', img_path))
        
        return expected_files
    
    def level1_file_integrity(self) -> Dict[str, List[str]]:
        """一级验证：文件完整性和可读取性"""
        logger.info("=" * 60)
        logger.info("【一级验证】文件完整性检查")
        
        corrupted_files = {'train': {'image': [], 'label': []},
                          'val': {'image': [], 'label': []},
                          'test': {'image': []}}
        
        expected_files = self.get_expected_files()
        total = len(expected_files)
        success = 0
        
        for split, file_type, rel_path in expected_files:
            file_path = self.h5_base_path / rel_path
            if self._check_single_file(file_path, split, file_type):
                success += 1
            else:
                corrupted_files[split][file_type].append(str(rel_path))
        
        # 统计结果
        logger.info(f"检查文件总数: {total}")
        logger.info(f"成功: {success}")
        logger.info(f"失败: {total - success}")
        logger.info(f"成功率: {success/total*100:.2f}%")
        
        # 详细报告
        for split in ['train', 'val', 'test']:
            for ftype in corrupted_files[split]:
                if corrupted_files[split][ftype]:
                    logger.warning(f"{split}集 {ftype}缺失/损坏: {len(corrupted_files[split][ftype])}个")
        
        return corrupted_files
    
    def _check_single_file(self, file_path: Path, split: str, file_type: str) -> bool:
        """检查单个HDF5文件"""
        try:
            if not file_path.exists():
                logger.error(f"❌ 文件不存在: {file_path}")
                return False
            
            with h5py.File(file_path, 'r') as f:
                if 'data' not in f:
                    logger.error(f"❌ 缺少'data' dataset: {file_path}")
                    return False
                
                data = f['data']
                # 验证属性
                for attr in ['affine', 'original_shape', 'original_dtype']:
                    if attr not in data.attrs:
                        logger.error(f"❌ 缺少属性 {attr}: {file_path}")
                        return False
                
                # 验证数据可读取
                test_slice = tuple(slice(0, min(5, s)) for s in data.shape)
                _ = data[test_slice]
                
            logger.debug(f"✅ {split}/{file_type}: {file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 文件损坏 {file_path}: {str(e)}")
            return False
    
    def level2_data_consistency(self, num_samples: int = 20) -> Dict[str, List[str]]:
        """二级验证：数据一致性（仅抽样train集）"""
        logger.info("=" * 60)
        logger.info(f"【二级验证】数据一致性检查（抽样{num_samples}个）")
        
        mismatches = {'image': [], 'label': []}
        
        # 只验证train集（用户代码中主要转换逻辑）
        if 'train' not in self.dataset_info:
            logger.warning("train集不存在，跳过一致性验证")
            return mismatches
        
        # 随机抽样
        train_items = self.dataset_info['train']
        if not train_items:
            return mismatches
        
        sample_indices = np.random.choice(len(train_items), 
                                         size=min(num_samples, len(train_items)), 
                                         replace=False)
        
        for idx in sample_indices:
            item = train_items[idx]
            logger.info(f"\n验证样本: {item['image']}")
            
            # 验证图像
            original_img = self.original_base_path / item['image']
            h5_img = self.h5_base_path / item['image'].replace('.nii.gz', '.h5')
            if not self._compare_data(original_img, h5_img, is_label=False):
                mismatches['image'].append(item['image'])
            
            # 验证标签
            if 'label' in item:
                original_label = self.original_base_path / item['label']
                h5_label = self.h5_base_path / item['label'].replace('.nii.gz', '.h5')
                if not self._compare_data(original_label, h5_label, is_label=True):
                    mismatches['label'].append(item['label'])
        
        # 总结
        if not mismatches['image'] and not mismatches['label']:
            logger.info("✅ 所有抽样数据与原始数据一致")
        else:
            logger.error(f"❌ 不一致文件: 图像{len(mismatches['image'])}个, 标签{len(mismatches['label'])}个")
        
        return mismatches
    
    def _compare_data(self, original_path: Path, h5_path: Path, is_label: bool) -> bool:
        """对比原始nii.gz和HDF5数据"""
        try:
            # 加载原始数据
            nib_data = nib.load(original_path).get_fdata()
            if is_label:
                nib_data = nib_data.astype(np.int64)
            else:
                nib_data = nib_data.astype(np.float32)
            
            # 加载HDF5数据
            with h5py.File(h5_path, 'r') as f:
                h5_data = f['data'][:]
            
            # 验证形状
            if nib_data.shape != h5_data.shape:
                logger.error(f"  ❌ 形状不匹配: {original_path.name}")
                return False
            
            # 验证数值
            if is_label:
                if not np.array_equal(nib_data, h5_data):
                    logger.error(f"  ❌ 标签数值不匹配: {original_path.name}")
                    return False
            else:
                if not np.allclose(nib_data, h5_data, rtol=1e-5, atol=1e-6):
                    max_diff = np.max(np.abs(nib_data - h5_data))
                    logger.error(f"  ❌ 图像数值不匹配: {original_path.name}, 最大差异: {max_diff:.2e}")
                    return False
            
            # 附加信息
            if is_label:
                unique_vals = np.unique(h5_data)
                logger.info(f"  ✅ 标签值分布: {unique_vals}")
            else:
                ct_range = (h5_data.min(), h5_data.max())
                logger.info(f"  ✅ CT值范围: [{ct_range[0]:.1f}, {ct_range[1]:.1f}] HU")
            
            return True
            
        except Exception as e:
            logger.error(f"  ❌ 对比失败 {original_path.name}: {str(e)}")
            return False
    
    def level3_functional_test(self, patch_shape: Tuple[int, int, int] = (128, 128, 128)) -> bool:
        """三级验证：Dataloader功能测试"""
        logger.info("=" * 60)
        logger.info("【三级验证】Dataloader功能测试")
        
        try:
            # 假设您的Dataset类在当前目录或已安装
            sys.path.append(os.path.dirname(__file__))
            from HDF5MSDLungPatchDataset import HDF5MSDLungPatchDataset
            
            # 测试train模式
            logger.info("测试train模式...")
            train_dataset = HDF5MSDLungPatchDataset(
                h5_base_path=str(self.h5_base_path),
                json_path=str(self.original_json_path),
                train=True,
                patch_shape=patch_shape,
                stride_shape=patch_shape,
                cache_size=2
            )
            
            if len(train_dataset) == 0:
                logger.error("❌ train_dataset为空")
                return False
            
            logger.info(f"✅ train_dataset创建成功，共{len(train_dataset)}个patches")
            
            # 测试test模式
            logger.info("测试test模式...")
            test_dataset = HDF5MSDLungPatchDataset(
                h5_base_path=str(self.h5_base_path),
                json_path=str(self.original_json_path),
                train=False,
                patch_shape=patch_shape,
                stride_shape=patch_shape,
                cache_size=1
            )
            
            if len(test_dataset) == 0:
                logger.error("❌ test_dataset为空")
                return False
            
            logger.info(f"✅ test_dataset创建成功，共{len(test_dataset)}个patches")
            
            # 测试加载单个样本
            logger.info("测试patch加载...")
            for split_name, dataset in [('train', train_dataset), ('test', test_dataset)]:
                # 从开头、中间、结尾各取一个样本
                test_indices = [0, len(dataset)//2, len(dataset)-1]
                
                for idx in test_indices:
                    try:
                        if split_name == 'train':
                            img_patch, label_patch = dataset[idx]
                            # 验证形状
                            if img_patch.shape != patch_shape or label_patch.shape != patch_shape:
                                logger.error(f"❌ {split_name} patch形状错误")
                                return False
                        else:
                            img_patch = dataset[idx]
                            if img_patch.shape != patch_shape:
                                logger.error(f"❌ {split_name} patch形状错误")
                                return False
                        
                        logger.debug(f"  {split_name}[{idx}]加载成功")
                        
                    except Exception as e:
                        logger.error(f"❌ 加载{split_name}[{idx}]失败: {str(e)}")
                        return False
            
            logger.info("✅ 所有功能测试通过")
            return True
            
        except Exception as e:
            logger.error(f"❌ 功能测试失败: {str(e)}")
            return False
    
    def generate_report(self) -> str:
        """生成完整验证报告"""
        logger.info("=" * 60)
        logger.info("【生成验证报告】")
        
        report = []
        report.append("=" * 60)
        report.append("MSD Lung HDF5转换验证报告")
        report.append("=" * 60)
        
        # 执行三级验证
        corrupted = self.level1_file_integrity()
        mismatches = self.level2_data_consistency(num_samples=5)
        functional_ok = self.level3_functional_test()
        
        # 报告一级验证结果
        report.append("\n【一级验证】文件完整性")
        total_corrupted = sum(len(v) for split in corrupted.values() for v in split.values())
        if total_corrupted == 0:
            report.append("✅ 通过：所有文件完整且可读取")
        else:
            report.append(f"❌ 失败：共{total_corrupted}个文件缺失或损坏")
        
        # 报告二级验证结果
        report.append("\n【二级验证】数据一致性")
        if not mismatches['image'] and not mismatches['label']:
            report.append("✅ 通过：抽样数据与原始数据一致")
        else:
            report.append(f"❌ 失败：{len(mismatches['image'])}个图像, {len(mismatches['label'])}个标签不匹配")
        
        # 报告三级验证结果
        report.append("\n【三级验证】功能测试")
        if functional_ok:
            report.append("✅ 通过：Dataloader可正常工作")
        else:
            report.append("❌ 失败：Dataloader加载异常")
        
        # 综合结论
        report.append("\n" + "=" * 60)
        overall_pass = (total_corrupted == 0 and 
                       not mismatches['image'] and 
                       not mismatches['label'] and 
                       functional_ok)
        if overall_pass:
            report.append("🎉 综合结果：验证通过，HDF5数据完全可用")
        else:
            report.append("⚠️  综合结果：验证失败，请检查上述问题")
        report.append("=" * 60)
        
        report_str = "\n".join(report)
        logger.info(report_str)
        return report_str


def main():
    parser = argparse.ArgumentParser(description='验证HDF5转换结果（适配您的转换脚本）')
    parser.add_argument('--h5_base_path', type=str, required=True, help='HDF5根目录')
    parser.add_argument('--original_base_path', type=str, required=True, help='原始数据根目录')
    parser.add_argument('--json_path', type=str, default=None, help='dataset.json路径')
    parser.add_argument('--level', type=int, choices=[1, 2, 3], default=3, help='验证级别')
    
    args = parser.parse_args()
    
    if args.json_path is None:
        args.json_path = os.path.join(args.original_base_path, 'dataset.json')
    
    validator = HDF5ValidatorV2(args.h5_base_path, args.json_path, args.original_base_path)
    
    # 根据级别执行验证
    if args.level >= 1:
        validator.level1_file_integrity()
    
    if args.level >= 2:
        validator.level2_data_consistency(num_samples=5)
    
    if args.level >= 3:
        validator.level3_functional_test()
    
    # 生成完整报告
    report = validator.generate_report()
    
    # 保存报告
    report_path = os.path.join(args.h5_base_path, 'validation_report_v2.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"\n验证报告已保存: {report_path}")


if __name__ == '__main__':
    main()