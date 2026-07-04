#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import os

# ========== 配置 ==========
INPUT_FILES = [
    "/root/CodeBERT-master123/CodeBERT/last/deduplicated/1.1.json",
    "/root/CodeBERT-master123/CodeBERT/last/deduplicated/1.2.json",
    "/root/CodeBERT-master123/CodeBERT/last/deduplicated/2.1.json",
    "/root/CodeBERT-master123/CodeBERT/last/deduplicated/2.2.json",
    "/root/CodeBERT-master123/CodeBERT/last/deduplicated/2.3.json",
    "/root/CodeBERT-master123/CodeBERT/last/deduplicated/2.4.json",
    "/root/CodeBERT-master123/CodeBERT/last/deduplicated/3.1.json",
    "/root/CodeBERT-master123/CodeBERT/last/deduplicated/3.2.json",
    "/root/CodeBERT-master123/CodeBERT/last/deduplicated/4.1.json",
    "/root/CodeBERT-master123/CodeBERT/last/deduplicated/4.2.json",
    "/root/CodeBERT-master123/CodeBERT/last/deduplicated/5.1.json",
    "/root/CodeBERT-master123/CodeBERT/last/deduplicated/5.2.json",
    "/root/CodeBERT-master123/CodeBERT/last/deduplicated/6.1.json",
    "/root/CodeBERT-master123/CodeBERT/last/deduplicated/7.1.json",
    "/root/CodeBERT-master123/CodeBERT/last/deduplicated/7.2.json",
]
OUTPUT_TEST = "/root/CodeBERT-master123/CodeBERT/last/test_set.json"      # 测试集（20%）
OUTPUT_CASE = "/root/CodeBERT-master123/CodeBERT/last/case_set.json"      # 案例集（80%）
SAMPLE_RATIO = 0.2   # 测试集比例 20%
RANDOM_SEED = 42     # 固定随机种子，保证可复现（设为 None 则每次不同）
# =========================

def load_json(file_path):
    """加载 JSON 数组文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{file_path} 根节点不是列表")
        return data
    except Exception as e:
        print(f"错误：无法读取 {file_path} - {e}")
        return None

def save_json(file_path, data):
    """保存 JSON 数组文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"已保存: {file_path} ({len(data)} 个样本)")
    except Exception as e:
        print(f"保存失败 {file_path}: {e}")

def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
    
    test_samples = []
    case_samples = []
    total_original = 0
    total_test = 0
    total_case = 0
    
    print("开始处理各文件...")
    for file_path in INPUT_FILES:
        print(f"\n处理: {os.path.basename(file_path)}")
        samples = load_json(file_path)
        if samples is None:
            continue
        
        n = len(samples)
        if n == 0:
            print("  文件为空，跳过")
            continue
        
        # 计算抽取测试集的数量（至少 1 个，不超过总数）
        test_n = max(1, int(n * SAMPLE_RATIO)) if n > 0 else 0
        if test_n > n:
            test_n = n
        
        # 随机抽取测试集索引（无放回）
        test_indices = set(random.sample(range(n), test_n))
        test_part = [samples[i] for i in test_indices]
        case_part = [samples[i] for i in range(n) if i not in test_indices]
        
        test_samples.extend(test_part)
        case_samples.extend(case_part)
        
        print(f"  原始: {n} 条")
        print(f"  测试集: {len(test_part)} 条 ({len(test_part)/n*100:.1f}%)")
        print(f"  案例集: {len(case_part)} 条 ({len(case_part)/n*100:.1f}%)")
        
        total_original += n
        total_test += len(test_part)
        total_case += len(case_part)
    
    # 保存结果
    print("\n保存结果...")
    save_json(OUTPUT_TEST, test_samples)
    save_json(OUTPUT_CASE, case_samples)
    
    print("\n====== 汇总 ======")
    print(f"总原始样本数: {total_original}")
    print(f"测试集总样本数: {total_test} ({total_test/total_original*100:.1f}%)")
    print(f"案例集总样本数: {total_case} ({total_case/total_original*100:.1f}%)")
    print(f"测试集文件: {OUTPUT_TEST}")
    print(f"案例集文件: {OUTPUT_CASE}")

if __name__ == "__main__":
    main()