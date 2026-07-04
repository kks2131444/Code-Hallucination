#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

# ========== 配置文件路径 ==========
INPUT_FILE = "/root/CodeBERT-master123/CodeBERT/newworld/label1_result.json"   # 修改为你的文件路径
OUTPUT_FILE = "/root/CodeBERT-master123/CodeBERT/last/deduplicated_by_src.json"
FIELD = "src"   # 去重依据的字段名
# =================================

def load_items(file_path, field):
    """从 JSON 文件中加载所有包含指定字段的数据条目"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"错误：无法读取文件 {file_path} - {e}")
        return []

    items = []
    if isinstance(data, list):
        # 根节点为列表，每个元素是一个条目
        for item in data:
            if isinstance(item, dict) and field in item:
                items.append(item)
    elif isinstance(data, dict):
        # 根节点为字典，尝试查找第一个值为列表的键（常见的数据列表）
        found_list = False
        for value in data.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and field in item:
                        items.append(item)
                found_list = True
                break
        if not found_list and field in data:
            # 整个字典本身就是一个条目
            items.append(data)
    else:
        print(f"警告：文件 {file_path} 的根类型不是列表或字典，无法处理")
    return items

def main():
    # 加载数据
    items = load_items(INPUT_FILE, FIELD)
    print(f"原始数据条数: {len(items)}")

    # 按 src 字段去重（保留第一次出现的条目）
    seen_src = set()
    deduplicated_items = []
    for item in items:
        src_value = item[FIELD]
        if src_value not in seen_src:
            seen_src.add(src_value)
            deduplicated_items.append(item)

    print(f"去重后数据条数: {len(deduplicated_items)} (每个 src 唯一)")

    # 保存到输出文件
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(deduplicated_items, f, ensure_ascii=False, indent=2)
        print(f"去重结果已保存至: {OUTPUT_FILE}")
    except Exception as e:
        print(f"写入文件失败: {e}")

if __name__ == "__main__":
    main()