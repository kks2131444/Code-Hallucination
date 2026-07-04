import json
import os
from tqdm import tqdm

# --------------------------
# 1. 标签映射（与原始代码保持一致）
# --------------------------
label2id = {
    "No Hallucination": 0,
    "1.1": 1,
    "1.2": 2,
    "2.1": 3,
    "2.2": 4,
    "2.3": 5,
    "2.4": 6,
    "3.1": 7,
    "3.2": 8,
    "4.1": 9,
    "4.2": 10,
    "5.1": 11,
    "5.2": 12,
    "6.1": 13,
    "7.1": 14,
    "7.2": 15,
}

# 标签名称到文件名的映射（空格替换为下划线，扩展名改为 .json）
label_to_filename = {
    label: f"{label.replace(' ', '_')}.json"   # 修改为 .json
    for label in label2id.keys()
}


# --------------------------
# 2. 读取数据集（支持json和jsonl格式）
# --------------------------
def load_dataset(file_path):
    """读取json或jsonl格式的数据集"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据集文件不存在：{file_path}")

    samples = []
    try:
        if file_path.endswith('.jsonl'):
            # 读取jsonl格式（每行一个样本）
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                        samples.append(sample)
                    except json.JSONDecodeError:
                        print(f"⚠️ 跳过第{line_num}行：JSON格式错误")
        else:
            # 读取json格式（样本列表）
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data
                elif isinstance(data, dict):
                    # 如果是字典，尝试取其值列表（常见于将数组存在某个键下）
                    if "data" in data and isinstance(data["data"], list):
                        samples = data["data"]
                    else:
                        samples = list(data.values())
                else:
                    raise ValueError("JSON文件格式应为列表或包含列表的字典")
        print(f"✅ 成功加载数据集，共 {len(samples)} 个样本")
        return samples
    except Exception as e:
        raise RuntimeError(f"读取数据集失败：{str(e)}")


# --------------------------
# 3. 按hallucination字段分类并保存（输出JSON数组，每个样本index从0开始）
# --------------------------
def classify_and_save(samples, output_dir):
    """按样本的hallucination标签分类，每个文件输出为一个JSON数组，样本index从0开始重新编号"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 存储每个标签的样本列表（内存中）
    label_samples = {label: [] for label in label2id.keys()}
    # 记录每个标签当前应分配的下一个 index
    label_index_counter = {label: 0 for label in label2id.keys()}
    label_count = {label: 0 for label in label2id.keys()}

    # 遍历所有样本
    for sample_idx, sample in enumerate(tqdm(samples, desc="分类样本"), 1):
        original_id = sample.get("index", f"加载顺序_{sample_idx}")

        # 检查 hallucination 字段
        if "hallucination" not in sample:
            print(f"⚠️ 样本原始index={original_id}：缺少hallucination字段，跳过")
            continue

        hallucination = sample["hallucination"]
        if not isinstance(hallucination, dict):
            print(f"⚠️ 样本原始index={original_id}的hallucination格式错误（需为字典），归为No Hallucination")
            target_labels = ["No Hallucination"]
        else:
            target_labels = [k for k in hallucination.keys() if k in label2id]
            if not target_labels:
                target_labels = ["No Hallucination"]
                print(f"⚠️ 样本原始index={original_id}无有效标签，归为No Hallucination")

        # 对每个目标标签，复制样本并分配新的 index，添加到对应列表
        for label in target_labels:
            sample_copy = sample.copy()
            sample_copy["index"] = label_index_counter[label]
            label_samples[label].append(sample_copy)
            label_index_counter[label] += 1
            label_count[label] += 1

    # 所有样本处理完成后，写入 JSON 文件（每个标签一个文件）
    print("\n开始写入 JSON 文件...")
    for label in label2id.keys():
        file_path = os.path.join(output_dir, label_to_filename[label])
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(label_samples[label], f, ensure_ascii=False, indent=2)
            print(f"✅ 已保存: {file_path} ({len(label_samples[label])} 个样本)")
        except Exception as e:
            print(f"❌ 写入失败 {file_path}: {e}")

    # 打印统计结果
    print("\n====== 分类统计 ======")
    print(f"总处理样本数：{len(samples)}")
    print("各标签样本数及最终index范围：")
    for label in label2id.keys():
        count = label_count[label]
        if count == 0:
            print(f"  {label}: 0个样本")
        else:
            print(f"  {label}: {count}个样本（index范围：0 ~ {count-1}）")
    print(f"\n所有文件已保存至：{output_dir}")


# --------------------------
# 4. 主执行流程
# --------------------------
if __name__ == "__main__":
    # 数据集路径（去重后的文件）
    dataset_path = ""
    # 输出目录
    output_dir = ""

    try:
        samples = load_dataset(dataset_path)
        classify_and_save(samples, output_dir)
    except Exception as e:
        print(f"执行失败：{str(e)}")