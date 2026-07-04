from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModel
from torch.utils.data import Dataset, ConcatDataset
import torch
import torch.nn as nn
import json
import os
import random
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from collections import defaultdict
from tqdm import tqdm

# ===================== 随机种子固定 =====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 标签定义 =====================
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

id2label = {
    0: "No Hallucination (0)",
    1: "Data Type Mismatch (1.1)",
    2: "Data Structure Misinterpretation (1.2)",
    3: "Global Logic Misalignment (2.1)",
    4: "Context Inconsistency (2.2)",
    5: "Local Logic Misalignment (2.3)",
    6: "Incomplete Implementation (2.4)",
    7: "Code Duplication (3.1)",
    8: "Dead Code (3.2)",
    9: "API Usage Errors (4.1)",
    10: "External Dependency Errors (4.2)",
    11: "Robustness Defects (5.1)",
    12: "Security Vulnerabilities (5.2)",
    13: "Resource Overuse (6.1)",
    14: "Syntax Overuse (7.1)",
    15: "Non-Code Content (7.2)",
}

# 完整标签名称（用于输出JSON）
full_label_names = {
    0: "No Hallucination",
    1: "1.1 Data Type Mismatch (DTM)",
    2: "1.2 Data Structure Misinterpretation (DSM)",
    3: "2.1 Global Logic Misalignment (GLM)",
    4: "2.2 Context Inconsistency (CI)",
    5: "2.3 Local Logic Misalignment (LoLM)",
    6: "2.4 Incomplete Implementation (II)",
    7: "3.1 Code Duplication (CD)",
    8: "3.2 Dead Code (DC)",
    9: "4.1 API Knowledge Errors (AKE)",
    10: "4.2 External Dependency Errors",
    11: "5.1 Robustness and Security Issues (RSI)",
    12: "5.2 Security Vulnerabilities",
    13: "6.1 Resource Overuse (RO)",
    14: "7.1 Syntax Errors (SE)",
    15: "7.2 Non-Code Content (NCC)",
}

# ===================== 自定义模型 =====================
class MultiLabelClassificationModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        return {"loss": loss, "logits": logits}

# ===================== 数据集划分与Dataset类 =====================
def split_with_minority_support(all_samples, test_size=0.5, seed=42):
    """带少数类支持的分割"""
    label_to_indices = defaultdict(list)
    for idx, (_, label_vec) in enumerate(all_samples):
        for label_id, val in enumerate(label_vec):
            if val == 1:
                label_to_indices[label_id].append(idx)

    used_indices = set()
    train_indices = set()

    for label_id, indices in label_to_indices.items():
        if not indices:
            continue
        np.random.seed(seed)
        chosen = np.random.choice(indices, max(1, len(indices) // 2), replace=False)
        train_indices.update(chosen)
        used_indices.update(indices)

    remaining = list(set(range(len(all_samples))) - train_indices)
    np.random.seed(seed)
    np.random.shuffle(remaining)
    test_count = int(len(all_samples) * test_size)
    test_indices = set(remaining[:test_count])
    train_indices.update(remaining[test_count:])

    return [all_samples[i] for i in train_indices], [all_samples[i] for i in test_indices]


class CodeHallucinationDataset(Dataset):
    def __init__(self, data_paths, tokenizer, max_length=512, split="train", test_size=0.5, seed=42):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.raw_samples = []      # 存储原始数据（用于输出JSON）
        all_samples = []

        print(f"🔍 读取数据文件：{data_paths}")
        for data_path in data_paths:
            if not os.path.exists(data_path):
                print(f"⚠️ 文件不存在: {data_path}")
                continue

            try:
                if data_path.endswith('.jsonl'):
                    with open(data_path, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f):
                            sample = json.loads(line.strip())
                            self._process_sample(sample, i + 1, all_samples, data_path)
                elif data_path.endswith('.json'):
                    with open(data_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            for key, sample in data.items():
                                self._process_sample(sample, key, all_samples, data_path)
                        elif isinstance(data, list):
                            for i, sample in enumerate(data):
                                self._process_sample(sample, i + 1, all_samples, data_path)
            except Exception as e:
                print(f"⚠️ 加载失败 {data_path}: {str(e)}")
                continue

        print(f"📦 总样本数：{len(all_samples)}")
        if not all_samples:
            raise ValueError("无有效样本")

        if split == "test":
            self.samples = all_samples
        else:
            train_samples, val_samples = split_with_minority_support(all_samples, test_size, seed)
            self.samples = train_samples if split == "train" else val_samples
            self._filter_raw_samples(split, train_samples, val_samples)

        print(f"✅ {split}样本数：{len(self.samples)} | 原始样本数：{len(self.raw_samples)}")

    def _process_sample(self, sample, line_id, all_samples, data_path):
        code_field = None
        if 'src' in sample:
            code_field = 'src'
        elif 'generation_code' in sample:
            code_field = 'generation_code'
        else:
            print(f"⚠️ [{data_path}:{line_id}] 缺少代码字段 (src/generation_code)")
            return

        required_keys = ['question', code_field, 'hallucination']
        if not all(k in sample for k in required_keys):
            print(f"⚠️ 缺失字段 [{data_path}:{line_id}]")
            return

        input_text = f"{sample['question'].strip()} </s> {sample[code_field].strip()}"
        test_case = sample.get('test_case', None)

        hallucinations = sample['hallucination']
        if not hallucinations or not isinstance(hallucinations, dict):
            valid_labels = [label2id["No Hallucination"]]
        else:
            valid_labels = [label2id[k] for k in hallucinations.keys() if k in label2id]

        if valid_labels:
            multi_hot = [0] * len(label2id)
            for label in valid_labels:
                multi_hot[label] = 1
            all_samples.append((input_text, multi_hot))

            # 保存原始样本
            enhanced_fields = {}
            for key in ["canonical_solution", "flaw_line", "flaw_line_index", "labeling_comments", "bm25_similarity_score"]:
                if key in sample:
                    enhanced_fields[key] = sample[key]

            self.raw_samples.append({
                "question": sample['question'],
                "original_code": sample[code_field],
                "raw_hallucination": {full_label_names[idx]: "" for idx in valid_labels},
                "label_ids": valid_labels,
                "input_text": input_text,
                "test_case": test_case,
                **enhanced_fields
            })

    def _filter_raw_samples(self, split, train_samples, val_samples):
        if split == "train":
            current_input_texts = [sample[0] for sample in train_samples]
        elif split == "val":
            current_input_texts = [sample[0] for sample in val_samples]
        else:
            current_input_texts = [sample[0] for sample in self.samples]

        self.raw_samples = [
            rs for rs in self.raw_samples
            if rs["input_text"] in current_input_texts
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_text, label_vec = self.samples[idx]
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_vec, dtype=torch.float),
            "raw_sample": self.raw_samples[idx] if idx < len(self.raw_samples) else None
        }

# ===================== 评估与阈值计算函数 =====================
def calculate_adaptive_thresholds(model, val_dataset):
    """计算每个类别的自适应阈值"""
    model.eval()
    probs_all = []
    for i in range(len(val_dataset)):
        encoding = val_dataset[i]
        with torch.no_grad():
            input_ids = encoding["input_ids"].unsqueeze(0).to(device)
            attention_mask = encoding["attention_mask"].unsqueeze(0).to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]                     # 字典访问
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            probs_all.append(probs)

    probs_all = np.array(probs_all)
    thresholds = {}
    for label_idx in range(len(label2id)):
        label_name = id2label[label_idx]
        mean_prob = np.mean(probs_all[:, label_idx])
        thresholds[label_name] = float(mean_prob)
        print(f"{label_name}: 平均预测概率={mean_prob:.4f}, 自适应阈值={mean_prob:.4f}")
    return thresholds


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits))
    preds = (probs >= 0.5).int()
    labels = torch.tensor(labels).int()
    return {
        "accuracy": accuracy_score(labels, preds),
        "micro_f1": f1_score(labels, preds, average="micro"),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def predict_and_evaluate_with_adaptive_thresholds(model, full_dataset, adaptive_thresholds):
    """在完整数据集上评估模型"""
    model.eval()
    probs_all = []
    labels_all = []
    for i in range(len(full_dataset)):
        encoding = full_dataset[i]
        with torch.no_grad():
            input_ids = encoding["input_ids"].unsqueeze(0).to(device)
            attention_mask = encoding["attention_mask"].unsqueeze(0).to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            label = encoding["labels"].cpu().numpy().flatten()
            probs_all.append(probs)
            labels_all.append(label)

    probs_all = np.array(probs_all)
    labels_all = np.array(labels_all)

    preds = []
    for sample_probs in probs_all:
        sample_pred = []
        for idx in range(len(label2id)):
            label_name = id2label[idx]
            sample_pred.append(1 if sample_probs[idx] >= adaptive_thresholds[label_name] else 0)
        preds.append(sample_pred)
    preds = np.array(preds)

    print("\n====== 自适应阈值分类报告 ======")
    print(classification_report(
        labels_all,
        preds,
        target_names=[id2label[i] for i in range(len(label2id))],
        zero_division=0
    ))
    accuracy = accuracy_score(labels_all, preds)
    micro_f1 = f1_score(labels_all, preds, average="micro")
    macro_f1 = f1_score(labels_all, preds, average="macro")
    print(f"accuracy: {accuracy:.4f} | micro_f1: {micro_f1:.4f} | macro_f1: {macro_f1:.4f}")
    return preds, probs_all


def multilabel_classify_test_set(test_data_path, model, tokenizer, adaptive_thresholds, output_dir):
    """对测试集进行多标签预测，并保存结果JSON"""
    test_dataset = CodeHallucinationDataset(
        data_paths=[test_data_path],
        tokenizer=tokenizer,
        split="test",
        test_size=0.0
    )

    print("🔍 测试集多标签预测...")
    classified_results = []
    model.eval()
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            batch = test_dataset[idx]
            input_ids = batch["input_ids"].unsqueeze(0).to(device)
            attention_mask = batch["attention_mask"].unsqueeze(0).to(device)
            raw_sample = batch["raw_sample"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs["logits"]).cpu().numpy()[0]

            predicted_label_ids = [
                i for i in range(len(label2id))
                if probs[i] >= adaptive_thresholds[id2label[i]]
            ]
            predicted_labels = [full_label_names[i] for i in predicted_label_ids]
            has_hallucination = 0 if (len(predicted_labels) == 1 and full_label_names[0] in predicted_labels) else 1
            if has_hallucination and full_label_names[0] in predicted_labels:
                predicted_labels.remove(full_label_names[0])

            classified_results.append({
                "sample_id": idx,
                "question": raw_sample["question"],
                "original_code": raw_sample["original_code"],
                "predicted_hallucinations": predicted_labels,
                "has_hallucination": has_hallucination,
                "raw_hallucination_info": raw_sample["raw_hallucination"],
                "round_0_code": raw_sample["original_code"],
                "test_case": raw_sample.get("test_case")
            })

    result_path = os.path.join(output_dir, "multilabel_classification_results.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(classified_results, f, ensure_ascii=False, indent=2)

    total = len(classified_results)
    hallucination_count = sum(1 for res in classified_results if res["has_hallucination"] == 1)
    print(f"📊 总样本{total} | 有幻觉{hallucination_count}({hallucination_count / total * 100:.2f}%)")
    print(f"结果保存至：{result_path}")
    return [res for res in classified_results if res["has_hallucination"] == 1], test_dataset


# ===================== 主函数 =====================
def main():
    config = {
        "classifier_train_data_paths": [

        ],
        "test_data_path": "",
        "base_model_path": "",
        "output_dir": "",
        "epochs": 7,
        "batch_size": 8,
        "seed": 42
    }

    os.makedirs(config["output_dir"], exist_ok=True)

    print("🚀 开始分类流程")
    print("=" * 60)

    # 1. 初始化Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["base_model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载训练/验证数据
    train_dataset = CodeHallucinationDataset(
        data_paths=config["classifier_train_data_paths"],
        tokenizer=tokenizer,
        split="train",
        seed=config["seed"]
    )
    val_dataset = CodeHallucinationDataset(
        data_paths=config["classifier_train_data_paths"],
        tokenizer=tokenizer,
        split="val",
        seed=config["seed"]
    )

    # 3. 初始化自定义模型
    model = MultiLabelClassificationModel(
        model_name=config["base_model_path"],
        num_labels=len(label2id)
    ).to(device)

    # 4. 配置训练参数
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        eval_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        logging_dir=os.path.join(config["output_dir"], "logs"),
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        seed=config["seed"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("\n====== 训练模型 ======")
    trainer.train()
    trainer.save_model(config["output_dir"])
    print(f"✅ 分类模型已保存至：{config['output_dir']}")

    # 5. 计算自适应阈值
    print("\n====== 计算自适应阈值 ======")
    adaptive_thresholds = calculate_adaptive_thresholds(model, val_dataset)

    # 保存阈值
    threshold_path = os.path.join(config["output_dir"], "adaptive_thresholds.json")
    with open(threshold_path, 'w', encoding='utf-8') as f:
        json.dump(adaptive_thresholds, f, ensure_ascii=False, indent=2)

    # 6. 在验证集+训练集上评估
    full_dataset = ConcatDataset([train_dataset, val_dataset])
    predict_and_evaluate_with_adaptive_thresholds(model, full_dataset, adaptive_thresholds)

    # 7. 对测试集进行预测并输出结果
    print("\n====== 测试集预测 ======")
    test_classified_samples, _ = multilabel_classify_test_set(
        test_data_path=config["test_data_path"],
        model=model,
        tokenizer=tokenizer,
        adaptive_thresholds=adaptive_thresholds,
        output_dir=config["output_dir"]
    )

    print(f"\n🎉 分类流程完成！结果位于 {config['output_dir']}")


if __name__ == "__main__":
    main()
