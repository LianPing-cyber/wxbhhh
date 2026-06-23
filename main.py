from dotenv import load_dotenv
load_dotenv()

from data_process_session import extract_trajectory_from_jsonl
from pathlib import Path
from check_pcap import predict_pcap_data
from check_session import predict_session_data

import json
import os, re
import csv
from tqdm import tqdm
import argparse
import itertools

def extract_data(dataset_path):
    base_path = Path(dataset_path)
    extracted_data = []

    if not base_path.exists() or not base_path.is_dir():
        print(f"[错误] 数据集路径不存在或不是文件夹: {dataset_path}")
        return extracted_data

    for sample_dir in base_path.iterdir():
        # 跳过可能存在的独立文件（如 results.csv），只处理子文件夹
        if not sample_dir.is_dir():
            continue

        sample_name = sample_dir.name
        
        jsonl_path = sample_dir / "session.jsonl"
        pcap_path = sample_dir / "network.pcap"
        audit_path = sample_dir / "audit.log"

        # [新增过滤]：如果该文件夹下没有任何特征文件，说明它可能是嵌套的父文件夹或无关缓存，直接跳过
        if not (jsonl_path.exists() or pcap_path.exists() or audit_path.exists()):
            continue

        sample_info = {
            "sample_id": sample_name,
            "session": None,
            "network_pcap_path": None,
            "audit_log_path": None
        }

        # === 1. 提取 Session 数据 ===
        if jsonl_path.exists():
            try:
                sample_info["session"] = extract_trajectory_from_jsonl(
                    jsonl_path=str(jsonl_path), 
                    session_id=sample_name
                )
            except Exception as e:
                print(f"  [错误] 样本 {sample_name} 解析 session.jsonl 失败: {e}")
        else:
            print(f"  [警告] 样本 {sample_name} 缺失 session.jsonl")

        # === 2. 提取 pcap 文件路径 ===
        if pcap_path.exists():
            sample_info["network_pcap_path"] = str(pcap_path)
        else:
            print(f"  [警告] 样本 {sample_name} 缺失 network.pcap")

        # === 3. 提取 audit.log 文件路径 ===
        if audit_path.exists():
            sample_info["audit_log_path"] = str(audit_path)
        else:
            print(f"  [警告] 样本 {sample_name} 缺失 audit.log")

        extracted_data.append(sample_info)
    
    return extracted_data


def trick_detect(extracted_data, keywords_in_path, keywords_match_path):
    audit_path_list = [sample["audit_log_path"] for sample in extracted_data]
    label_list = []
    
    with open(keywords_in_path, "r", encoding="utf-8") as f:
        keywords_in = [k for k in json.load(f) if k.strip()]
        
    def match_content(path):
        if path is None or not os.path.exists(path):
            return 0

        mal_flag = 0 
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            for keyword in keywords_in:
                if keyword in content:
                    mal_flag = 1
                    break
        except Exception as e:
            print(f"读取文件时出错: {e}")
            
        return mal_flag

    print("[*] 正在执行 trick_detect 规则匹配...")
    for content in tqdm(audit_path_list, desc="Trick Detect"):
        mal_flag = match_content(content)
        label_list.append(mal_flag)

    return label_list


def calculate_and_print_metrics(name, preds, truths, label_mode):
    """
    计算并输出准确率(Accuracy)、召回率(Recall)和误报率(FPR)。
    根据真实的 label_mode 动态显示有效的指标。
    """
    total = len(truths)
    if total == 0:
        print(f"  - {name} : 无效样本数据 (总数为0)")
        return

    # 统计 TP, TN, FP, FN
    tp = sum(1 for p, t in zip(preds, truths) if p == 1 and t == 1) 
    tn = sum(1 for p, t in zip(preds, truths) if p == 0 and t == 0) 
    fp = sum(1 for p, t in zip(preds, truths) if p == 1 and t == 0) 
    fn = sum(1 for p, t in zip(preds, truths) if p == 0 and t == 1) 

    actual_pos = tp + fn  # 真实的恶意样本总数
    actual_neg = fp + tn  # 真实的正常样本总数

    # 计算指标，防止 /0 错误
    accuracy = ((tp + tn) / total) * 100.0 if total > 0 else 0.0
    recall = (tp / actual_pos) * 100.0 if actual_pos > 0 else 0.0
    fpr = (fp / actual_neg) * 100.0 if actual_neg > 0 else 0.0

    # 根据不同的场景输出最合理的指标
    if label_mode == 0:
        # 全是0样本，计算召回率无意义，主看准确率和误报率
        print(f"  - {name} : 准确率 = {accuracy:.2f}%, 误报率(FPR) = {fpr:.2f}%")
    elif label_mode == 1:
        # 全是1样本，计算误报率无意义，主看准确率和召回率
        print(f"  - {name} : 准确率 = {accuracy:.2f}%, 召回率 = {recall:.2f}%")
    elif label_mode == 2:
        # 混合样本，全面输出
        print(f"  - {name} : 准确率 = {accuracy:.2f}%, 召回率 = {recall:.2f}%, 误报率(FPR) = {fpr:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="数据提取与多维度恶意行为检测")
    parser.add_argument("--dataset_path", type=str, required=True, help="数据集的目标文件夹路径")
    parser.add_argument("--keywords_in_path", type=str, required=True, help="keywords_in.json 的路径")
    parser.add_argument("--keywords_match_path", type=str, required=True, help="keywords_match.json 的路径")
    parser.add_argument("--svm_model_path", type=str, required=True, help="svm模型路径")
    parser.add_argument("--model_path", type=str, required=True, help="Session检测的主模型路径")
    parser.add_argument("--lora_path", type=str, default="", help="Session检测的LoRA参数路径（可选）")
    
    parser.add_argument("--label_mode", type=int, choices=[0, 1, 2], required=True,
                        help="0: 所有样本按0算; 1: 所有样本按1算; 2: 从results.csv读取")

    parser.add_argument("--run_pcap", action="store_true", help="启动 PCAP/流量 模型预测")
    parser.add_argument("--run_session", action="store_true", help="启动 Session 大模型预测")
    parser.add_argument("--run_trick", action="store_true", help="启动日志规则匹配检测")
    
    args = parser.parse_args()

    if not any([args.run_pcap, args.run_session, args.run_trick]):
        print("[*] 未明确指定启动模块参数，默认启动所有三个检测模块。")
        args.run_pcap = True
        args.run_session = True
        args.run_trick = True

    print(f"\n[*] 开始从 {args.dataset_path} 提取数据...")
    extracted_data = extract_data(args.dataset_path)
    
    if not extracted_data:
        print("[-] 未提取到任何有效数据，程序退出。")
        return

    # --- 获取 Ground Truth ---
    true_labels = []
    if args.label_mode == 0:
        true_labels = [0] * len(extracted_data)
        print("[*] 真实标签模式：全部置为 0 (正常)")
    elif args.label_mode == 1:
        true_labels = [1] * len(extracted_data)
        print("[*] 真实标签模式：全部置为 1 (恶意)")
    elif args.label_mode == 2:
        # [修改] 读取的文件名更新为 results.csv
        csv_path = Path(args.dataset_path) / "results.csv"
        gt_dict = {}
        if not csv_path.exists():
            print(f"[-] [错误] 参数设定为从CSV读取，但未找到 {csv_path}。程序退出。")
            return
        
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    subfolder_name, label_val = row[0].strip(), row[1].strip()
                    try:
                        gt_dict[subfolder_name] = int(label_val)
                    except ValueError:
                        pass

        for sample in extracted_data:
            sid = sample["sample_id"]
            if sid in gt_dict:
                true_labels.append(gt_dict[sid])
            else:
                print(f"  [警告] 样本 {sid} 在 results.csv 中未找到对应标签，默认认定为 0。")
                true_labels.append(0)
        print(f"[*] 真实标签模式：已从 {csv_path} 加载真实标签")

    module_results = {}

    if args.run_pcap:
        print("\n[*] 正在执行 PCAP/流量 模型预测...")
        try:
            model_labels = predict_pcap_data(extracted_data, args.svm_model_path)
            module_results["PCAP"] = model_labels
        except Exception as e:
            print(f"[-] PCAP预测执行失败: {e}")
            module_results["PCAP"] = [None] * len(extracted_data)

    if args.run_session:
        print("\n[*] 正在执行 Session 大模型预测...")
        try:
            session_labels = predict_session_data(args.model_path, extracted_data)
            module_results["Session"] = session_labels
        except Exception as e:
            print(f"[-] Session 预测执行失败: {e}")
            module_results["Session"] = [None] * len(extracted_data)

    if args.run_trick:
        trick_labels = trick_detect(
            extracted_data=extracted_data, 
            keywords_in_path=args.keywords_in_path, 
            keywords_match_path=args.keywords_match_path
        )
        module_results["Trick"] = trick_labels


    print("\n=== 检测结果统计分析 ===")
    
    if not module_results:
        print("[-] 没有模块被成功执行，无法进行统计分析。")
        return

    min_length = min(len(labels) for labels in module_results.values())
    min_length = min(min_length, len(true_labels))

    if min_length == 0:
        print("[-] 有效样本数量为 0，无法进行统计分析。")
        return

    active_modules = list(module_results.keys())
    name_map = {
        "PCAP": "PCAP 流量模型",
        "Session": "Session 大模型",
        "Trick": "日志规则匹配"
    }

    print(f"总处理并参与评估的样本数: {min_length}")

    # --- [情况 1] 各自独立的检测效果 ---
    print(f"\n[1] 各自独立的检测效果 (启动模块数: {len(active_modules)}):")
    for mod in active_modules:
        raw_preds = module_results[mod][:min_length]
        preds = [1 if p in [1, "1"] else 0 for p in raw_preds]
        calculate_and_print_metrics(name_map[mod], preds, true_labels[:min_length], args.label_mode)


    # --- [情况 2] 两两并集检测效果 ---
    if len(active_modules) >= 2:
        print("\n[2] 两两并集检测效果 (任一报毒即视为 1):")
        for mod1, mod2 in itertools.combinations(active_modules, 2):
            l1_raw = module_results[mod1][:min_length]
            l2_raw = module_results[mod2][:min_length]
            
            union_preds_2 = [
                1 if (p1 in [1, "1"] or p2 in [1, "1"]) else 0 
                for p1, p2 in zip(l1_raw, l2_raw)
            ]
            calculate_and_print_metrics(f"{name_map[mod1]} ∪ {name_map[mod2]}", union_preds_2, true_labels[:min_length], args.label_mode)


    # --- [情况 3] 三者并集检测效果 ---
    if len(active_modules) == 3:
        print("\n[3] 三者并集检测效果 (任一报毒即视为 1):")
        l1_raw = module_results[active_modules[0]][:min_length]
        l2_raw = module_results[active_modules[1]][:min_length]
        l3_raw = module_results[active_modules[2]][:min_length]
        
        union_preds_3 = [
            1 if (p1 in [1, "1"] or p2 in [1, "1"] or p3 in [1, "1"]) else 0 
            for p1, p2, p3 in zip(l1_raw, l2_raw, l3_raw)
        ]
        calculate_and_print_metrics("PCAP ∪ Session ∪ 日志规则", union_preds_3, true_labels[:min_length], args.label_mode)

if __name__ == "__main__":
    main()