from dotenv import load_dotenv
load_dotenv()

from data_process_session import extract_trajectory_from_jsonl
from pathlib import Path
from check_pcap import predict_pcap_data
from check_session import predict_session_data

import json
import os, re
from tqdm import tqdm
import argparse

def extract_data(dataset_path):
    base_path = Path(dataset_path)
    extracted_data = []

    if not base_path.exists() or not base_path.is_dir():
        print(f"[错误] 数据集路径不存在或不是文件夹: {dataset_path}")
        return extracted_data

    for sample_dir in base_path.iterdir():
        # 跳过可能存在的独立文件，只处理子文件夹
        if not sample_dir.is_dir():
            continue

        sample_name = sample_dir.name
        
        # 初始化当前样本的数据字典，缺失的内容默认值为 None
        sample_info = {
            "sample_id": sample_name,
            "session": None,  # 将存放 trajectory 字典
            "network_pcap_path": None,
            "audit_log_path": None
        }

        # 构建文件路径
        jsonl_path = sample_dir / "session.jsonl"
        pcap_path = sample_dir / "network.pcap"
        audit_path = sample_dir / "audit.log"

        # === 1. 提取 Session 数据 ===
        if jsonl_path.exists():
            try:
                # 调用工具函数解析文件，传入 sample_name 作为会话 ID
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

        # 将当前样本的信息收集起来（无论是否完整）
        extracted_data.append(sample_info)
    
    return extracted_data


def trick_detect(extracted_data, keywords_in_path, keywords_match_path):
    audit_path_list = [sample["audit_log_path"] for sample in extracted_data]
    label_list = []
    
    with open(keywords_in_path, "r", encoding="utf-8") as f:
        keywords_in = [k for k in json.load(f) if k.strip()]
    with open(keywords_match_path, "r", encoding="utf-8") as f:
        keywords_match = [k for k in json.load(f) if k.strip()]
        
    def match_content(path):
        if path is None:
            return 0
        if not os.path.exists(path):
            return 0

        mal_flag = 0 

        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            # 检查里面是否包含 keyword
            for keyword in keywords_in:
                if keyword in content:
                    mal_flag = 1
                    break
            if mal_flag == 1:
                return mal_flag
                
            """for keyword in keywords_match:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, content):
                    mal_flag = 1
                    break  # 只要命中一个关键字就跳出循环"""
        except Exception as e:
            print(f"读取文件时出错: {e}")
            
        return mal_flag

    print("[*] 正在执行 trick_detect 规则匹配...")
    for content in tqdm(audit_path_list, desc="Trick Detect"):
        mal_flag = match_content(content)
        label_list.append(mal_flag)

    return label_list


def main():
    import itertools  # 用于动态生成两两组合

    # 1. 添加所有的参数
    parser = argparse.ArgumentParser(description="数据提取与多维度恶意行为检测")
    parser.add_argument("--dataset_path", type=str, required=True, help="数据集的目标文件夹路径")
    parser.add_argument("--keywords_in_path", type=str, required=True, help="keywords_in.json 的路径")
    parser.add_argument("--keywords_match_path", type=str, required=True, help="keywords_match.json 的路径")
    parser.add_argument("--svm_model_path", type=str, required=True, help="svm模型路径")
    parser.add_argument("--model_path", type=str, required=True, help="Session检测的主模型路径")
    parser.add_argument("--lora_path", type=str, default="", help="Session检测的LoRA参数路径（可选）")
    
    # 新增：模块启动控制开关 (加上即为 True)
    parser.add_argument("--run_pcap", action="store_true", help="启动 PCAP/流量 模型预测")
    parser.add_argument("--run_session", action="store_true", help="启动 Session 大模型预测")
    parser.add_argument("--run_trick", action="store_true", help="启动日志规则匹配检测")
    
    args = parser.parse_args()

    # 如果用户没有指定任何启动参数，则默认启动所有三个模块
    if not any([args.run_pcap, args.run_session, args.run_trick]):
        print("[*] 未明确指定启动模块参数，默认启动所有三个检测模块。")
        args.run_pcap = True
        args.run_session = True
        args.run_trick = True

    # 2. 提取数据
    print(f"\n[*] 开始从 {args.dataset_path} 提取数据...")
    extracted_data = extract_data(args.dataset_path)
    
    if not extracted_data:
        print("[-] 未提取到任何有效数据，程序退出。")
        return

    # 用于动态存储各个模块的预测结果
    module_results = {}

    # 3. 流量模型预测 (predict_fusion_data)
    if args.run_pcap:
        print("\n[*] 正在执行 PCAP/流量 模型预测 (predict_fusion_data)...")
        try:
            model_labels = predict_pcap_data(extracted_data, args.svm_model_path)
            module_results["PCAP"] = model_labels
        except Exception as e:
            print(f"[-] PCAP预测执行失败: {e}")
            module_results["PCAP"] = [None] * len(extracted_data)

    # 4. Session 大模型预测 (predict_session_data)
    if args.run_session:
        print("\n[*] 正在执行 Session 大模型预测 (predict_session_data)...")
        try:
            session_labels = predict_session_data(args.model_path, args.lora_path, extracted_data)
            module_results["Session"] = session_labels
        except Exception as e:
            print(f"[-] Session 预测执行失败: {e}")
            module_results["Session"] = [None] * len(extracted_data)

    # 5. 日志规则匹配检测 (trick_detect)
    if args.run_trick:
        trick_labels = trick_detect(
            extracted_data=extracted_data, 
            keywords_in_path=args.keywords_in_path, 
            keywords_match_path=args.keywords_match_path
        )
        module_results["Trick"] = trick_labels

    # 6. 统计各维度预测为 1 的概率
    print("\n=== 检测结果统计分析 ===")
    
    if not module_results:
        print("[-] 没有模块被成功执行，无法进行统计分析。")
        return

    # 取已运行模块的最短结果长度以防异常截断
    min_length = min(len(labels) for labels in module_results.values())
    
    if min_length == 0:
        print("[-] 有效样本数量为 0，无法进行统计分析。")
        return

    # 辅助计算百分比的函数
    def calc_prob(count, total):
        return (count / total) * 100

    active_modules = list(module_results.keys())
    
    # 映射字典，让输出更易读
    name_map = {
        "PCAP": "PCAP 流量模型",
        "Session": "Session 大模型",
        "Trick": "日志规则匹配"
    }

    print(f"总处理样本数: {min_length}")

    # [步骤 1] 各自预测为 1 的概率
    print(f"\n[1] 各自独立的预测为 1 的概率 (启动模块数: {len(active_modules)}):")
    for mod in active_modules:
        labels = module_results[mod]
        count_1 = sum(1 for i in range(min_length) if labels[i] in [1, "1"])
        print(f"  - {name_map[mod]} 预测为1 : {count_1}/{min_length} ({calc_prob(count_1, min_length):.2f}%)")

    # [步骤 2] 两两并集预测为 1 的概率 (任一报毒即为 1)
    if len(active_modules) >= 2:
        print("\n[2] 两两并集预测为 1 的概率 (任一报毒即为 1):")
        # 自动生成所有激活模块的两两组合
        for mod1, mod2 in itertools.combinations(active_modules, 2):
            labels1 = module_results[mod1]
            labels2 = module_results[mod2]
            count_union_2 = sum(
                1 for i in range(min_length) 
                if labels1[i] in [1, "1"] or labels2[i] in [1, "1"]
            )
            print(f"  - {name_map[mod1]} ∪ {name_map[mod2]} : {count_union_2}/{min_length} ({calc_prob(count_union_2, min_length):.2f}%)")

    # [步骤 3] 三者并集预测为 1 的概率 (仅在三个模块全开时展示)
    if len(active_modules) == 3:
        print("\n[3] 三者并集预测为 1 的概率 (任一报毒即为 1):")
        l1, l2, l3 = module_results[active_modules[0]], module_results[active_modules[1]], module_results[active_modules[2]]
        count_union_all = sum(
            1 for i in range(min_length) 
            if l1[i] in [1, "1"] or l2[i] in [1, "1"] or l3[i] in [1, "1"]
        )
        print(f"  - PCAP ∪ Session ∪ 日志规则 : {count_union_all}/{min_length} ({calc_prob(count_union_all, min_length):.2f}%)")


if __name__ == "__main__":
    main()