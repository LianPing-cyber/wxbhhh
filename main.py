
from dotenv import load_dotenv
load_dotenv()

from wxbhhh.data_process_session import extract_trajectory_from_jsonl
from pathlib import Path
from wxbhhh.check import predict_fusion_data

import json
import os,re
from tqdm import tqdm

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
                # 如果文件内容损坏导致解析出错，也会警告且不中断
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


def quick_detect(extracted_data):
    audit_path_list = [sample["audit_log_path"] for sample in extracted_data]
    label_list = []
    keywords_in = []
    keywords_match = []
    with open("keywords_in.json", "r", encoding="utf-8") as f:
        keywords_in = [k for k in json.load(f) if k.strip()]
    with open("keywords_match.json", "r", encoding="utf-8") as f:
        keywords_match = [k for k in json.load(f) if k.strip()]
        
    def match_content(path):
        if path is None:
            return 0
        if not os.path.exists(path):
            return 

        # 默认初始化 mal_flag 为 0（或者你可以在函数外部定义它）
        mal_flag = 0 

        try:
            # 2. 读取文件内容（这里以文本模式读取，根据需要可以改用 'rb'）
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            # 3. 检查里面是否包含 keyword
            for keyword in keywords_in:
                if keyword in content:
                    print(keyword)
                    print("okok")
                    mal_flag = 1
                    break
            if mal_flag == 1:
                return mal_flag
            for keyword in keywords_match:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, content):
                    print(keyword)
                    print("okok")
                    mal_flag = 1
                    break  # 只要命中一个关键字就跳出循环
        except Exception as e:
            print(f"读取文件时出错: {e}")
        return mal_flag

    for content in tqdm(audit_path_list):
        mal_flag = match_content(content)
        if mal_flag == 1:
            label_list.append(1)
        else:
            label_list.append(0)
    
    """with open('keywords_filtered.json', 'w', encoding="utf-8") as f:
        json.dump(keywords, f, indent=2, ensure_ascii=False)"""

    print(sum(label_list)/len(label_list))


def main():
    #extract 5 dimenstrates information of all files using extract data
    extracted_data = extract_data("/data/yuchen.yang/new_data")
    quick_detect(extracted_data)

    

if __name__ == "__main__":
    main()