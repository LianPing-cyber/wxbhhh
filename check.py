import sys
from pathlib import Path
import numpy as np

# ==========================================
# 1. 跨目录环境配置：将 sys_channel/lib 注入环境变量
# ==========================================
CURRENT_DIR = Path(__file__).resolve().parent
LIB_DIR = CURRENT_DIR / "sys_channel" / "lib"

# 将 lib 目录加入最高优先级路径，使得内部模块可以被找到
sys.path.insert(0, str(LIB_DIR))

# 导入 sys_channel 底层的特征提取与融合模型模块
import audit_pcap_fusion_svm_classifier as pipeline
from tokenize_audit_log import count_tokens_in_audit_file

# ==========================================
# 2. 三大核心预测函数 (Audit / PCAP / Fusion)
# ==========================================

def predict_audit_data(extracted_data: list[dict], model_dir: str | Path) -> list[dict]:
    """仅基于 Audit 模态进行恶意性预测"""
    svm, artifacts, config = pipeline.load_fusion_model_bundle(Path(model_dir))
    pipeline.BGE_CHUNK_POOLING = config.get("bge_chunk_pooling", "mean")
    pipeline.BGE_CHUNK_TOKENS = config.get("bge_chunk_tokens", 384)
    max_repeat = config.get("max_token_repeat", 30)

    samples, texts_to_encode, encode_idx = [], [], []

    for idx, item in enumerate(extracted_data):
        audit_path = item.get("audit_log_path")
        has_audit, audit_text = False, ""
        host_num = np.zeros(len(pipeline.HOST_NUMERIC_NAMES), dtype=np.float64)

        if audit_path and Path(audit_path).exists():
            path_obj = Path(audit_path)
            has_audit = True
            payload_text, host_num = pipeline.extract_host_side_features(path_obj)
            counter = count_tokens_in_audit_file(path_obj)
            token_text = pipeline.counter_to_document(list(counter.items()), max_repeat=max_repeat)
            audit_text = f"{token_text} {payload_text}".strip()

        sample = pipeline.SampleFeatures(
            file_name=item["sample_id"], label=-1, audit_text=audit_text, host_numeric=host_num,
            net_text="", net_numeric=np.zeros(len(pipeline.NET_NUMERIC_NAMES), dtype=np.float64),
            has_audit=has_audit, has_pcap=False
        )
        samples.append(sample)

        if has_audit and config.get("audit_encoder") == "bge-m3":
            texts_to_encode.append(audit_text)
            encode_idx.append(idx)

    audit_bge = None
    if config.get("audit_encoder") == "bge-m3":
        if texts_to_encode:
            embs = pipeline.encode_with_bge_m3(texts_to_encode)
            audit_bge = np.zeros((len(samples), embs.shape[1]), dtype=np.float64)
            for local_i, global_i in enumerate(encode_idx): audit_bge[global_i] = embs[local_i]
        else:
            audit_bge = np.zeros((len(samples), 1024), dtype=np.float64)

    y_pred = pipeline.predict_fusion_samples(samples, svm, artifacts, audit_bge_matrix=audit_bge, net_bge_matrix=None)
    
    return [
        {"sample_id": s.file_name, "prediction": int(p), "mode": "audit", "session": item["session"]} 
        for s, p, item in zip(samples, y_pred, extracted_data)
    ]


def predict_pcap_data(extracted_data: list[dict], model_dir: str | Path) -> list[dict]:
    """仅基于 PCAP 模态进行恶意性预测"""
    svm, artifacts, config = pipeline.load_fusion_model_bundle(Path(model_dir))
    
    samples, texts_to_encode, encode_idx = [], [], []

    for idx, item in enumerate(extracted_data):
        pcap_path = item.get("network_pcap_path")
        has_pcap, net_text = False, ""
        net_num = np.zeros(len(pipeline.NET_NUMERIC_NAMES), dtype=np.float64)

        if pcap_path and Path(pcap_path).exists():
            path_obj = Path(pcap_path)
            has_pcap = True
            net_num, net_text = pipeline.extract_pcap_features(path_obj)

        sample = pipeline.SampleFeatures(
            file_name=item["sample_id"], label=-1, audit_text="", 
            host_numeric=np.zeros(len(pipeline.HOST_NUMERIC_NAMES), dtype=np.float64),
            net_text=net_text, net_numeric=net_num, has_audit=False, has_pcap=has_pcap
        )
        samples.append(sample)

        if has_pcap and config.get("net_encoder") == "bge-m3":
            texts_to_encode.append(net_text)
            encode_idx.append(idx)

    net_bge = None
    if config.get("net_encoder") == "bge-m3":
        if texts_to_encode:
            embs = pipeline.encode_with_bge_m3(texts_to_encode)
            net_bge = np.zeros((len(samples), embs.shape[1]), dtype=np.float64)
            for local_i, global_i in enumerate(encode_idx): net_bge[global_i] = embs[local_i]
        else:
            net_bge = np.zeros((len(samples), 1024), dtype=np.float64)

    y_pred = pipeline.predict_fusion_samples(samples, svm, artifacts, audit_bge_matrix=None, net_bge_matrix=net_bge)
    
    return [
        {"sample_id": s.file_name, "prediction": int(p), "mode": "pcap", "session": item["session"]} 
        for s, p, item in zip(samples, y_pred, extracted_data)
    ]


def predict_fusion_data(extracted_data: list[dict], model_dir: str | Path) -> list[dict]:
    """基于 Audit 和 PCAP 双模态进行融合恶意性预测"""
    svm, artifacts, config = pipeline.load_fusion_model_bundle(Path(model_dir))
    pipeline.BGE_CHUNK_POOLING = config.get("bge_chunk_pooling", "mean")
    pipeline.BGE_CHUNK_TOKENS = config.get("bge_chunk_tokens", 384)
    max_repeat = config.get("max_token_repeat", 30)

    samples = []
    audit_texts_to_encode, audit_encode_idx = [], []
    net_texts_to_encode, net_encode_idx = [], []

    for idx, item in enumerate(extracted_data):
        # Audit 特征提取
        audit_path = item.get("audit_log_path")
        has_audit, audit_text = False, ""
        host_num = np.zeros(len(pipeline.HOST_NUMERIC_NAMES), dtype=np.float64)
        if audit_path and Path(audit_path).exists():
            has_audit = True
            payload_text, host_num = pipeline.extract_host_side_features(Path(audit_path))
            counter = count_tokens_in_audit_file(Path(audit_path))
            token_text = pipeline.counter_to_document(list(counter.items()), max_repeat=max_repeat)
            audit_text = f"{token_text} {payload_text}".strip()

        # PCAP 特征提取
        pcap_path = item.get("network_pcap_path")
        has_pcap, net_text = False, ""
        net_num = np.zeros(len(pipeline.NET_NUMERIC_NAMES), dtype=np.float64)
        if pcap_path and Path(pcap_path).exists():
            has_pcap = True
            net_num, net_text = pipeline.extract_pcap_features(Path(pcap_path))

        # 封装为底层的 SampleFeatures
        sample = pipeline.SampleFeatures(
            file_name=item["sample_id"], label=-1, audit_text=audit_text, host_numeric=host_num,
            net_text=net_text, net_numeric=net_num, has_audit=has_audit, has_pcap=has_pcap
        )
        samples.append(sample)

        # 收集按需 BGE 编码的文本
        if has_audit and config.get("audit_encoder") == "bge-m3":
            audit_texts_to_encode.append(audit_text)
            audit_encode_idx.append(idx)
        if has_pcap and config.get("net_encoder") == "bge-m3":
            net_texts_to_encode.append(net_text)
            net_encode_idx.append(idx)

    # 批量 BGE 编码 Audit
    audit_bge = None
    if config.get("audit_encoder") == "bge-m3":
        if audit_texts_to_encode:
            embs = pipeline.encode_with_bge_m3(audit_texts_to_encode)
            audit_bge = np.zeros((len(samples), embs.shape[1]), dtype=np.float64)
            for local_i, global_i in enumerate(audit_encode_idx): audit_bge[global_i] = embs[local_i]
        else:
            audit_bge = np.zeros((len(samples), 1024), dtype=np.float64)

    # 批量 BGE 编码 Net
    net_bge = None
    if config.get("net_encoder") == "bge-m3":
        if net_texts_to_encode:
            embs = pipeline.encode_with_bge_m3(net_texts_to_encode)
            net_bge = np.zeros((len(samples), embs.shape[1]), dtype=np.float64)
            for local_i, global_i in enumerate(net_encode_idx): net_bge[global_i] = embs[local_i]
        else:
            net_bge = np.zeros((len(samples), 1024), dtype=np.float64)

    # 调用融合预测模型
    y_pred = pipeline.predict_fusion_samples(samples, svm, artifacts, audit_bge_matrix=audit_bge, net_bge_matrix=net_bge)
    
    # 结构化返回数据，保留 session 字典供后续模块继续接力
    return [
        {"sample_id": s.file_name, "prediction": int(p), "mode": "fusion", "session": item["session"]} 
        for s, p, item in zip(samples, y_pred, extracted_data)
    ]