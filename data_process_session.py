import json
import os
from typing import Any, Dict, List, Optional

def _extract_user_content(message_obj: Dict) -> str:
    content_list = message_obj.get("content", [])
    texts = [item["text"] for item in content_list if item.get("type") == "text"]
    return "\n".join(texts)

def _extract_assistant_parts(message_obj: Dict) -> Dict[str, str]:
    content_list = message_obj.get("content", [])
    texts = []
    tool_calls = []

    for item in content_list:
        if item.get("type") == "text":
            texts.append(item["text"])
        elif item.get("type") == "toolCall":
            tool_call = {
                "id": item.get("id"),
                "name": item.get("name"),
                "arguments": item.get("arguments"),
            }
            tool_calls.append(tool_call)

    thought = "\n".join(texts)
    if len(tool_calls) == 1:
        action = json.dumps(tool_calls[0], ensure_ascii=False)
    elif len(tool_calls) > 1:
        action = json.dumps(tool_calls, ensure_ascii=False)
    else:
        action = ""

    return {"thought": thought, "action": action}

def _extract_tool_result_content(message_obj: Dict) -> str:
    content_list = message_obj.get("content", [])
    texts = [item["text"] for item in content_list if item.get("type") == "text"]
    if not texts and content_list:
        return json.dumps(content_list, ensure_ascii=False)
    return "\n".join(texts)

def _is_duplicate_final_message(message_obj: Dict, prev_message_obj: Optional[Dict]) -> bool:
    if prev_message_obj is None:
        return False
    if message_obj.get("role") != "assistant" or prev_message_obj.get("role") != "assistant":
        return False

    msg_content = message_obj.get("content", [])
    prev_content = prev_message_obj.get("content", [])
    if msg_content == prev_content:
        return True

    msg_api = message_obj.get("api", "")
    prev_api = prev_message_obj.get("api", "")
    if msg_api == "cli" and prev_api == "openai-completions":
        msg_texts = [c.get("text", "") for c in msg_content if c.get("type") == "text"]
        prev_texts = [c.get("text", "") for c in prev_content if c.get("type") == "text"]
        if msg_texts and msg_texts == prev_texts:
            return True

    return False

def _build_contents(messages: List[Dict]) -> List[List[Dict]]:
    contents: List[List[Dict]] = []
    current_round: List[Dict] = []
    prev_message_obj: Optional[Dict] = None

    for entry in messages:
        msg = entry["message"]
        role = msg["role"]

        if _is_duplicate_final_message(msg, prev_message_obj):
            continue

        if role == "user":
            if current_round:
                contents.append(current_round)
                current_round = []
            current_round.append({"role": "user", "content": _extract_user_content(msg)})

        elif role == "assistant":
            parts = _extract_assistant_parts(msg)
            if not parts["thought"] and not parts["action"]:
                prev_message_obj = msg
                continue
            current_round.append({
                "role": "agent",
                "thought": parts["thought"],
                "action": parts["action"],
            })

        elif role == "toolResult":
            current_round.append({"role": "environment", "content": _extract_tool_result_content(msg)})

        prev_message_obj = msg

    if current_round:
        contents.append(current_round)

    return contents

# ==========================================
# 核心工具函数
# ==========================================

def extract_trajectory_from_jsonl(jsonl_path: str, session_id: Optional[str] = None, profile: Optional[str] = None) -> Dict[str, Any]:
    """
    读取指定的 jsonl 文件，提取并转换为 trajectory 格式的数据结构。

    :param jsonl_path: jsonl 文件的完整路径
    :param session_id: 会话 ID。如果不提供，则默认使用文件名（不含后缀）
    :param profile: Agent 的 profile 设定。如果不提供，则使用默认的提示词
    :return: 包含 conv_id, profile, contents 的字典
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"找不到指定的 jsonl 文件: {jsonl_path}")

    # 读取并逐行解析 JSON
    with open(jsonl_path, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    all_entries = []
    for raw_line in raw_lines:
        try:
            all_entries.append(json.loads(raw_line))
        except json.JSONDecodeError:
            pass # 直接跳过无法解析的行

    # 过滤出消息实体
    message_entries = [e for e in all_entries if e.get("type") == "message"]
    
    # 构建对话内容
    contents = _build_contents(message_entries)

    # 处理默认参数
    if session_id is None:
        session_id = os.path.splitext(os.path.basename(jsonl_path))[0]
        
    if profile is None:
        profile = (
            "You are a helpful assistant.\n"
            "Available tools: read, exec, process, web_search, web_fetch, nodes, agents_list"
        )

    # 返回组装好的数据字典
    return {
        "conv_id": session_id,
        "profile": profile,
        "contents": contents,
    }