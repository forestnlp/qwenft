import json
import re
from datetime import datetime

def clean_html_content(html):
    """清理HTML标签和多余空格"""
    clean_text = re.sub(r'<[^>]+>', '', str(html))
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

def convert_chat_to_qwen3(input_txt_path, output_qwen3_jsonl_path):
    qwen3_valid_data = []
    system_prompt = "您是邮政在线电商客服助手，需专业、礼貌地回应用户咨询，解答订单、物流、服务相关问题，语气亲切自然。"

    with open(input_txt_path, 'r', encoding='utf-8') as f_in:
        line_num = 0
        for line in f_in:
            line_num += 1
            line = line.strip()
            if not line:
                continue

            # 处理尾部逗号
            if line.endswith(','):
                line = line[:-1]

            # 解析JSON
            try:
                chat_session = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"警告：第{line_num}行JSON格式错误，跳过。错误：{str(e)[:50]}")
                continue

            # 提取会话信息
            session_id = chat_session.get('meetingid', f"session_{line_num}")
            chat_history = chat_session.get('chatHistory', [])
            agent_id = chat_session.get('agentloginid', '')
            create_time = chat_session.get('createmeetingtime', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            # 过滤并转换对话历史
            qwen3_conversations = []
            last_role = None

            for msg in chat_history:
                msg_content = msg.get('htmlBody', '')
                msg_sender = msg.get('fromRecipient', '')
                clean_content = clean_html_content(msg_content)
                
                if not clean_content:
                    continue

                # 区分角色
                current_role = None
                if msg_sender == session_id:
                    current_role = 'user'
                elif msg_sender == agent_id and agent_id:
                    current_role = 'assistant'
                elif msg_sender == '系统消息':
                    continue
                else:
                    continue

                # 确保角色交替（连续相同角色保留最新一条）
                if current_role == last_role:
                    if qwen3_conversations:
                        qwen3_conversations.pop()
                    else:
                        continue
                qwen3_conversations.append({
                    "role": current_role,
                    "content": clean_content,
                    "send_time": msg.get('sendTime', '')
                })
                last_role = current_role

            # -------------------------- 核心修改：处理用户结尾的对话 --------------------------
            # 场景1：对话轮数≥2，但结尾是user → 截断到上一个assistant（保留完整的“user→assistant”轮次）
            if len(qwen3_conversations) >= 2 and qwen3_conversations[-1]['role'] == 'user':
                # 倒序查找最后一个assistant的位置
                last_assistant_idx = None
                for i in range(len(qwen3_conversations)-1, -1, -1):
                    if qwen3_conversations[i]['role'] == 'assistant':
                        last_assistant_idx = i
                        break
                # 截断对话：保留到最后一个assistant（确保结尾是assistant）
                if last_assistant_idx is not None and last_assistant_idx >= 1:
                    qwen3_conversations = qwen3_conversations[:last_assistant_idx+1]
                    print(f"提示：第{line_num}行会话（ID：{session_id}）以用户结尾，已截断为有效轮次（{len(qwen3_conversations)}轮）")
                else:
                    # 特殊情况：只有user没有assistant → 跳过（无有效回复）
                    print(f"警告：第{line_num}行会话（ID：{session_id}）仅含用户消息，无客服回复，跳过")
                    continue

            # 场景2：对话轮数≥2且结尾是assistant → 直接保留（符合要求）
            # 验证最终格式：至少1轮完整对话（user→assistant），且结尾是assistant
            if len(qwen3_conversations) >= 2 and qwen3_conversations[-1]['role'] == 'assistant':
                # 移除send_time，保留核心字段
                final_conversations = [{"role": c["role"], "content": c["content"]} for c in qwen3_conversations]
                qwen3_entry = {
                    "system": system_prompt,
                    "conversations": final_conversations,
                    "metadata": {
                        "session_id": session_id,
                        "create_time": create_time,
                        "agent_id": agent_id,
                        "raw_line_num": line_num,
                        "original_rounds": len(chat_history),  # 原始轮数（便于追溯）
                        "final_rounds": len(final_conversations)  # 最终保留轮数
                    }
                }
                qwen3_valid_data.append(qwen3_entry)
            else:
                # 仅1轮对话（如只有user或只有assistant）→ 跳过
                print(f"警告：第{line_num}行会话（ID：{session_id}）有效轮次不足，跳过")

    # 保存结果
    with open(output_qwen3_jsonl_path, 'w', encoding='utf-8') as f_out:
        for entry in qwen3_valid_data:
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # 统计信息
    print(f"\n转换完成！")
    print(f"原始数据总行数：{line_num}")
    print(f"有效Qwen3训练数据条数：{len(qwen3_valid_data)}")
    print(f"输出路径：{output_qwen3_jsonl_path}")

if __name__ == "__main__":
    INPUT_RAW_DATA = "/root/qwenft/data/chatHis.txt"
    OUTPUT_QWEN3_DATA = "/root/qwenft/data/qwen3_finetune_data.jsonl"
    convert_chat_to_qwen3(INPUT_RAW_DATA, OUTPUT_QWEN3_DATA)