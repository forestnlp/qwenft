import json
import uuid
from datetime import datetime

def convert_txt_to_jsonl(input_txt_path, output_processed_jsonl_path, output_qwen3_jsonl_path):
    processed_data = []
    qwen3_data = []

    system_message = "您是邮政在线电商客服助手，请提供专业、礼貌、准确的回答。"

    with open(input_txt_path, 'r', encoding='utf-8') as f_in:
        content = f_in.read()
        # Assuming the file is a list of json objects, separated by commas, and enclosed in brackets
        if content.startswith('[') and content.endswith(']'):
            try:
                json_content = json.loads(content)
            except json.JSONDecodeError:
                # Try to fix common errors like trailing commas
                content = content.strip()
                if content.endswith(','):
                    content = content[:-1]
                try:
                    json_content = json.loads(f'[{content}]')
                except json.JSONDecodeError as e:
                    print(f"Failed to parse content as JSON array: {e}")
                    return
        else: # Assuming JSONL
            lines = content.strip().split('\n')
            json_content = []
            for line in lines:
                line = line.strip()
                if line.endswith(','):
                    line = line[:-1]
                if line:
                    try:
                        json_content.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Skipping malformed JSON line: {line[:100]}... Error: {e}")
                        continue
    
    for chat_session in json_content:
        chat_history = chat_session.get('chatHistory', [])
        session_id = chat_session.get('meetingid', str(uuid.uuid4().int)[:16])
        create_time = chat_session.get('createmeetingtime', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        conversations = []
        for msg in chat_history:
            if 'htmlBody' in msg and msg['htmlBody']:
                role = None
                if msg.get('fromRecipient') == session_id:
                    role = 'user'
                elif msg.get('fromRecipient') != '系统消息':
                    role = 'assistant'

                if role:
                    conversations.append({"from": role, "value": msg['htmlBody']})

        if conversations:
            # For processed_chat_data.jsonl
            processed_entry = {
                "conversations": conversations,
                "session_id": session_id,
                "create_time": create_time
            }
            processed_data.append(processed_entry)

            # For qwen3_finetune_data.jsonl
            qwen3_messages = [{"role": "system", "content": system_message}]
            
            # Ensure alternating roles for qwen3 format
            last_role = 'system'
            for conv in conversations:
                current_role = 'user' if conv['from'] == 'user' else 'assistant'
                if current_role != last_role:
                    qwen3_messages.append({"role": current_role, "content": conv['value']})
                    last_role = current_role

            if len(qwen3_messages) > 1:
                qwen3_entry = {
                    "messages": qwen3_messages,
                    "id": session_id,
                    "metadata": {"create_time": create_time}
                }
                qwen3_data.append(qwen3_entry)

    with open(output_processed_jsonl_path, 'w', encoding='utf-8') as f_out_processed:
        for entry in processed_data:
            f_out_processed.write(json.dumps(entry, ensure_ascii=False) + '\n')

    with open(output_qwen3_jsonl_path, 'w', encoding='utf-8') as f_out_qwen3:
        for entry in qwen3_data:
            f_out_qwen3.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Successfully converted {len(processed_data)} conversations.")
    print(f"Output written to {output_processed_jsonl_path} and {output_qwen3_jsonl_path}")

if __name__ == "__main__":
    input_txt = "/root/qwenft/data/chatHis.txt"
    output_processed_jsonl = "/root/qwenft/data/processed_chat_data.jsonl"
    output_qwen3_jsonl = "/root/qwenft/data/qwen3_finetune_data.jsonl"

    convert_txt_to_jsonl(input_txt, output_processed_jsonl, output_qwen3_jsonl)