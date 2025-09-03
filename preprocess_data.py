import json
from tqdm import tqdm

def preprocess_chat_history(input_file, output_file, limit=20):
    processed_data = []
    with open(input_file, 'r', encoding='utf-8') as f_in:
        content = f_in.read()
        decoder = json.JSONDecoder()
        idx = 0
        processed_count = 0
        with tqdm(total=limit, desc="Processing chat history") as pbar:
            while idx < len(content) and processed_count < limit:
                try:
                    data, new_idx = decoder.raw_decode(content[idx:])
                    idx += new_idx
                    chat_history = data.get('chatHistory', [])
                    # Skip any whitespace or non-JSON characters between objects
                    while idx < len(content) and content[idx].isspace():
                        idx += 1
                except json.JSONDecodeError:
                    # If a JSON object cannot be decoded, advance by one character and try again
                    idx += 1
                    continue

                messages = []
                for chat in chat_history:
                    role = ""
                    message_content = chat.get('body', '').strip()
                    external_nickname = chat.get('externalNickName', '')

                    if message_content:
                        if external_nickname:
                            role = "assistant"
                        else:
                            role = "user"
                        messages.append({"role": role, "content": message_content})
                
                if messages:
                    processed_data.append({"messages": messages})
                    processed_count += 1
                    pbar.update(1)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for entry in processed_data:
            json.dump(entry, f_out, ensure_ascii=False)
            f_out.write('\n')

    print(f"Processed {len(processed_data)} entries and saved to {output_file}")

if __name__ == "__main__":
    input_json_file = "/root/qwenft/data/chatHis.txt"
    output_jsonl_file = "/root/qwenft/data/processed_chat_data.jsonl"
    preprocess_chat_history(input_json_file, output_jsonl_file, limit=20)