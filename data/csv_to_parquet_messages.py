
import sys
import re
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
def parse_prompt_to_messages(prompt_text: str):
    """
    Parse the prompt text for tokens like:
    <|im_start|>role\n...content...\n<|im_end|>
    Returns list of {"role": role, "content": content} in the order found.
    """
    if prompt_text is None:
        return []
    pattern = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>", re.DOTALL | re.IGNORECASE)
    messages = []
    for m in pattern.finditer(prompt_text):
        role = m.group(1).lower()
        content = m.group(2)
        # strip leading/trailing whitespace/newlines but preserve internal structure
        content = content.strip()
        messages.append({"role": role, "content": content})
    return messages

def convert_row(prompt_text, response_text):
    """
    Build messages list from prompt and response.
    Append the response as an assistant message (unless empty/NaN).
    """
    msgs = parse_prompt_to_messages(prompt_text)
    # If response_text is non-empty, append as assistant content
    if response_text is not None and str(response_text).strip() != "":
        # If the last parsed message is already an assistant and its content equals the response,
        # avoid duplicating (safe-guard).
        append_assistant = True
        if msgs:
            last = msgs[-1]
            if last.get("role") == "assistant" and last.get("content").strip() == str(response_text).strip():
                append_assistant = False
        if append_assistant:
            msgs.append({"role": "assistant", "content": str(response_text).strip()})
    return msgs

def main(in_csv, out_parquet):
    df = pd.read_csv(in_csv, dtype=str)  # read everything as string to avoid NaNs for parsing
    # Ensure columns exist
    if "prompt" not in df.columns or "response" not in df.columns:
        raise SystemExit("输入 CSV 必须包含 'prompt' 和 'response' 两列。")
    # Build messages for each row and store as JSON string for reliable parquet storage
    messages_json_list = []
    messages_obj_list = []  # optional: keep Python objects in memory if needed
    for idx, row in df.iterrows():
        prompt_text = row.get("prompt", "")
        response_text = row.get("response", "")
        msgs = convert_row(prompt_text, response_text)
        messages_obj_list.append(msgs)
        if idx == 0:
            print("示例解析的消息对象：", msgs)
        messages_json_list.append(json.dumps(msgs, ensure_ascii=False))
    # attach to dataframe
    df["messages_json"] = messages_json_list
    # Save to parquet
    # Pandas to_parquet uses pyarrow or fastparquet. We prefer pyarrow if available.
    schema = pa.schema([
        pa.field("messages", pa.list_(
            pa.struct([
                pa.field("role", pa.string()),
                pa.field("content", pa.string()),
            ])
        ))
    ])

    table = pa.Table.from_pandas(pd.DataFrame({"messages": messages_obj_list}), schema=schema, preserve_index=False)
    pq.write_table(table, out_parquet)
    print(f"Saved {len(df)} rows to {out_parquet}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python csv_to_parquet_messages.py input.csv output.parquet")
        sys.exit(1)
    in_csv = sys.argv[1]
    out_parquet = sys.argv[2]
    main(in_csv, out_parquet)

    df = pd.read_parquet(out_parquet)
    print(df.iloc[0]["messages"])