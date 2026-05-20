import os
from openai import OpenAI
import pandas as pd
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import chardet
from tqdm import tqdm

# ---------------------- 配置 ----------------------
base_url = "https://ai.nengyongai.cn/v1"
api_key = "sk-JD48JDzodv0j6rbjliKVa8AXJJN7tBv5yLT74nThzvq5Fnhg"

client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

dataset_path = "./Data/Refined_Data/traffic_96_refined.csv"
saved_path = "./Data/Refined_Data/Split_the_text/traffic_96_jsons"
os.makedirs(saved_path, exist_ok=True)

progress_file = os.path.join(saved_path, "progress.txt")
error_log = os.path.join(saved_path, "error_log.txt")

max_retries = 3
max_workers = 64  # 并行线程数

lock = Lock()  # 用于线程安全写文件

# ---------------------- CSV读取 ----------------------
with open(dataset_path, 'rb') as f:
    result = chardet.detect(f.read(50000))
    encoding = result['encoding']
    if encoding is None or encoding.lower() == 'ascii':
        encoding = 'utf-8-sig'

try:
    data = pd.read_csv(dataset_path, encoding=encoding)
except UnicodeDecodeError:
    print(f"⚠️ 使用 {encoding} 解码失败，尝试使用 gbk ...")
    data = pd.read_csv(dataset_path, encoding='gbk')

# 检查必要列
for col in ["Text", "SampleID"]:
    if col not in data.columns:
        raise ValueError(f"❌ CSV 文件中未找到 '{col}' 列！")

# 已存在 JSON
existing_jsons = {
    f.split("_")[1].split(".")[0]
    for f in os.listdir(saved_path)
    if f.startswith("sample_") and f.endswith(".json")
}
print(f"📂 已检测到 {len(existing_jsons)} 个已处理 SampleID，将自动跳过。")

# ---------------------- 函数定义 ----------------------
def get_completion(user_prompt):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are an expert in time series interpretation. "
                "Split the following overall description into three concise parts: "
                "trend, seasonality, and residual. Respond strictly in JSON format "
                "with keys 'TrendText', 'SeasonText', and 'ResidualText'.")},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
    )
    return completion.choices[0].message.content

def parse_json_from_output(output_data: str):
    try:
        return json.loads(output_data.strip())
    except:
        if "```json" in output_data:
            start = output_data.find("```json") + 7
            end = output_data.find("```", start)
            return json.loads(output_data[start:end].strip())
        elif "```" in output_data:
            start = output_data.find("```") + 3
            end = output_data.find("```", start)
            return json.loads(output_data[start:end].strip())
        else:
            raise ValueError("模型输出不是有效 JSON:\n" + output_data)

def ensure_nonempty_fields(parsed):
    defaults = {
        "TrendText": "No significant trend detected.",
        "SeasonText": "No apparent seasonality.",
        "ResidualText": "No notable residual fluctuations."
    }
    for key, default in defaults.items():
        if not parsed.get(key, "").strip():
            parsed[key] = default
    return parsed

def save_to_json(sample_id, Text, parsed):
    data_dict = {
        "SampleID": sample_id,
        "original_text": Text,
        "TrendText": parsed.get("TrendText", ""),
        "SeasonText": parsed.get("SeasonText", ""),
        "ResidualText": parsed.get("ResidualText", "")
    }
    save_path = os.path.join(saved_path, f"sample_{sample_id}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)

def update_progress(sample_id):
    with lock:
        with open(progress_file, "w") as f:
            f.write(str(sample_id))

def log_error(sample_id, err_msg):
    with lock:
        with open(error_log, "a", encoding="utf-8") as f:
            f.write(f"SampleID={sample_id} 失败: {err_msg}\n")

def process_sample(row):
    sample_id = str(row["SampleID"])
    Text = str(row["Text"])
    if sample_id in existing_jsons:
        return

    user_prompt = f"""
Split the following time series description into:
1. TrendText
2. SeasonText
3. ResidualText

Output in JSON format:
{{
  "TrendText": "...",
  "SeasonText": "...",
  "ResidualText": "..."
}}

Description:
{Text}
"""
    retries = 0
    while retries < max_retries:
        try:
            output = get_completion(user_prompt)
            parsed = parse_json_from_output(output)
            parsed = ensure_nonempty_fields(parsed)
            save_to_json(sample_id, Text, parsed)
            update_progress(sample_id)
            return
        except Exception as e:
            retries += 1
            time.sleep(2)
            if retries == max_retries:
                log_error(sample_id, str(e))

# ---------------------- 多线程处理 ----------------------
rows_to_process = [row for idx, row in data.iterrows() if str(row["SampleID"]) not in existing_jsons]
print(f"⚡ 需要处理 {len(rows_to_process)} 个样本...")

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(process_sample, row): row["SampleID"] for row in rows_to_process}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
        try:
            future.result()
        except Exception as e:
            sample_id = futures[future]
            print(f"❌ SampleID={sample_id} 出现异常: {e}")

print("✅ 全部样本处理完成！")
