import os
import json
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import openai
import time

# ===== 配置 =====
base_url = "https://yunwu.ai/v1"
api_key = "sk-af3eJRpmGiBkXzgGA9GUAWmZFTG4ZZDCZu68IYDArVf37enE"
client = openai.OpenAI(api_key=api_key, base_url=base_url)

source_directory = './Data/Refined_Data/Split_the_text/electricity_48_jsons_fewshot'  # 已有的 JSON 文件目录
target_json_directory = './Data/Refined_Data/Split_the_text/electricity_48_emb_jsons_fewshot'
input_csv_path = './Data/Refined_Data/electricity_48_refined.csv'  # 原始 CSV
output_csv_path = './Data/Split_Text_Data/electricity_48_SplitText_fewshot.csv'
# source_directory = './Data/Refined_Data/Split_the_text/electricity_24_jsons_structured'  # 已有的 JSON 文件目录
# target_json_directory = './Data/Refined_Data/Split_the_text/electricity_24_emb_jsons_structured'
# input_csv_path = './Data/Refined_Data/electricity_24_refined.csv'  # 原始 CSV
# output_csv_path = './Data/Split_Text_Data/electricity_24_SplitText_structured.csv'
embedding_dim = 128
max_workers = 64
max_retries = 3

# ===== 获取 embedding（可选，如果 JSON 已有就跳过） =====
def get_embedding_with_retry(text, retries=max_retries):
    text = text.replace("\n", " ")
    for attempt in range(retries):
        try:
            return client.embeddings.create(
                input=[text],
                model="text-embedding-3-large",
                dimensions=embedding_dim
            ).data[0].embedding
        except Exception as e:
            print(f"⚠️ Embedding request failed (attempt {attempt+1}/{retries}): {e}")
            time.sleep(1)
    print(f"❌ Failed to get embedding after {retries} retries. Returning zero vector.")
    return [0]*embedding_dim

# ===== 处理单个 JSON 文件 =====
def process_file(filename):
    source_path = os.path.join(source_directory, filename)
    target_path = os.path.join(target_json_directory, filename)

    # ----- Step 1: 如果目标 JSON 已存在，先检查它是否合法 -----
    if os.path.exists(target_path):
        try:
            with open(target_path, 'r', encoding='utf-8') as f:
             return json.load(f)
        except json.JSONDecodeError as e:
            print("\n\n====================== JSON 解析错误 ======================")
            print(f"❌ 错误文件: {filename}")
            print(f"❌ 位置: line {e.lineno}, column {e.colno}")
            print(f"❌ 错误信息: {e.msg}")

            # 打印附近内容
            with open(target_path, "r", encoding="utf-8") as f2:
                lines = f2.readlines()
                err_line = lines[e.lineno - 1].rstrip("\n")
                print("\n错误行内容：")
                print(err_line)

                # 指示箭头
                print(" " * (e.colno - 1) + "↑")

            print("===========================================================\n")
            raise  # 直接抛出，让你看到具体错误
        except Exception as e:
            print(f"❌ 无法读取 JSON 文件（{filename}）: {e}")
            raise

    # ----- Step 2: 读取源 JSON -----
    try:
        with open(source_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        trend_text = data.get("TrendText", "")
        season_text = data.get("SeasonText", "")

        # 生成 embedding（如果 JSON 已有 embedding 可直接跳过）
        # ----- Step 3: 生成 embedding -----
        embeddings = {}
        for key, text in zip(["TrendTextEmb","SeasonTextEmb"],
                             [trend_text, season_text]):
            embeddings[key] = get_embedding_with_retry(text) if text else [0]*embedding_dim

        data.update(embeddings)

        # 写回 JSON
        # ----- Step 4: 写入新 JSON -----
        if not os.path.exists(target_json_directory):
            os.makedirs(target_json_directory)
        with open(target_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        return data

    except json.JSONDecodeError as e:
        # 如果源 JSON 出错，精确打印错误
        print("\n\n====================== 源 JSON 解析错误 ======================")
        print(f"❌ 错误文件: {filename}")
        print(f"❌ 位置: line {e.lineno}, column {e.colno}")
        print(f"❌ 错误信息: {e.msg}")
        with open(source_path, "r", encoding="utf-8") as f2:
            lines = f2.readlines()
            err_line = lines[e.lineno - 1].rstrip("\n")
            print("\n错误行内容：")
            print(err_line)
            print(" " * (e.colno - 1) + "↑")
        print("===========================================================\n")
        raise
    except Exception as e:
        print(f"❌ Error processing {filename}: {e}")
        return None

# ===== Step 1: 并行处理 JSON 文件 =====
json_files = sorted([f for f in os.listdir(source_directory) if f.endswith('.json')],
                    key=lambda x: int(re.search(r'(\d+)', x).group(1)))

all_json_data = []
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(process_file, f): f for f in json_files}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing JSON files"):
        result = future.result()
        if result is not None:
            all_json_data.append(result)

print(f"✅ Processed {len(all_json_data)} JSON files with embeddings.")

# ===== Step 2: 更新已有 CSV =====

import chardet
try:
    df = pd.read_csv(input_csv_path, encoding='utf-8-sig')
except UnicodeDecodeError:
    try:
        df = pd.read_csv(input_csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_csv_path, encoding='gbk')


# 建立 SampleID -> JSON 数据的映射
json_map = {str(item.get("SampleID", "")): item for item in all_json_data}

# 遍历 CSV，更新列
for idx, row in df.iterrows():
    sample_id = str(row['SampleID'])
    if sample_id in json_map:
        item = json_map[sample_id]
        df.at[idx, 'TrendText'] = item.get('TrendText', '')
        df.at[idx, 'TrendTextEmb'] = json.dumps(item.get('TrendTextEmb', []))
        df.at[idx, 'SeasonText'] = item.get('SeasonText', '')
        df.at[idx, 'SeasonTextEmb'] = json.dumps(item.get('SeasonTextEmb', []))

# 保存 CSV
df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
print(f"✅ CSV updated with SplitText and embeddings saved to {output_csv_path}")
