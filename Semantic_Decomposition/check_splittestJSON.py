import json
import os
import re
import pandas as pd

# === 配置部分 ===
input_folder = './Data/Refined_Data/Split_the_text/electricity_48_jsons_fewshot'  # 你的JSON文件目录
output_report = './Data/Refined_Data/Split_the_text/electricity_48_jsons_fewshot/check_JSON_report.csv'     # 输出检测报告路径

# === 辅助函数 ===

def extract_numbers(text):
    """提取浮点数或整数"""
    return re.findall(r'\d+\.\d+|\d+', text)

def contains_conflicting_words(text):
    """检测趋势表述中的冲突词"""
    ups = ['increase', 'rise', 'upward']
    downs = ['decrease', 'drop', 'decline', 'fall', 'downward']
    up_found = any(word in text.lower() for word in ups)
    down_found = any(word in text.lower() for word in downs)
    return up_found and down_found

def check_consistency(original, subtext):
    """子文本是否与原文内容一致"""
    # 提取数字
    orig_nums = extract_numbers(original)
    sub_nums = extract_numbers(subtext)
    overlap_ratio = len(set(sub_nums) & set(orig_nums)) / (len(sub_nums) + 1e-6)
    return overlap_ratio >= 0.5 or len(sub_nums) == 0

# === 主检测函数 ===
results = []

for file in os.listdir(input_folder):
    if not file.endswith('.json'):
        continue
    path = os.path.join(input_folder, file)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sample_id = data.get("SampleID", os.path.splitext(file)[0])
    original = data.get("original_text", "")
    trend = data.get("TrendText", "")
    season = data.get("SeasonText", "")

    # 检查字段完整性
    Fields_OK = all([original.strip(), trend.strip(), season.strip()])

    # 检查一致性
    trend_consistent = check_consistency(original, trend)
    season_consistent = check_consistency(original, season)

    # 检查冲突
    trend_conflict = contains_conflicting_words(trend)

    results.append({
        "SampleID": sample_id,
        "Fields_OK": Fields_OK,
        "Trend_Consistent": trend_consistent,
        "Season_Consistent": season_consistent,
        "Trend_Conflict": trend_conflict,
    })

# === 汇总与保存 ===
df = pd.DataFrame(results)
df.to_csv(output_report, index=False, encoding='utf-8-sig')

# 找出字段不完整的文件
incomplete_files = df.loc[~df["Fields_OK"], "SampleID"].tolist()
if incomplete_files:
    print("⚠️ 以下 SampleID 的文件字段不完整：")
    for fid in incomplete_files:
        print(fid)
else:
    print("✅ 所有文件字段完整。")

# === 统计汇总 ===
summary = {
    "total_files": int(len(results)),
    "valid_files": int(df["Fields_OK"].sum()),
    # "consistent_trend": int(df["Trend_Consistent"].sum()),
    # "consistent_season": int(df["Season_Consistent"].sum()),
    # "trend_conflict": int(df["Trend_Conflict"].sum()),
}

print("✅ 验证完成！")
print(json.dumps(summary, indent=2, ensure_ascii=False))
print(f"📄 检查报告已保存到：{output_report}")

