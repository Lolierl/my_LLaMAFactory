import os
import pandas as pd

# ==== 配置路径 ====
input_dir = "./OpenThoughts-114k/data"   # 输入目录，里面全是 parquet 文件
output_file = "./OpenThoughts-114k/OpenThoughts-114k_math.parquet" # 输出文件

# ==== 读取并筛选 ====
filtered_dfs = []

for root, dirs, files in os.walk(input_dir):
    for filename in files:
        if filename.endswith(".parquet"):
            path = os.path.join(root, filename)
            df = pd.read_parquet(path)
            if "domain" in df.columns:
                df_math = df[df["domain"] == "math"]
                filtered_dfs.append(df_math)

# 合并所有筛选结果
if filtered_dfs:
    result_df = pd.concat(filtered_dfs, ignore_index=True)
    result_df.to_parquet(output_file, engine="pyarrow", index=False)
    print(f"筛选完成，共 {len(result_df)} 条 math 数据，已保存到 {output_file}")
else:
    print("没有找到 domain 为 math 的数据。")