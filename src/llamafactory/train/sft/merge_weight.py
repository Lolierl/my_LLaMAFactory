import pickle
import os
from datasets import Dataset
import time
def merge_weight(
    pkl_path_1: str,
    pkl_path_2: str,
    output_path: str,
    weight2_key: str = "weights2"
):
    """
    Merge 'weights' from dataset_module_2 into dataset_module_1 as 'weights2'.
    Both must have same structure and ordering.
    """
    # === Load ===
    with open(pkl_path_1, 'rb') as f:
        dataset_module_1 = pickle.load(f)
    with open(pkl_path_2, 'rb') as f:
        dataset_module_2 = pickle.load(f)
    print(f"Loaded two dataset modules:\n  - {pkl_path_1}\n  - {pkl_path_2}")

    merged_dataset_module = {}

    # === Merge ===
    for name in dataset_module_1.keys():
        ds1 = dataset_module_1[name]
        ds2 = dataset_module_2[name]
        assert len(ds1) == len(ds2), f"Dataset {name} length mismatch: {len(ds1)} vs {len(ds2)}"
        # 提取第二个的weights列
        weights2 = ds2["weights"]
        assert len(weights2) == len(ds1), f"Weight2 length mismatch in {name}"

        # 添加到第一个dataset中
        ds_merged = ds1.add_column(weight2_key, weights2)
        merged_dataset_module[name] = ds_merged
        print(f"✅ Added '{weight2_key}' to dataset '{name}' ({len(ds1)} examples)")

    # === Save ===
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(merged_dataset_module, f)
    print(f"✅ Merged dataset module saved to: {output_path}")

    # === Verify ===
    for name, ds in merged_dataset_module.items():
        assert "weights" in ds.column_names and weight2_key in ds.column_names
    print("✅ Verification passed: both 'weights' and 'weights2' present in all datasets.")

    return merged_dataset_module

if __name__ == "__main__":
    pkl_path_1 = "./cache_datasets/processed_dataset_module_cache_base.pkl"
    pkl_path_2 = "./cache_datasets/processed_dataset_module_cache_OneShotRLVR.pkl"
    output_path = "./cache_datasets/merged_dataset_module_cache_base_OneShotRLVR.pkl"

    merged_dataset_module = merge_weight(
        pkl_path_1,
        pkl_path_2,
        output_path,
        weight2_key="weights2"
    )
    with open(output_path, 'wb') as f:
        pickle.dump(merged_dataset_module, f)
    print(f"Processed dataset module cached at: {output_path}")