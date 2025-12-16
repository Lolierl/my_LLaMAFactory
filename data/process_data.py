from datasets import Dataset, load_dataset, concatenate_datasets
import pyarrow.parquet as pq
import pyarrow as pa

SYSTEM = r"Please reason step by step, and put your final answer within \boxed{}."

def generate_data(data):
    for sample in data:
        for generation, correctness in zip(sample["generations"], sample["correctness_math_verify"]):
            #if correctness:
            yield {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": sample["problem"]},
                    {"role": "assistant", "content": generation},
                ],
                "original_solution": sample["answer"],
                "domain": sample["problem_type"],
                "source": sample["source"],
            }
            break

def main():
    # 加载 default 和 extended split
    full_data = load_dataset("/nfs/nfs-home/siyuan/LLaMA-Factory/data/OpenR1-Math-220k", "default", split="train")

    print("Total combined num:", len(full_data))

    # 生成合成数据
    dataset = Dataset.from_generator(generate_data, gen_kwargs={"data": full_data})
    dataset = dataset.shuffle(seed=42) 
    # 保存为 parquet 格式
    dataset.to_parquet("OpenR1-Math-94k_full/data.parquet")
    print("Saved to OpenR1-Math-94k_full/data.parquet")

if __name__ == "__main__":
    main()