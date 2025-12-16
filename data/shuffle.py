import json
import random

def shuffle_jsonl(input_path, output_path, seed=42):
    random.seed(seed)

    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    print(f"Loaded {len(data)} samples")

    random.shuffle(data)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Shuffled dataset saved to {output_path}")

if __name__ == "__main__":
    shuffle_jsonl(
        input_path="/nfs/nfs-home/siyuan/LLaMA-Factory/data/iGSM/iGSM_2to8_2000each.jsonl",
        output_path="/nfs/nfs-home/siyuan/LLaMA-Factory/data/iGSM/iGSM_2to8_2000each_shuffled.jsonl",
        seed=42,)