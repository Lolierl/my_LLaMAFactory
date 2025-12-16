import json
from tqdm import trange
from data_gen.pretrain.id_gen import IdGen
from tools.tools import tokenizer, fix_seed

SYSTEM = r"Please reason step by step, and put your final answer within \boxed{}."

def generate_one(id_gen):
    id_gen.gen_prob([i for i in range(23)], p_format="pq")

    problem = tokenizer.decode(id_gen.prob_token)
    solution_raw = tokenizer.decode(id_gen.sol_token)
    answer = tokenizer.decode(id_gen.ans_token)
    answer = answer.strip()
    solution = solution_raw + f"\n\nSo the final answer is \\boxed{{{answer}}}."
    return {
        "problem": problem,
        "solution": solution,
        "answer": answer,
    }

def main():
    fix_seed(20050601)

    output_path = "/nfs/nfs-home/siyuan/LLaMA-Factory/data/iGSM_2to20_50each.jsonl"
    total = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for max_op in range(2, 21):
            max_edge = int(max_op * 4 / 3.0) + 1

            id_gen = IdGen(
                style="heavy",
                op_style="heavy",
                max_op=max_op,
                op=max_op,         
                max_edge=max_edge,
                perm_level=5,
                detail_level=0,
            )

            for i in trange(50, desc=f"op={max_op}"):
                sample = generate_one(id_gen)
               
                sample.update({
                    "op": max_op,
                    "max_op": max_op,
                    "max_edge": max_edge,
                    "style": "heavy",
                    "source": "id_gen",
                })
                
                if i == 0:
                    print("Sample generated:", sample)
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                total += 1

    print(f"Saved {total} samples to {output_path}")

if __name__ == "__main__":
    main()