---
dataset_info:
  features:
  - name: messages
    list:
    - name: content
      dtype: string
    - name: role
      dtype: string
  - name: original_solution
    dtype: string
  - name: domain
    dtype: string
  - name: source
    dtype: string
  splits:
  - name: train
    num_bytes: 1124238153
    num_examples: 65106
  download_size: 494661604
  dataset_size: 1124238153
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
license: apache-2.0
task_categories:
- question-answering
language:
- en
size_categories:
- 10K<n<100K
tags:
- llama-factory
---

This dataset was converted from [open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) using the following script.

```python
from datasets import Dataset, load_dataset


SYSTEM = r"Please reason step by step, and put your final answer within \boxed{}."


def generate_data(data):
    for sample in data:
        for generation, correctness in zip(sample["generations"], sample["correctness_math_verify"]):
            if correctness:
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
    data = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")
    print("Data num:", len(data))
    dataset = Dataset.from_generator(generate_data, gen_kwargs={"data": data})
    dataset.push_to_hub("llamafactory/OpenR1-Math-94k")


if __name__ == "__main__":
    main()
```
