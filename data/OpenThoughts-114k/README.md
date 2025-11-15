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
    num_bytes: 2769316810
    num_examples: 113957
  download_size: 1178170708
  dataset_size: 2769316810
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
tags:
- llama-factory
size_categories:
- 100K<n<1M
---

This dataset was converted from [open-thoughts/OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) using the following script.

```python
from datasets import Dataset, load_dataset


SYSTEM = (
    "You are an assistant that thoroughly explores questions through a systematic long thinking process "
    "before providing the final precise and accurate solutions. "
    "This requires engaging in a comprehensive cycle of analysis, summarization, exploration, reassessment, "
    "reflection, backtracing, and iteration to develop a well-considered thinking process. "
    "Detail your reasoning process using the specified format: <think>thought with steps separated by '\n\n'</think> "
    "Each step should include detailed considerations such as analyzing questions, summarizing relevant findings, "
    "brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, "
    "and revisiting previous steps. Based on various attempts, explorations, and reflections from the thoughts, "
    "you should systematically present the final solution that you deem correct. "
    "The solution should remain a logical, accurate, concise expression style and detail necessary steps needed to "
    "reach the conclusion. Now, try to solve the following question through the above guidelines."
)


def generate_data(data):
    for sample in data:
        response = "<think>\n{}\n</think>\n\n{}".format(sample["deepseek_reasoning"], sample["deepseek_solution"])
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": sample["problem"]},
                {"role": "assistant", "content": response},
            ],
            "original_solution": sample["ground_truth_solution"],
            "domain": sample["domain"],
            "source": sample["source"],
        }


def main():
    data = load_dataset("open-thoughts/OpenThoughts-114k", "metadata", split="train")
    print("Data num:", len(data))
    dataset = Dataset.from_generator(generate_data, gen_kwargs={"data": data})
    dataset.push_to_hub("llamafactory/OpenThoughts-114k")


if __name__ == "__main__":
    main()
```
