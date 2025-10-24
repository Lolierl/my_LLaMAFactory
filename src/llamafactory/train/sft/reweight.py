import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from logging import getLogger
from ...hparams import RLreweightingArguments
from tqdm import tqdm
import torch.nn.functional as F
import os
logger = getLogger(__name__)
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
def print_gpu_mem(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"[GPU Mem {tag}] allocated: {allocated:.2f} MB, reserved: {reserved:.2f} MB")
    else:
        print("[GPU Mem] CUDA not available")

class RLWeightCalculator:
    def __init__(self, args: RLreweightingArguments):
        self.args = args
        self.rl_model = None
        self.rl_tokenizer = None
        
        if args.rl_model_name is not None:
            self._load_rl_model()
    
    def _load_rl_model(self):
        """Load RL model and tokenizer"""
        logger.info(f"Loading RL model: {self.args.rl_model_name}")
        self.rl_tokenizer = AutoTokenizer.from_pretrained(self.args.rl_model_name, use_fast=True)
        
        with torch.no_grad():
            self.rl_model = AutoModelForCausalLM.from_pretrained(
                self.args.rl_model_name
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.rl_model.eval()
        logger.info(f"RL model loaded on device: {self.rl_model.device}")

    def _calculate_token_log_probs_batch(self, batch_input_ids: List[List[int]]) -> List[torch.Tensor]:
        if self.rl_model is None:
            return [torch.ones(len(ids)) for ids in batch_input_ids]

        # ---- padding ----
        max_len = max(len(ids) for ids in batch_input_ids)
        pad_token_id = self.rl_tokenizer.pad_token_id or self.rl_tokenizer.eos_token_id
        input_tensor = torch.full(
            (len(batch_input_ids), max_len),
            pad_token_id,
            dtype=torch.long,
            device=self.rl_model.device
        )
        attention_mask = torch.zeros_like(input_tensor)

        for i, ids in enumerate(batch_input_ids):
            input_tensor[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :len(ids)] = 1

        # ---- forward ----
        with torch.no_grad(), torch.cuda.amp.autocast():
            logits = self.rl_model(input_tensor, attention_mask=attention_mask, output_hidden_states=False).logits # (B, L, V)

            # shift for prediction alignment
            shift_logits = logits[:, :-1, :]   # (B, L-1, V)
            shift_labels = input_tensor[:, 1:] # (B, L-1)

            results = []
            for i, ids in enumerate(batch_input_ids):
                seq_len = len(ids)
                # [L-1, V]
                seq_logits = shift_logits[i, :seq_len-1, :]  
                # [L-1]
                seq_labels = shift_labels[i, :seq_len-1]      

                token_log_probs = -F.cross_entropy(
                    seq_logits, seq_labels, reduction="none"
                )  # [L-1]

                first_token_log_prob = torch.tensor([0.0], device=token_log_probs.device)
                all_log_probs = torch.cat([first_token_log_prob, token_log_probs])  # (L,)

                results.append(all_log_probs.cpu().tolist())

                del seq_labels, token_log_probs, all_log_probs
            del input_tensor, attention_mask, logits, shift_logits, shift_labels
            
        return results

    def add_weights_to_dataset(self, dataset_module: Dict[str, Dataset], batch_size: int = 4) -> Dict[str, Dataset]:
        """Add weight column to dataset with batching"""
        if self.rl_model is None:
            logger.warning("No RL model loaded, skipping weight calculation")
            return dataset_module
        for dataset_name, dataset in dataset_module.items():
            logger.info(f"Calculating weights for {dataset_name} with {len(dataset)} examples")

            total = len(dataset)
            all_weights = []
            for start in tqdm(range(0, total, batch_size), desc="Calculating token log probs", ncols=100):
                end = min(start + batch_size, total)
                batch = dataset[start:end]
                input_ids_list = []
                if isinstance(batch, dict):
                    input_ids_list = batch["input_ids"]
                else:
                    for ex in batch:
                        input_ids_list.append(ex["input_ids"])

                batch_weights = self._calculate_token_log_probs_batch(input_ids_list)
                all_weights.extend(batch_weights)

                if self.rl_model.device.type == "cuda":
                    torch.cuda.empty_cache()

            assert len(all_weights) == len(dataset), f"weights length mismatch {len(all_weights)} vs {len(dataset)}"
            dataset_module[dataset_name] = dataset.add_column("weights", all_weights)
            logger.info(f"Added weights to {dataset_name}")

        return dataset_module

def calculate_and_add_weights(
    dataset_module: Dict[str, Dataset],
    args
) -> Dict[str, Dataset]:
    """Convenience function to calculate and add weights to dataset"""
    if args.rl_model_name is None:
        return dataset_module
    
    weight_calculator = RLWeightCalculator(args)
    return weight_calculator.add_weights_to_dataset(dataset_module)

if __name__ == "__main__":
    import pickle
    import os
    output_dir = "./cache_datasets"
    os.makedirs(output_dir, exist_ok=True)

    dataset_cache_path = os.path.join(output_dir, "dataset_module_pi1.pkl")
    with open(dataset_cache_path, 'rb') as f:
        dataset_module = pickle.load(f)
    print("Dataset module loaded from cache")

    RLreweighting_args = RLreweightingArguments(
        rl_model_name="/nfs/nfs-home/siyuan/open-r1/models/OneShotRLVR",
        max_clip_value=6,
        min_clip_value=0.1,
        rl_reweighting_temperature=0.6,
    )
    processed_dataset_module = calculate_and_add_weights(dataset_module, RLreweighting_args)

    processed_dataset_cache_path = os.path.join(output_dir, "processed_dataset_module_pi1_OneShotRLVR.pkl")
    with open(processed_dataset_cache_path, 'wb') as f:
        pickle.dump(processed_dataset_module, f)
    print(f"Processed dataset module cached at: {processed_dataset_cache_path}")

    processed_dataset_cache_path = os.path.join(output_dir, "processed_dataset_module_pi1_OneShotRLVR.pkl")
    with open(processed_dataset_cache_path, 'rb') as f:
        dataset_module = pickle.load(f)
    for dataset_name, dataset in dataset_module.items():
        for i, example in enumerate(dataset):
            assert(len(example['input_ids']) == len(example['weights']))
    
    print("Weights verified successfully")
