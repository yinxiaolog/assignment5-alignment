import json
import os
from typing import override
from pathlib import Path
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from loguru import logger as log

# vLLM 0.14.1 defaults to `fork`, which breaks CUDA once the parent process
# has touched torch. Force `spawn` before vLLM creates worker processes.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
from vllm import LLM
from unittest.mock import patch


def tokenize_prompt_and_output(
    prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizerBase
) -> dict[str, Tensor]:
    input_ids = []
    labels = []
    response_mask = []
    ids = []
    pad_token_id = tokenizer.pad_token_id
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer(prompt)["input_ids"]
        output_ids = tokenizer(output)["input_ids"]
        ids.append(prompt_ids + output_ids)
        response_mask.append([False] * (len(prompt_ids) - 1) + [True] * len(output_ids))

    max_len = max([len(input_id) for input_id in ids])
    for input_id, mask in zip(ids, response_mask):
        input_id = input_id + [pad_token_id] * (max_len - len(input_id))
        input_ids.append(input_id[0:-1])
        labels.append(input_id[1:])
        mask.extend([False] * (max_len - 1 - len(mask)))

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "response_mask": torch.tensor(response_mask, dtype=torch.bool),
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    x = F.softmax(logits, dim=-1)
    x = x * torch.log(x)
    x = x.sum(dim=-1)
    x *= -1
    return x


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids)["logits"]
    log_probs = F.softmax(logits, dim=-1)
    log_probs = (
        torch.log(log_probs).gather(dim=2, index=labels.unsqueeze(-1)).squeeze(-1)
    )
    token_entropy = compute_entropy(logits) if return_token_entropy else None
    return {"log_probs": log_probs, "token_entropy": token_entropy}


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    return tensor.masked_fill(~mask, 0).sum(dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = (
        -masked_normalize(policy_log_probs, response_mask, normalize_constant, dim=-1)
    ).mean(dim=-1) / gradient_accumulation_steps
    loss.backward()
    return loss, {"loss": loss.item}


def log_generations():
    pass


def init_vllm(
    model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85
):
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    with world_size_patch:
        if device != "cuda":
            log.warning(
                "vLLM 0.14.1 LLM() does not take `device`; requested device={} "
                "will be ignored and vLLM will use platform default.",
                device,
            )
        return LLM(
            model=model_id,
            seed=seed,
            dtype="bfloat16",
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


class MATHDataet(Dataset):
    def __init__(
        self, prompt_path: str, data_path: str, tokenizer: PreTrainedTokenizerBase
    ):
        super().__init__()
        with open(prompt_path, "r") as f:
            prompt_template = f.read()
        path = Path(data_path)
        prompts = []
        solutions = []
        for file in path.rglob("*"):
            if file.is_file():
                with file.open("r") as f:
                    qa = json.load(f)
                prompts.append(prompt_template.format(question=qa["problem"]))
                solutions.append(qa["solution"])
        tokenized = tokenize_prompt_and_output(prompts, solutions, tokenizer)
        self.input_ids = tokenized["input_ids"]
        self.labels = tokenized["labels"]
        self.response_mask = tokenized["response_mask"]

    @override
    def __len__(self) -> int:
        return len(self.input_ids)

    @override
    def __getitem__(self, index):
        return self.input_ids[index], self.labels[index], self.response_mask[index]


def train():
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dataset = MATHDataet(
        prompt_path="cs336_alignment/prompts/r1_zero.prompt",
        data_path="./data/MATH/train",
        tokenizer=tokenizer,
    )
    dataloader = DataLoader(dataset, batch_size=16)
    llm = AutoModelForCausalLM.from_pretrained(model_id)
    vllm_llm = init_vllm(model_id, device="cuda", seed=42, gpu_memory_utilization=0.5)
    optimizer = AdamW(llm.parameters(), lr=1e-5)

    for epoch in range(2):
        for input_ids, labels, response_mask in dataloader:
            policy_log_probs = get_response_log_probs(llm, input_ids, labels)["log_probs"]
            loss, _ = sft_microbatch_train_step(policy_log_probs, response_mask, 4)
            log.info(loss.item())
            log.info("=================")
            exit(0)
    log.info(dataset[100])


if __name__ == "__main__":
    train()
