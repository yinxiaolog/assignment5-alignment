import os
from pathlib import Path
import json
from typing import Callable, List, Dict
import hashlib

from vllm import LLM, SamplingParams
from loguru import logger as log

from drgrpo_grader import r1_zero_reward_fn


def main() -> None:
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: You have two circles, one with radius $r$ and the other with radius $R$. You wish for the difference in the areas of these two circles to be less than or equal to 5$\\pi$. If $r+R=10$, what is the maximum difference in the lengths of the radii? Assistant: <think>",
    ]

    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
    )

    llm = LLM(
        model="Qwen/Qwen2.5-Math-1.5B", gpu_memory_utilization=0.3, enforce_eager=True
    )

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text}")


def math_to_promts(data_path: str, promt_path: str) -> List[str]:
    math_promts = []
    ground_truth = {}
    with open(promt_path) as f:
        prompt = f.read()

    for dir in Path(data_path).iterdir():
        for file in dir.iterdir():
            with open(file) as f:
                qa = json.load(f)
                problem = qa["problem"]
                solution = qa["solution"]
                math_promts.append(prompt.format(question=problem))
                ground_truth[hashlib.sha256(math_promts[-1].encode()).hexdigest()] = (
                    solution
                )
    return math_promts, ground_truth


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
    ground_truth: dict[str, str],
    output_file_name: str = "output.json",
) -> None:
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    res = []
    format_answer = 0
    format = 0
    answer = 0
    for output in outputs:
        prompt = output.prompt
        key = hashlib.sha256(prompt.encode()).hexdigest()
        assert key in ground_truth
        solution = ground_truth[key]
        generated_text = output.outputs[0].text
        reward = reward_fn(generated_text, solution)
        format_reward = reward["format_reward"]
        answer_reward = reward["answer_reward"]
        if format_reward == 1.0 and answer_reward == 1.0:
            format_answer += 1.0
        elif format_reward == 1.0 and answer_reward == 0.0:
            format += 1.0
        elif format_reward == 0.0 and answer_reward == 0.0:
            answer += 1.0
        else:
            raise ValueError("unknown reward")
        res.append(
            {
                "prompt": prompt,
                "solution": solution,
                "generated_text": generated_text,
                "reward": reward,
            }
        )
    res.insert(
        0,
        {
            "format_answer_reward": format_answer,
            "format_reward": format,
            "formath_answer_error": answer,
            "all": len(prompts),
        },
    )
    with open(output_file_name, "w") as f:
        json.dump(res, f, indent=4)


if __name__ == "__main__":
    prompts, gt = math_to_promts(
        "data/MATH/test", "cs336_alignment/prompts/r1_zero.prompt"
    )
    llm = LLM(
        model="Qwen/Qwen2.5-Math-1.5B", gpu_memory_utilization=0.3, enforce_eager=True
    )
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
    )
    evaluate_vllm(llm, r1_zero_reward_fn, prompts, sampling_params, gt)
