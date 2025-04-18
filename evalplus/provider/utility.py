from concurrent.futures import ThreadPoolExecutor
from typing import List

EOS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
]


def extra_eos_for_direct_completion(dataset) -> List[str]:
    if dataset.lower() == "humaneval":
        return ["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert "]
    elif dataset.lower() == "mbpp":
        return ['\n"""', "\nassert"]
    raise ValueError(f"Unknown dataset: {dataset}")


# some random words which serves as the splitter
_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"


def make_raw_chat_prompt(
    task_prompt: str,
    instruction_prefix: str,
    response_prefix: str,
    tokenizer,
    r1_system_prompt: bool = False,
) -> str:
    if r1_system_prompt:
        # directly return prompt if it does not have a tokenizer.chat_template
        if tokenizer.chat_template is None:
            return task_prompt

        assert instruction_prefix is not None, "Instruction prefix is required!"
        assert response_prefix is not None, "Response prefix is required!"

        task_prompt = f"""\
{instruction_prefix}
```
{task_prompt.strip()}
```
"""
        response = f"""\
{response_prefix}
```python
{_MAGIC_SPLITTER_}
```
"""
        # task_prompt = tokenizer.apply_chat_template(
        #     [
        #         {"role": "user", "content": task_prompt},
        #         {"role": "assistant", "content": response},
        #     ],
        #     tokenize=False,
        # ).split(_MAGIC_SPLITTER_)[0]
        task_prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": """\
    A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
    """},
                {"role": "user", "content": task_prompt},
                # {"role": "assistant", "content": response},
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        return task_prompt
    else:
        # directly return prompt if it does not have a tokenizer.chat_template
        if tokenizer.chat_template is None:
            return task_prompt

        assert instruction_prefix is not None, "Instruction prefix is required!"
        assert response_prefix is not None, "Response prefix is required!"

        task_prompt = f"""\
{instruction_prefix}
```
{task_prompt.strip()}
```
    """
        response = f"""\
{response_prefix}
```python
{_MAGIC_SPLITTER_}
```
"""
        task_prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": task_prompt},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
        ).split(_MAGIC_SPLITTER_)[0]
        return task_prompt


def concurrent_call(n, callback, /, *args, **kwargs):
    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = [executor.submit(callback, *args, **kwargs) for _ in range(n)]
        return [future.result() for future in futures]
