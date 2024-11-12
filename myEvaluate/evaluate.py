from evalplus.codegen import run_codegen as create_inference
from evalplus.sanitize import script as sanitize
from evalplus.evaluate import evaluate as evaluate_model
from fire import Fire
# tuple in the form of (original model path, original model name, custom model path, custom model name)
MODELS = {
    "codellama_instruct_7b": "codellama/CodeLlama-7b-Instruct-hf",
    "mistral_instruct_v3_7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama3_instruct_8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "nxcode_cq_orpo_7b": "NTQAI/Nxcode-CQ-7B-orpo",
    "code_qwen_v1.5_7b": "Qwen/CodeQwen1.5-7B-Chat",
    "qwen_coder_2.5_7b": "Qwen/Qwen2.5-Coder-7B-Instruct"
}


def evaluate(model_name: str, n_samples:int = 1):
    model_path = MODELS[model_name]
    model_path_2 = model_path.replace("/", "--")
    
    for dataset in ["humaneval", "mbpp"]:
        if n_samples == 1:
            create_inference(
                model=model_path,
                greedy=True,
                root=f"inferenced_output",
                jsonl_fmt=True,
                dataset=dataset,
                backend="vllm",
            )
            sanitize(samples=f"inferenced_output/{dataset}/{model_path_2}_vllm_temp_0.0.jsonl")
        else:
            create_inference(
                model=model_path,
                greedy=False,
                root=f"inferenced_output",
                jsonl_fmt=True,
                dataset=dataset,
                backend="vllm",
                n_samples=n_samples,
                temperature=1.0,
            )
            sanitize(samples=f"inferenced_output/{dataset}/{model_path_2}_vllm_temp_1.0.jsonl")
        
    for dataset in ["humaneval", "mbpp"]:
        print(f"-----------{dataset}-{model_name}-------------")
        ram = model_path.replace("/", "--")
        if n_samples == 1:
            evaluate_model(dataset=dataset, samples=f"inferenced_output/{dataset}/{ram}_vllm_temp_0.0-sanitized.jsonl")
        else:
            evaluate_model(dataset=dataset, samples=f"inferenced_output/{dataset}/{ram}_vllm_temp_1.0-sanitized.jsonl")

        
if __name__ == "__main__":
    Fire(evaluate)