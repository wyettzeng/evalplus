from evalplus.codegen import run_codegen as create_inference
from evalplus.sanitize import script as sanitize
from evalplus.evaluate import evaluate as evaluate_model
from fire import Fire

def evaluate(model_path: str):
    print(f"Starting inference for {model_path}")
    model_path_2 = model_path.replace("/", "--")
    if model_path_2.startswith("--"):
        model_path_2 = model_path_2[2:]
    
    for dataset in ["humaneval", "mbpp"]:
        create_inference(
            model=model_path,
            greedy=True,
            root=f"inferenced_output/rl_results/",
            jsonl_fmt=True,
            dataset=dataset,
            backend="vllm",
            r1_system_prompt=True
        )
        sanitize(samples=f"inferenced_output/rl_results/{dataset}/{model_path_2}_vllm_temp_0.0.jsonl")

        
    for dataset in ["humaneval", "mbpp"]:
        evaluate_model(dataset=dataset, samples=f"inferenced_output/rl_results/{dataset}/{model_path_2}_vllm_temp_0.0-sanitized.jsonl")

        
if __name__ == "__main__":
    Fire(evaluate)