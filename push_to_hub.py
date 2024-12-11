from huggingface_hub import HfApi

repos = [
    "toxic-spans-lyeonii-bert-tiny",
    "toxic-spans-lyeonii-bert-small",
    "toxic-spans-lyeonii-bert-medium",
    "toxic-spans-google-bert-bert-base-uncased",
    "toxic-spans-google-bert-bert-large-uncased"
]

folders = [
    "bert\\lyeonii_bert-tiny",
    "bert\\lyeonii_bert-small",
    "bert\\lyeonii_bert-medium",
    "bert\\google-bert_bert-base-uncased",
    "bert\\google-bert_bert-large-uncased",

]

api = HfApi()
def upload(repo, folder):
    repo_id = f"charleyisballer/{repo}"
    files = {
        f"C:\\Users\\charley\\Desktop\\ml-stuf\\toxic_spans_results\\{folder}\\best_model\\tensorflow\\tf_model.h5": "tf_model.h5",
        f"C:\\Users\\charley\\Desktop\\ml-stuf\\toxic_spans_results\\{folder}\\best_model\\flax\\flax_model.msgpack": "flax_model.msgpack",
        f"C:\\Users\\charley\\Desktop\\ml-stuf\\toxic_spans_results\\{folder}\\best_model\\pytorch\\model.safetensors": "model.safetensors",
        f"C:\\Users\\charley\\Desktop\\ml-stuf\\toxic_spans_results\\{folder}\\best_model\\pytorch_model.bin": "pytorch_model.bin",
        f"C:\\Users\\charley\\Desktop\\ml-stuf\\toxic_spans_results\\{folder}\\best_model\\hyperparameters.json": "hyperparameters.json",
        f"C:\\Users\\charley\\Desktop\\ml-stuf\\toxic_spans_results\\{folder}\\best_model\\eval_results.json": "eval_results.json",

    }

    for local_path, repo_path in files.items():
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="model"
        )

for repo, folder in zip(repos, folders):
    upload(repo, folder)