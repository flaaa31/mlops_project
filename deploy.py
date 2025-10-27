import os
from huggingface_hub import HfApi

def deploy_to_hub():
    """
    Load local model to Hugging Face Hub.
    """
    
    try:
        # Retrieving credentials from environmental variables
        hf_token = os.environ["HF_TOKEN"]
        hf_username = os.environ["HF_USERNAME"]
    except KeyError as e:
        print(f"Error: Variable {e} not found.")
        print("Deploy failed: missing credentials HF_TOKEN o HF_USERNAME.")
        exit(1) # error code

    # Defining repo and local folder
    repo_id = f"{hf_username}/sentiment_model_for_hf"
    local_folder_path = "sentiment_model_local"
    
    print(f"Deploying on Hugging Face Hub...")
    print(f"Loading folder: '{local_folder_path}'")
    print(f"Destination repository: '{repo_id}'")

    # API Initialization
    api = HfApi() # implicit search of hf_token
    
    # Loading folder
    api.upload_folder(
        folder_path=local_folder_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Deploy new fine-tuned model from MLOps pipeline"
    )
    
    print("Upload completed.")

if __name__ == "__main__":
    deploy_to_hub()