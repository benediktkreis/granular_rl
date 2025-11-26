import os
from typing import Optional
from huggingface_hub import HfApi, upload_folder, upload_file, snapshot_download
from requests.exceptions import HTTPError
from wasabi import Printer
from pathlib import Path

msg = Printer()

def check_repo(repo_id, token):
    repo_url = HfApi().create_repo(
        repo_id=repo_id,
        token=token,
        private=True,
        exist_ok=True,
    )
    return repo_url

def load_online_model(config_file_name, log_dir_name):
    online_model = None
    try:
        pull_from_hub(
            repo_id="sand-gym/"+config_file_name,
            filename=log_dir_name,
        )
        online_model_exists = True
    except HTTPError as http_err:
        if http_err.response.status_code == 404:
            online_model_exists = False
        else:
            print(f"A HTTP error occurred while trying to load the online model: {http_err}")
            raise ValueError
    except Exception as general_err:
        print(f"Exception occurred while trying to load the online model: {general_err}")
        raise ValueError
    
    return online_model_exists, online_model

def pull_from_hub(repo_id: str, filename: str):
    msg.info(f"Pulling repo {repo_id} from the Hugging Face Hub")
    filename_path = os.path.abspath(filename)
    snapshot_download(
        repo_id=repo_id,
        local_dir=filename_path
        )
    msg.good(
        f"The repo has been pulled successfully from the Hub, you can find it here: {filename_path}"
    )

def push_to_hub(
    repo_id: str,
    filename: str,
    commit_message: str,
    token: Optional[str] = None,
    delete_old_best_models: bool = False,
    delete_old_final_models: bool = False,
    skip_buffer_upload: bool = False,
):
    success = False
    repo_url = check_repo(repo_id, token)

    # Add the model
    filename_path = os.path.abspath(filename)

    msg.info(f"Pushing repo {repo_id} to the Hugging Face Hub")
    delete_old_files = []
    if delete_old_best_models:
        delete_old_files.append("best_model*")
    if delete_old_final_models:
        delete_old_files.append("final_model*")
    
    ignored_files = []
    if skip_buffer_upload:
        ignored_files.append("*replay_buffer*")

    try:
        repo_url = upload_folder(
            repo_id=repo_id,
            folder_path=filename_path,
            path_in_repo="",
            commit_message=commit_message,
            token=token,
            delete_patterns=delete_old_files,
            ignore_patterns=ignored_files,
        )
        msg.good(
            f"Your model has been uploaded to the Hub, you can find it here: {repo_url}"
        )
        success = True
    except Exception as err:
        msg.info(
            f"Error occurred while uploading model {filename}. The error is {err=} of {type(err)=}."
        )
        success = False
    return success

def push_file_to_hub(
    repo_id: str,
    filename: str,
    commit_message: str,
    token: Optional[str] = None,
):
    success = False
    repo_url = check_repo(repo_id, token)

    # Add the model
    filename_path = os.path.abspath(filename)

    msg.info(f"Pushing repo {repo_id} to the Hugging Face Hub")
    try:
        repo_url = upload_file(
            repo_id=repo_id,
            path_or_fileobj=filename_path,
            path_in_repo=Path(filename_path).name,
            commit_message=commit_message,
            token=token,
        )
        msg.good(
            f"Your model has been uploaded to the Hub, you can find it here: {repo_url}"
        )
        success = True
    except Exception as err:
        msg.info(
            f"Error occurred while uploading model {filename}. The error is {err=} of {type(err)=}."
        )
        success = False
    
    return success