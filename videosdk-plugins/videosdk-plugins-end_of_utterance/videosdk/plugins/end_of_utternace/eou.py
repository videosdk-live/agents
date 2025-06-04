from .model import HG_MODEL, MODEL_REVISIONS, ONNX_FILENAME, EOUModelType

def _download_from_hf_hub(repo_id, filename, **kwargs):
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
    return local_path