import requests
import os
from urllib.parse import urljoin

def download_model_files_to_directory(
    base_cdn_url: str,
    file_names: list[str],
    local_save_directory: str = "downloaded_model_assets",
    overwrite_existing: bool = False
) -> list[str]:
    
    os.makedirs(local_save_directory, exist_ok=True)
    handled_paths = []

    for file_name in file_names:
        full_cdn_url = urljoin(base_cdn_url, file_name)
        local_file_path = os.path.join(local_save_directory, file_name)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        if os.path.exists(local_file_path) and not overwrite_existing:
            handled_paths.append(local_file_path)
            continue

        try:
            response = requests.get(full_cdn_url, stream=True)
            response.raise_for_status()

            with open(local_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            handled_paths.append(local_file_path)

        except Exception as e:
            print(f"Error downloading '{file_name}': {e}")

    return handled_paths