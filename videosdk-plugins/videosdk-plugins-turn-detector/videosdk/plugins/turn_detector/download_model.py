import os
import requests
from tqdm import tqdm

def download_model_files_to_directory(
    base_cdn_url: str,
    file_names: list[str],
    local_save_directory: str,
    overwrite_existing: bool = False,
):
    os.makedirs(local_save_directory, exist_ok=True)

    for filename in file_names:
        local_path = os.path.join(local_save_directory, filename)
        if os.path.exists(local_path) and not overwrite_existing:
            print(f"[✓] Skipping {filename} (already exists)")
            continue

        url = f"{base_cdn_url.rstrip('/')}/{filename}"
        print(f"Downloading: {url}")

        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0))
                block_size = 8192

                with open(local_path, "wb") as f, tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"{filename}",
                ) as bar:
                    for chunk in r.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))

            print(f"[✓] Downloaded {filename} → {local_path}")

        except Exception as e:
            print(f"[!] Failed to download {filename}: {e}")