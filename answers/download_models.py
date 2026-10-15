import os, time, winreg
os.environ.setdefault("HF_TOKEN", winreg.QueryValueEx(winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment"), "HF_TOKEN")[0])
from huggingface_hub import snapshot_download
for mid, rev in [("meta-llama/Llama-3.2-3B-Instruct", "0cb88a4f764b7a12671c53f0838cd831a0843b95"),
                 ("google/gemma-2-2b-it", "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8")]:
    for attempt in range(20):
        try:
            snapshot_download(mid, revision=rev, allow_patterns=["*.json", "*.safetensors", "tokenizer*"]); print("downloaded", mid, flush=True); break
        except Exception as e:
            print("retry", mid, attempt, type(e).__name__, flush=True); time.sleep(10)
