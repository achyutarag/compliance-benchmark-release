import os, time, winreg, json
os.environ.setdefault("HF_TOKEN", winreg.QueryValueEx(winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment"), "HF_TOKEN")[0])
from huggingface_hub import snapshot_download, model_info
revs = {}
for mid in ("lytang/MiniCheck-RoBERTa-Large", "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"):
    rev = model_info(mid).sha
    pats = ["*.json", "*.txt", "*.model", "model.safetensors"] if "Moritz" in mid else ["*.json", "*.txt", "pytorch_model.bin"]
    for a in range(20):
        try:
            snapshot_download(mid, revision=rev, allow_patterns=pats); print("downloaded", mid, rev, flush=True); break
        except Exception as e:
            print("retry", mid, a, type(e).__name__, flush=True); time.sleep(10)
    revs[mid] = rev
json.dump(revs, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "CHECKER_REVISIONS.json"), "w"), indent=2)
