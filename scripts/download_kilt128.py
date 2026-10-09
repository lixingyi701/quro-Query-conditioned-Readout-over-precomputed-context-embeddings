"""Download kilt-128 shards via direct GET (args: shard numbers; default 1-42).

hf_hub_download's HEAD requests and the redirect to cas-bridge.xethub.hf.co both
picked up a dead HTTP(S)_PROXY from the tmux environment, so this uses a session
with trust_env=False and retries each shard until it succeeds.
"""
import os, time, requests, hashlib
from pathlib import Path

MIRROR = "https://hf-mirror.com"
BLOB_DIR = Path("/data02/quro/hf-cache/datasets--dmrau--kilt-128/blobs")
BLOB_DIR.mkdir(parents=True, exist_ok=True)
LOG = Path(os.devnull)

# Map shard filenames to their expected blob hashes (we'll compute on the fly)
TOTAL = 43

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def download_shard(i):
    shard = f"data/train-{i:05d}-of-00043.parquet"
    url = f"{MIRROR}/datasets/dmrau/kilt-128/resolve/main/{shard}"
    # Use a temp filename, then rename
    tmp = BLOB_DIR / f"shard_{i:05d}.tmp"
    final_prefix = BLOB_DIR / f"shard_{i:05d}.parquet"

    # Check if already downloaded
    if final_prefix.exists():
        log(f"shard {i}/42 already exists, skipping")
        return str(final_prefix)

    t0 = time.time()
    try:
        session = requests.Session()
        session.trust_env = False  # ignore system proxy (127.0.0.1:7897 is not running)
        r = session.get(url, stream=True, timeout=(30, 120))
        r.raise_for_status()
        total = 0
        hasher = hashlib.sha256()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)
                hasher.update(chunk)
                total += len(chunk)
        sz = total
        dt = time.time() - t0
        # Rename to final
        tmp.rename(final_prefix)
        log(f"shard {i}/42: {sz/1e6:.0f}MB in {dt:.0f}s = {sz/dt/1e6:.1f} MB/s  sha256={hasher.hexdigest()[:16]}...")
        return str(final_prefix)
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        log(f"shard {i}/42 FAILED: {e}")
        return None

if __name__ == "__main__":
    import sys
    todo = [int(x) for x in sys.argv[1:]] or list(range(1, TOTAL))
    log(f"Starting download of kilt-128 shards {todo}")
    for i in todo:
        attempt = 0
        while download_shard(i) is None:
            attempt += 1
            log(f"  retry {attempt} for shard {i} in {min(60, 5 * attempt)}s")
            time.sleep(min(60, 5 * attempt))
    log(f"done: shards {todo}")
