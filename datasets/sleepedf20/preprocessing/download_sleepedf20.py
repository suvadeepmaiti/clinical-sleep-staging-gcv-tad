"""Download and verify the raw Sleep-EDF-20 subset from PhysioNet."""
import sys
import concurrent.futures
import hashlib
from pathlib import Path
import re
import time
import urllib.request

BASE = "https://physionet.org/files/sleep-edfx/1.0.0/"
DATA_BASE = "https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0/"
sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "src" / "repo_paths.py").exists()) / "src"))
import repo_paths  # noqa: E402  (data/results locations; puts the cohort modelling folders on sys.path)
OUT = repo_paths.RAW["sleepedf20"].parent  # files are named sleep-cassette/<recording>.edf


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(name, expected):
    target = OUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256(target) == expected:
        return f"Verified existing: {name}"
    partial = target.with_name(target.name + ".part")
    if partial.exists() and sha256(partial) == expected:
        partial.replace(target)
        return f"Verified resumed: {name}"
    for attempt in range(5):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            request = urllib.request.Request(DATA_BASE + name, headers=headers)
            with urllib.request.urlopen(request, timeout=90) as response:
                append = offset and response.status == 206
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Unexpected range response")
                with partial.open("ab" if append else "wb") as destination:
                    while chunk := response.read(1024 * 1024):
                        destination.write(chunk)
            if sha256(partial) != expected:
                partial.unlink()
                raise ValueError(f"SHA-256 mismatch for {name}")
            partial.replace(target)
            return f"Downloaded and verified: {name}"
        except Exception as error:
            if getattr(error, "code", None) == 416 and partial.exists():
                if sha256(partial) == expected:
                    partial.replace(target)
                    return f"Verified resumed: {name}"
                partial.unlink()
            if attempt == 4:
                raise
            print(f"Retry {attempt + 1}: {name}: {error}", flush=True)
            time.sleep(2 ** attempt)


def main():
    with urllib.request.urlopen(BASE + "SHA256SUMS.txt", timeout=90) as response:
        manifest = response.read().decode("utf-8")
    entries = {}
    for line in manifest.splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*").removeprefix("./")
        if re.fullmatch(r"sleep-cassette/SC4(?:0[0-9]|1[0-9])[12][A-Z0-9]{2}-(?:PSG|Hypnogram)\.edf", name):
            entries[name] = digest
    psg = {Path(n).name[:6] for n in entries if n.endswith("-PSG.edf")}
    hyp = {Path(n).name[:6] for n in entries if n.endswith("-Hypnogram.edf")}
    if len(entries) != 78 or len(psg) != 39 or psg != hyp:
        raise RuntimeError(f"Unexpected subset: {len(entries)} files, {len(psg)} PSGs, {len(hyp)} annotations")
    (OUT / "SHA256SUMS-sleepedf20.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(entries.items()))
    )
    print("Sleep-EDF-20: 20 subjects, 39 nights, 78 EDF files", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, name, digest) for name, digest in sorted(entries.items())]
        errors = []
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                print(f"[{count}/78] {future.result()}", flush=True)
            except Exception as error:
                errors.append(str(error))
                print(f"[{count}/78] Failed: {error}", flush=True)
    if errors:
        raise RuntimeError("Download incomplete; rerun to resume: " + "; ".join(errors))
    total = sum((OUT / name).stat().st_size for name in entries)
    print(f"Complete: all 78 EDF files verified; {total:,} bytes", flush=True)


if __name__ == "__main__":
    main()
