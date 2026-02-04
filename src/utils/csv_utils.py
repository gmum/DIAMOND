import csv
import os


def read_prompts_file(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def read_prompt_seed_csv(path: str):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            if "prompt" not in row or "seed" not in row:
                raise ValueError("CSV must have 'prompt' and 'seed' columns")
            rows.append((row["prompt"], int(row["seed"])))
        return rows


def append_prompt_seed(path: str, prompt: str, seed: int):
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([prompt, seed])


def ensure_csv_header(path: str):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["prompt", "seed"])
        return

    with open(path, "r", encoding="utf-8") as handle:
        first_line = handle.readline()
    if first_line.strip():
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["prompt", "seed"])
