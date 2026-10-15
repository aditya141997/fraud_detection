"""Register the bundled model as `champion` if the registry has none (first start of a fresh container).

    python -m src.serving.seed
"""
import os
from pathlib import Path

from src.serving.registry import alias_version, register

MODEL_DIR = Path(os.environ.get("SEED_MODEL_DIR", "deploy/model"))


def main():
    if alias_version("champion") is not None:
        print("champion already registered; nothing to seed")
        return
    v = register(str(MODEL_DIR / "model_v1.txt"), str(MODEL_DIR / "model_v1.json"), "champion", "tuned_v1")
    print(f"seeded champion -> version {v}")


if __name__ == "__main__":
    main()
