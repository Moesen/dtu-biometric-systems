from pathlib import Path

SRC_ROOT = Path(__file__).parent.resolve()
REPO_ROOT = SRC_ROOT.parent.resolve()

if __name__ == "__main__":
    print(SRC_ROOT, REPO_ROOT)
