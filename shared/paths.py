
from pathlib import Path

# Ye file shared/paths.py hai, isliye project root ek level upar hai
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DOCX_PATH = DATA_DIR / "novatech_ai_governance_policy.docx"
GOLDEN_DATASET_PATH = DATA_DIR / "golden_dataset.json"

CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "novatech_ai_policy"
