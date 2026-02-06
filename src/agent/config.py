import os
from pathlib import Path

# ===== 项目路径 =====
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "rag_data"

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src 目录的上两级 → AI_Project/src
VECTOR_DB_DIR = PROJECT_ROOT / "src" / "storage/vector_db/trigonometry"

# ===== API Key =====
os.environ.setdefault(
    "ALIYUN_API_KEY",
    "sk-30b0ba857316437087ace218df67aa95"
)
