"""每次程序运行的统一数据目录。"""

from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_OUTPUT_ROOT = PROJECT_ROOT / "data" / "runs"


def create_run_output_directory(
    root: str | Path = RUN_OUTPUT_ROOT,
    now: datetime | None = None,
) -> Path:
    """创建并返回一个不会与其他运行混用的输出目录。"""
    timestamp = datetime.now() if now is None else now
    run_id = timestamp.strftime("run_%Y%m%d_%H%M%S_%f")
    output_dir = Path(root) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir
