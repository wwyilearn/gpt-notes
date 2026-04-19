
import h5py
from pathlib import Path
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np


# =========================
# 你只需要改这几个参数
# =========================

H5_PATH = "/Users/wenya/Documents/毕设/MPIIFaceGaze.h5"  # 改成你自己的数据集路径
OUT_DIR = "./paper_figures"

# 图1：数据集样本展示图
# 建议 9 张：3行×3列
DATASET_SAMPLES: List[Tuple[str, int]] = [
    ("p00", 0),
    ("p01", 120),
    ("p02", 240),
    ("p03", 360),
    ("p04", 480),
    ("p05", 600),
    ("p06", 720),
    ("p07", 840),
    ("p08", 960),
]

# 图2：任务难点展示图
# 建议 6 张：2行×3列
# 最好人工挑选“长得像但 gaze 不同”或“头姿态不同但 gaze 接近”的图
DIFFICULTY_SAMPLES: List[Tuple[str, int]] = [
    ("p00", 10),
    ("p00", 11),
    ("p01", 50),
    ("p01", 51),
    ("p02", 90),
    ("p02", 91),
]

# 是否在子图下显示 pose / gaze
SHOW_LABELS = True


def ensure_out_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _get_person_group(f: h5py.File, person_id: str):
    if person_id in f:
        return f[person_id]
    raise KeyError(f"Cannot find person group: {person_id}")


def _resolve_sample(group, idx: int) -> Dict[str, np.ndarray]:
    result = {}

    if "image" in group and f"{idx:04d}" in group["image"]:
        result["image"] = group["image"][f"{idx:04d}"][()]
    elif "images" in group:
        result["image"] = group["images"][idx]
    else:
        raise KeyError(f"Cannot find image for index {idx}")

    if "pose" in group and f"{idx:04d}" in group["pose"]:
        result["pose"] = group["pose"][f"{idx:04d}"][()]
    elif "poses" in group:
        result["pose"] = group["poses"][idx]
    else:
        result["pose"] = None

    if "gaze" in group and f"{idx:04d}" in group["gaze"]:
        result["gaze"] = group["gaze"][f"{idx:04d}"][()]
    elif "gazes" in group:
        result["gaze"] = group["gazes"][idx]
    else:
        raise KeyError(f"Cannot find gaze for index {idx}")

    return result


def _to_display_image(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)

    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 1:
        img = np.concatenate([img] * 3, axis=-1)

    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img, 0, 1)

    return img


def _format_angle_pair(arr) -> str:
    if arr is None:
        return "None"
    arr = np.asarray(arr).reshape(-1)
    if len(arr) >= 2:
        return f"[{arr[0]:.3f}, {arr[1]:.3f}]"
    return str(arr.tolist())


def load_samples(h5_path: str, sample_list: List[Tuple[str, int]]):
    loaded = []
    with h5py.File(h5_path, "r") as f:
        for person_id, idx in sample_list:
            person_group = _get_person_group(f, person_id)
            sample = _resolve_sample(person_group, idx)
            loaded.append({
                "person_id": person_id,
                "idx": idx,
                "image": _to_display_image(sample["image"]),
                "pose": sample["pose"],
                "gaze": sample["gaze"],
            })
    return loaded


def plot_grid(samples, nrows: int, ncols: int, title: str, out_path: Path):
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.6 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    for ax in axes.ravel():
        ax.axis("off")

    for ax, s in zip(axes.ravel(), samples):
        ax.imshow(s["image"])
        ax.set_title(f"{s['person_id']} / #{s['idx']}", fontsize=10)
        if SHOW_LABELS:
            caption = f"gaze={_format_angle_pair(s['gaze'])}"
            if s["pose"] is not None:
                caption += f"\npose={_format_angle_pair(s['pose'])}"
            ax.text(
                0.5, -0.08, caption,
                transform=ax.transAxes,
                ha="center", va="top", fontsize=8
            )
        ax.axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    out_dir = ensure_out_dir(OUT_DIR)

    dataset_samples = load_samples(H5_PATH, DATASET_SAMPLES)
    plot_grid(
        dataset_samples,
        nrows=3,
        ncols=3,
        title="MPIIFaceGaze Dataset Samples",
        out_path=out_dir / "dataset_samples_grid.png",
    )

    difficulty_samples = load_samples(H5_PATH, DIFFICULTY_SAMPLES)
    plot_grid(
        difficulty_samples,
        nrows=2,
        ncols=3,
        title="Representative Challenging Samples",
        out_path=out_dir / "difficulty_samples_grid.png",
    )


if __name__ == "__main__":
    main()
