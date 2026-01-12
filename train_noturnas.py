"""
Treino YOLOv8 segmentacao para o dataset Noturnas usando configs em configs/noturnas_args.yaml.
Execute: python train_noturnas.py
"""

import csv
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


ROOT = Path(__file__).resolve().parent
CONFIGS_ROOT = ROOT / "configs"
DATASETS_ROOT = ROOT / "datasets"
DATASET_NAME = "Noturnas"
DEFAULT_RUN_DIR = "train"
DEFAULT_ARGS_NAME = "noturnas_args.yaml"
DEFAULT_RUN_NAME = "train_noturnas_api"


def dataset_base_dirs(dataset_name: str):
    return [
        DATASETS_ROOT / dataset_name,
        ROOT / "dataset" / dataset_name,
        ROOT / dataset_name,
    ]


def resolve_dataset_dir(dataset_name: str) -> Path:
    for candidate in dataset_base_dirs(dataset_name):
        if (candidate / "data.yaml").exists():
            return candidate
    return DATASETS_ROOT / dataset_name


def resolve_args_path(dataset_name: str, run_dir: str, args_name: str) -> Path:
    config_path = CONFIGS_ROOT / args_name
    if config_path.exists():
        return config_path
    for candidate in dataset_base_dirs(dataset_name):
        args_path = candidate / "runs" / "segment" / run_dir / args_name
        if args_path.exists():
            return args_path
    for candidate in dataset_base_dirs(dataset_name):
        args_path = candidate / "runs" / "segment" / run_dir / "args.yaml"
        if args_path.exists():
            return args_path
    return config_path


DEFAULT_ARGS_PATH = resolve_args_path(DATASET_NAME, DEFAULT_RUN_DIR, DEFAULT_ARGS_NAME)
DEFAULT_DATASET_DIR = resolve_dataset_dir(DATASET_NAME)
DEFAULT_DATA_PATH = DEFAULT_DATASET_DIR / "data.yaml"
DEFAULT_PROJECT_DIR = DEFAULT_DATASET_DIR / "runs" / "segment"

DROP_TRAIN_KEYS = {"model", "mode", "task", "save_dir"}

LOSS_COLUMNS = [
    "train/box_loss",
    "train/seg_loss",
    "train/cls_loss",
    "val/box_loss",
    "val/seg_loss",
    "val/cls_loss",
]

METRIC_GROUPS = [
    ("metrics/seg_p", "metrics/seg_r", "metrics/seg_mAP50", "metrics/seg_mAP50-95"),
    ("metrics/precision(M)", "metrics/recall(M)", "metrics/mAP50(M)", "metrics/mAP50-95(M)"),
    ("metrics/precision", "metrics/recall", "metrics/mAP50", "metrics/mAP50-95"),
]

F1_KEY = "metrics/f1"


def load_args(args_path: Path) -> dict:
    if not args_path.exists():
        raise FileNotFoundError(f"Arquivo de args nao encontrado: {args_path}")
    with args_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Conteudo invalido em: {args_path}")
    return data


def select_device() -> str:
    return "0" if torch.cuda.is_available() else "cpu"


def build_search_dirs(data_path: Path, args_path: Path, dataset_name: str):
    search_dirs = []
    candidates = [data_path.parent, *args_path.parents, *dataset_base_dirs(dataset_name)]
    for candidate in candidates:
        if candidate not in search_dirs:
            search_dirs.append(candidate)
    return search_dirs


def resolve_model_source(model_value, search_dirs) -> str:
    model_value = model_value or "best.pt"
    candidate = Path(model_value)
    if candidate.is_absolute():
        return str(candidate)
    for base_dir in search_dirs:
        local_candidate = base_dir / model_value
        if local_candidate.exists():
            return str(local_candidate)
    return model_value


def resolve_resume_value(resume_value, search_dirs):
    if not resume_value:
        return resume_value

    resume_path = Path(str(resume_value))
    if resume_path.is_absolute():
        return str(resume_path) if resume_path.exists() else False

    for base_dir in search_dirs:
        local_candidate = base_dir / resume_path
        if local_candidate.exists():
            return str(local_candidate)

    return False


def prepare_train_args(
    base_args: dict,
    data_path: Path,
    args_path: Path,
    project_dir: Path,
    run_name: str,
    device: str,
):
    args = dict(base_args)
    args["data"] = str(data_path.as_posix())
    args["project"] = args.get("project") or str(project_dir.as_posix())
    args["name"] = args.get("name") or run_name
    args["device"] = device
    search_dirs = build_search_dirs(data_path, args_path, DATASET_NAME)
    args["resume"] = resolve_resume_value(args.get("resume"), search_dirs)

    model_source = resolve_model_source(args.get("model"), search_dirs)
    train_args = {k: v for k, v in args.items() if k not in DROP_TRAIN_KEYS}
    return model_source, train_args


def get_save_dir(results, model):
    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        return Path(save_dir)

    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None) if trainer else None
    return Path(save_dir) if save_dir else None


def read_results_csv(results_csv: Path):
    with results_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def pick_metric_columns(available):
    for group in METRIC_GROUPS:
        present = [key for key in group if key in available]
        if present:
            return present
    return []


def find_precision_recall(columns):
    pairs = [
        ("metrics/seg_p", "metrics/seg_r"),
        ("metrics/precision(M)", "metrics/recall(M)"),
        ("metrics/precision", "metrics/recall"),
    ]
    for precision_key, recall_key in pairs:
        if precision_key in columns and recall_key in columns:
            return precision_key, recall_key
    return None, None


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def collect_series(rows, columns):
    series = {key: [] for key in columns}
    epochs = []

    for row in rows:
        epoch_val = row.get("epoch")
        try:
            epochs.append(int(epoch_val))
        except (TypeError, ValueError):
            epochs.append(len(epochs))

        for key in columns:
            series[key].append(parse_float(row.get(key)))

    return epochs, series


def compute_f1(precision_values, recall_values):
    f1 = []
    for p, r in zip(precision_values, recall_values):
        if p != p or r != r or (p + r) == 0:  # NaN check or zero division
            f1.append(float("nan"))
        else:
            f1.append(2 * p * r / (p + r))
    return f1


def main():
    base_args = load_args(DEFAULT_ARGS_PATH)

    device = select_device()
    print(f"Usando device: {device}")

    model_source, train_args = prepare_train_args(
        base_args,
        DEFAULT_DATA_PATH,
        DEFAULT_ARGS_PATH,
        DEFAULT_PROJECT_DIR,
        DEFAULT_RUN_NAME,
        device,
    )

    model = YOLO(model_source)
    results = model.train(**train_args)

    save_dir = get_save_dir(results, model)
    if not save_dir:
        print("Nao foi possivel localizar o diretorio de resultados.")
        return

    results_csv = save_dir / "results.csv"
    plot_path = save_dir / "metrics_plot.png"
    plot_metrics(results_csv, plot_path)


def plot_metrics(results_csv: Path, plot_path: Path) -> None:
    if not results_csv.exists():
        print(f"Nao encontrei {results_csv} para gerar grafico.")
        return
    if plt is None:
        print("matplotlib nao instalado; pulando grafico.")
        return

    rows = read_results_csv(results_csv)
    if not rows:
        print(f"Nenhum dado em {results_csv}.")
        return

    available = set(rows[0].keys())
    loss_columns = [key for key in LOSS_COLUMNS if key in available]
    metric_columns = pick_metric_columns(available)
    columns = loss_columns + metric_columns

    if not columns:
        print("Nao encontrei colunas conhecidas para plotar.")
        return

    epochs, series = collect_series(rows, columns)

    precision_key, recall_key = find_precision_recall(set(columns))
    if precision_key and recall_key:
        series[F1_KEY] = compute_f1(series[precision_key], series[recall_key])
        metric_columns = metric_columns + [F1_KEY]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if loss_columns:
        for key in loss_columns:
            axes[0].plot(epochs, series[key], label=key)
        axes[0].set_title("Losses")
        axes[0].set_xlabel("Epoch")
        axes[0].grid(alpha=0.3)
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "Sem colunas de loss", ha="center", va="center")

    if metric_columns:
        for key in metric_columns:
            axes[1].plot(epochs, series[key], label=key)
        axes[1].set_title("Metricas (P/R/F1/mAP)")
        axes[1].set_xlabel("Epoch")
        axes[1].grid(alpha=0.3)
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "Sem colunas de metricas", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    print(f"Grafico salvo em: {plot_path}")


if __name__ == "__main__":
    main()
