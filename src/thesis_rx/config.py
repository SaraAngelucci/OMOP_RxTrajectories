from pathlib import Path
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_output_dir(cfg: dict) -> Path:
    outdir = Path(cfg["project"]["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

REQUIRED_KEYS = {
    "project": ["name", "output_dir"],
    "analysis": [
        "washout_days", "followup_months", "exposure_gap_days",
        "maintenance_min_total_days", "early_discontinuation_days",
        "restart_window_days", "switch_window_days",
        "polypharmacy_threshold", "turnover_low", "turnover_high"
    ],
    "clustering": ["k_grid", "seed"],
    "run": ["run_exposure_sensitivity", "save_top_ingredients_per_cluster"]
}


def validate_config(cfg: dict) -> None:
    """Raise a clear error if any required config key is missing."""
    missing = []
    for section, keys in REQUIRED_KEYS.items():
        if section not in cfg:
            missing.append(f"[{section}]  (entire section missing)")
            continue
        for k in keys:
            if k not in cfg[section]:
                missing.append(f"[{section}].{k}")
    if missing:
        raise ValueError(
            "Missing required config keys:\n  " + "\n  ".join(missing)
        )
