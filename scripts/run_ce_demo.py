#!/usr/bin/env python
import sys
import json
import yaml
from pathlib import Path
from datetime import datetime, UTC

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
HISTORY_PATH = ROOT / "ce_history" / "ce_history.json"
REPORTS_DIR = ROOT / "reports"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_history() -> dict:
    if not HISTORY_PATH.exists():
        return {}
    with HISTORY_PATH.open("r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return {}
        return json.loads(content)


def save_history(history: dict) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, sort_keys=True)


def build_key(model_cfg: dict) -> str:
    return f"{model_cfg['model_id']}::{model_cfg['version']}::{model_cfg['evaluation_profile']}"


def has_successful_eval(history: dict, key: str) -> bool:
    entry = history.get(key)
    return bool(entry and entry.get("status") == "pass")


def run_deepeval_stub(model_cfg: dict) -> dict:

    # To simulate different behaviors, you could vary metrics by version.
    version = model_cfg["version"]
    if version.endswith("0"):
        overall_score = 0.80
        min_group_score = 0.78
    else:
        overall_score = 0.72
        min_group_score = 0.68

    metrics = {
        "model_id": model_cfg["model_id"],
        "version": model_cfg["version"],
        "evaluation_profile": model_cfg["evaluation_profile"],
        "overall_score": overall_score,
        "group_scores": {
            "group_a": min_group_score,
            "group_b": min_group_score + 0.02,
        },
        "raw_metrics": {
            "demographic_parity": 0.81,
            "equal_selection_parity": 0.76,
        },
    }
    return metrics


def apply_gatekeeper(model_cfg: dict, metrics: dict) -> tuple[bool, list[str]]:
    gate = model_cfg.get("gate", {})
    min_overall = gate.get("min_overall_score", 0.0)
    min_group = gate.get("min_group_score", 0.0)

    reasons: list[str] = []
    ok = True

    # Check overall score
    if metrics["overall_score"] < min_overall:
        ok = False
        reasons.append(
            f"overall_score {metrics['overall_score']:.3f} < min_overall_score {min_overall:.3f}"
        )

    # Check per-group score
    for group, score in metrics["group_scores"].items():
        if score < min_group:
            ok = False
            reasons.append(
                f"group {group} score {score:.3f} < min_group_score {min_group:.3f}"
            )

    return ok, reasons


def write_artifacts(model_cfg: dict, metrics: dict, passed: bool, reasons: list[str]) -> tuple[str, str]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    base_name = f"{model_cfg['model_id']}_{model_cfg['version']}_{model_cfg['evaluation_profile']}"
    metrics_path = REPORTS_DIR / f"{base_name}_metrics.json"
    report_path = REPORTS_DIR / f"{base_name}_report.html"

    # Save metrics JSON
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save a simple HTML report (placeholder for DeepEval HTML)
    status_str = "PASS" if passed else "FAIL"
    reasons_html = (
        "<ul>" + "".join(f"<li>{r}</li>" for r in reasons) + "</ul>" if reasons else "<p>No failures</p>"
    )
    html_content = f"""
    <html>
      <head>
        <title>CE Report - {model_cfg['model_id']} v{model_cfg['version']}</title>
      </head>
      <body>
        <h1>Continuous Evaluation Report</h1>
        <p><strong>Model:</strong> {model_cfg['model_id']}</p>
        <p><strong>Version:</strong> {model_cfg['version']}</p>
        <p><strong>Profile:</strong> {model_cfg['evaluation_profile']}</p>
        <p><strong>Status:</strong> {status_str}</p>
        <h2>Metrics</h2>
        <pre>{json.dumps(metrics, indent=2)}</pre>
        <h2>Gatekeeper</h2>
        {reasons_html}
      </body>
    </html>
    """
    with report_path.open("w", encoding="utf-8") as f:
        f.write(html_content)

    return str(metrics_path), str(report_path)


def evaluate_model_manifest(manifest_path: Path, history: dict) -> tuple[bool, dict]:
    print(f"\n=== Evaluating manifest: {manifest_path} ===")
    model_cfg = load_yaml(manifest_path)
    key = build_key(model_cfg)

    if has_successful_eval(history, key):
        print(f"Skipping: already have a successful CE run for {key}")
        return True, history

    # Orchestrator runs DeepEval (stub)
    metrics = run_deepeval_stub(model_cfg)

    # Gatekeeper
    passed, reasons = apply_gatekeeper(model_cfg, metrics)

    # Write artifacts
    metrics_path, report_path = write_artifacts(model_cfg, metrics, passed, reasons)

    # Update history
    history[key] = {
        "status": "pass" if passed else "failed_gate",
        "metrics_path": metrics_path,
        "report_path": report_path,
        "reasons": reasons,
        "created_at": datetime.now(UTC).isoformat(),
    }

    if passed:
        print(f"[PASS] {key}")
    else:
        print(f"[FAIL] {key}")
        for r in reasons:
            print(f"  - {r}")

    return passed, history


def main(argv: list[str]) -> int:
    if not argv:
        print("No manifest paths provided. Nothing to do.")
        return 0

    history = load_history()
    all_ok = True

    for path_str in argv:
        manifest_path = (ROOT / path_str).resolve()
        if not manifest_path.exists():
            print(f"WARNING: manifest does not exist: {manifest_path}")
            continue

        passed, history = evaluate_model_manifest(manifest_path, history)
        if not passed:
            all_ok = False

    save_history(history)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
