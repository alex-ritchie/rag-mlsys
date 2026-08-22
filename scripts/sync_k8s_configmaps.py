#!/usr/bin/env python3
"""Embed k8s/prometheus/alert-rules.yml and k8s/grafana/dashboards/*.json into their ConfigMap manifests.

Plain manifests only (spec §5.8) means no Kustomize configMapGenerator; this script keeps the
source files (editable, reviewable) and the generated ConfigMaps in sync. Run after editing either.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
K8S = ROOT / "k8s"


def indent(text: str, n: int = 4) -> str:
    pad = " " * n
    return "\n".join(pad + line if line.strip() else "" for line in text.rstrip("\n").splitlines())


def main() -> None:
    rules = (K8S / "prometheus" / "alert-rules.yml").read_text()
    (K8S / "61-prometheus-rules-configmap.yaml").write_text(
        "# Generated from k8s/prometheus/alert-rules.yml by scripts/sync_k8s_configmaps.py — edit the source file.\n"
        "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: prometheus-rules\n  namespace: mlsysbook\ndata:\n  alert-rules.yml: |\n"
        + indent(rules)
        + "\n"
    )
    dash_dir = K8S / "grafana" / "dashboards"
    entries = []
    for p in sorted(dash_dir.glob("*.json")):
        json.loads(p.read_text())  # validate
        entries.append(f"  {p.name}: |\n" + indent(p.read_text()))
    (K8S / "72-grafana-dashboards-configmap.yaml").write_text(
        "# Generated from k8s/grafana/dashboards/*.json by scripts/sync_k8s_configmaps.py — edit the JSON files.\n"
        "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: grafana-dashboards\n  namespace: mlsysbook\ndata:\n"
        + "\n".join(entries)
        + "\n"
    )
    print(f"synced alert rules + {len(entries)} dashboards into ConfigMaps")


if __name__ == "__main__":
    main()
