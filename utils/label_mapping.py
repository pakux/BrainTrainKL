import json
import os
from typing import Dict, Optional

import pandas as pd


def _normalize_key(value) -> str:
    """Normalize label value for JSON-safe mapping keys."""
    return str(value).strip()


def _is_int_like(value) -> bool:
    try:
        text = str(value).strip()
        if text == "":
            return False
        return float(text).is_integer()
    except Exception:
        return False


def labels_are_integer_encoded(labels: pd.Series) -> bool:
    cleaned = labels.dropna().map(lambda x: str(x).strip())
    if cleaned.empty:
        return False
    return bool(cleaned.map(_is_int_like).all())


def build_label_mapping(labels: pd.Series) -> Dict[str, int]:
    """
    Build deterministic class-name -> index mapping.
    For integer labels, keeps identity mapping. For strings, maps sorted labels to [0..n-1].
    """
    cleaned = labels.dropna().map(_normalize_key)
    if cleaned.empty:
        raise ValueError("Cannot build label mapping from empty label column.")

    if labels_are_integer_encoded(cleaned):
        unique_vals = sorted({int(float(v)) for v in cleaned.tolist()})
        return {str(v): int(v) for v in unique_vals}

    unique_vals = sorted(set(cleaned.tolist()))
    return {label: idx for idx, label in enumerate(unique_vals)}


def save_label_mapping(class_to_index: Dict[str, int], output_path: str, column_name: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    index_to_class = {str(v): k for k, v in class_to_index.items()}
    payload = {
        "column_name": column_name,
        "n_classes": len(class_to_index),
        "class_to_index": class_to_index,
        "index_to_class": index_to_class,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def load_label_mapping(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "class_to_index" in payload:
        mapping = payload["class_to_index"]
    elif isinstance(payload, dict):
        mapping = payload
    else:
        raise ValueError(f"Invalid label map format in {path}")
    return {_normalize_key(k): int(v) for k, v in mapping.items()}


def resolve_or_create_label_mapping(
    csv_path: str,
    column_name: str,
    mapping_path: Optional[str] = None,
    auto_create: bool = False,
) -> Optional[Dict[str, int]]:
    """
    Resolve mapping for a classification target.
    - If `mapping_path` exists: load and return.
    - Else infer from CSV labels.
      - For integer-encoded labels: return None (no mapping needed).
      - For non-integer labels: create mapping only when `auto_create=True`.
    """
    labels = pd.read_csv(csv_path, usecols=[column_name])[column_name]

    if mapping_path and os.path.exists(mapping_path):
        return load_label_mapping(mapping_path)

    if labels_are_integer_encoded(labels):
        return None

    if not auto_create:
        raise ValueError(
            f"Non-integer labels found in column '{column_name}', but no label map is available.\n"
            f"Provide --label-map PATH or enable --label-map-auto."
        )

    mapping = build_label_mapping(labels)
    if mapping_path:
        save_label_mapping(mapping, mapping_path, column_name)
        print(f"Label map created at: {mapping_path}")
    return mapping
