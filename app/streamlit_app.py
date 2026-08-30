"""Interactive Streamlit demo backed by the FastAPI inference service."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "model.joblib"
METADATA = ROOT / "artifacts" / "model_metadata.json"
CHECKSUM = ROOT / "artifacts" / "model.joblib.sha256"
LOGO = ROOT / "docs" / "assets" / "images" / "star-wars-logo.png"

BEST_MODEL_ARGS: list[str] = []


def _api_call(base_url: str, path: str, payload: Any | None = None) -> Any:
    """Call a JSON endpoint without coupling the UI to serving internals."""
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_state() -> dict[str, Any]:
    if not ARTIFACT.exists():
        return {"exists": False}
    digest = hashlib.sha256(ARTIFACT.read_bytes()).hexdigest()
    recorded = CHECKSUM.read_text(encoding="utf-8").split()[0] if CHECKSUM.exists() else None
    return {
        "exists": True,
        "path": str(ARTIFACT.relative_to(ROOT)),
        "size_mb": round(ARTIFACT.stat().st_size / 1_048_576, 2),
        "sha256": digest,
        "checksum_matches": recorded == digest if recorded else None,
    }


def _run_training(args: list[str]) -> tuple[int, str]:
    command = [sys.executable, "-m", "starwars_autocalls.cli", "train", *args]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    output = (process.stdout + "\n" + process.stderr).strip()
    return process.returncode, output


def _render_brand() -> None:
    encoded_logo = base64.b64encode(LOGO.read_bytes()).decode("ascii") if LOGO.exists() else ""
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{ background: #fff; color: #000; }}
        [data-testid="stHeader"] {{
            background-color: #000 !important;
            background-image: url("data:image/png;base64,{encoded_logo}");
            background-repeat: no-repeat;
            background-position: 18px center;
            background-size: 96px auto;
        }}
        [data-testid="stHeader"]::after {{
            content: "Star Wars Autocalls";
            color: #fff;
            font-size: 0.9rem;
            font-weight: 600;
            left: 130px;
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
        }}
        [data-testid="stHeader"] button,
        [data-testid="stHeader"] button svg {{ color: #fff !important; fill: #fff !important; }}
        [data-testid="stSidebar"] {{ background: #fff; border-right: 1px solid #000; }}
        h1, h2, h3, p, label, [data-testid="stMetricValue"] {{ color: #000 !important; }}
        .stButton > button {{ background: #000; color: #fff !important; border: 1px solid #000; }}
        .stButton > button p {{ color: #fff !important; }}
        .stButton > button:hover {{ background: #222; color: #fff !important; border-color: #000; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _training_tab() -> None:
    st.subheader("Entrenamiento")
    metadata = _load_json(METADATA)
    current_model = metadata.get("model_name", "mejor modelo validado")
    artifact_exists = _artifact_state().get("exists", False)
    st.caption(f"Modelo final: **{current_model}**")
    st.info(
        "Reentrena desde los tres CSV de `data/raw`. El ajuste usa las RFQs ejecutadas "
        "con target de train y validation; test se reserva para la evaluación final."
    )
    action = "Reentrenar" if artifact_exists else "Entrenar mejor modelo"
    if st.button(action, type="primary"):
        with st.status("Entrenando…", expanded=True) as status:
            return_code, output = _run_training(BEST_MODEL_ARGS)
            st.code(output or "Sin salida", language="text")
            if return_code == 0:
                status.update(label="Entrenamiento completado", state="complete")
                st.session_state["last_training_output"] = output
                st.rerun()
            status.update(label="El entrenamiento ha fallado", state="error")


def _results_tab() -> None:
    st.subheader("Resultados")
    metadata = _load_json(METADATA)
    artifact = _artifact_state()
    if not artifact["exists"]:
        st.info("Todavía no hay un artefacto entrenado.")
        return
    metrics = metadata.get("test_metrics", {})
    columns = st.columns(4)
    columns[0].metric("MAE test", _format_metric(metrics.get("mae")))
    columns[1].metric("RMSE test", _format_metric(metrics.get("rmse")))
    columns[2].metric("R² test", _format_metric(metrics.get("r2")))
    columns[3].metric("Filas test", f"{metadata.get('test_rows', '—'):,}")
    st.markdown(
        f"**{metadata.get('model_name', '—')}** · "
        f"{metadata.get('model_family', '—')} · "
        f"features `{metadata.get('feature_block', '—')}`"
    )
    st.caption(
        f"Artefacto: {artifact['size_mb']} MB · "
        f"checksum {'válido' if artifact.get('checksum_matches') else 'no verificado'}"
    )
    with st.expander("Detalles del modelo"):
        st.json(
            {
                "modelo": metadata.get("model_name"),
                "familia": metadata.get("model_family"),
                "estrategia": metadata.get("selection_source"),
                "features": metadata.get("feature_block"),
                "entrenamiento_utc": metadata.get("trained_at_utc"),
            }
        )
    st.download_button(
        "Guardar metadatos",
        data=json.dumps(metadata, indent=2, ensure_ascii=False),
        file_name="model_metadata.json",
        mime="application/json",
    )


def _format_metric(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def _load_uploaded_payloads(uploaded: Any) -> list[dict[str, Any]]:
    if uploaded.name.lower().endswith(".json"):
        parsed = json.loads(uploaded.getvalue().decode("utf-8"))
        payloads = parsed if isinstance(parsed, list) else [parsed]
    else:
        payloads = list(csv.DictReader(uploaded.getvalue().decode("utf-8").splitlines()))
    if not payloads or not all(isinstance(row, dict) for row in payloads):
        raise ValueError("El archivo debe contener una o más filas tipo objeto.")
    return payloads


def _inference_tab(api_url: str) -> None:
    st.subheader("Inferencia")
    uploaded = st.file_uploader("Carga un fichero JSON o CSV", type=["json", "csv"])
    if uploaded is None:
        st.info("La inferencia requiere un fichero con una o más filas.")
        return
    try:
        payloads = _load_uploaded_payloads(uploaded)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        st.error(f"Archivo inválido: {exc}")
        return
    st.caption(f"{len(payloads)} fila(s) cargada(s) desde `{uploaded.name}`")
    st.dataframe(payloads[:5], use_container_width=True, hide_index=True)
    if len(payloads) > 5:
        st.caption("Se muestran las cinco primeras filas; se procesará el archivo completo.")
    if len(payloads) > 1_000:
        st.error("El endpoint batch admite como máximo 1.000 RFQs por petición.")
        return
    if st.button("Calcular duración", type="primary"):
        try:
            response = _api_call(api_url, "/predict-batch", {"requests": payloads})
            predictions = response["predictions"]
            results = [
                {"fila": index, **result} for index, result in enumerate(predictions, start=1)
            ]
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError, TypeError) as exc:
            st.error(f"La inferencia ha fallado: {exc}")
            return
        if len(results) == 1:
            st.metric(
                "Duración estimada",
                f"{results[0]['predicted_avg_duration_months']:.2f} meses",
            )
        st.dataframe(results, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Star Wars Autocalls", page_icon="⚔", layout="wide")
    _render_brand()
    st.title("Star Wars Autocalls")
    default_api_url = os.environ.get("STARWARS_AUTOCALLS_API_URL", "http://127.0.0.1:8000")
    api_url = default_api_url
    tabs = st.tabs(["Entrenamiento", "Resultados", "Inferencia"])
    with tabs[0]:
        _training_tab()
    with tabs[1]:
        _results_tab()
    with tabs[2]:
        _inference_tab(api_url)


if __name__ == "__main__":
    main()
