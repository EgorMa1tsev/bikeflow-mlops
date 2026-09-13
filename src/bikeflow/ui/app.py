"""Streamlit web interface for BikeFlow.

The page is a thin client: it only calls the HTTP API, so it can run beside the
API in Docker or Kubernetes without sharing the journal or the model. The
experiments tab is the exception — it reads the MLflow store directly, the same
way the MLflow UI does.

    streamlit run src/bikeflow/ui/app.py
"""

from __future__ import annotations

import datetime as dt
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

from bikeflow.config import get_settings
from bikeflow.ml.config import load_config
from bikeflow.ml.features import api_input_contract

SEOUL = ZoneInfo("Asia/Seoul")
TIMEOUT = 30
CONTRACT = api_input_contract()
UI = load_config()["ui"]

LABELS = {
    "temperature_c": "Температура, °C",
    "humidity_pct": "Влажность, %",
    "wind_speed_m_s": "Ветер, м/с",
    "visibility_10m": "Видимость, ×10 м",
    "dew_point_c": "Точка росы, °C",
    "solar_radiation_mj_m2": "Солнечная радиация, МДж/м²",
    "rainfall_mm": "Дождь, мм",
    "snowfall_cm": "Снег, см",
}
DEFAULTS = {
    "temperature_c": 12.0,
    "humidity_pct": 55.0,
    "wind_speed_m_s": 1.5,
    "visibility_10m": 1800.0,
    "dew_point_c": 3.0,
    "solar_radiation_mj_m2": 0.5,
    "rainfall_mm": 0.0,
    "snowfall_cm": 0.0,
}


def api_url() -> str:
    return st.session_state.get("api_url", get_settings().api_url).rstrip("/")


def get(path: str) -> Any:
    response = requests.get(api_url() + path, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def post(path: str, body: dict[str, Any] | None = None) -> tuple[int, Any]:
    """POST to the API, returning the status code so the page can explain a refusal."""
    response = requests.post(api_url() + path, json=body or {}, timeout=TIMEOUT)
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, {"detail": response.text}


def error_text(payload: Any) -> str:
    detail = payload.get("detail") if isinstance(payload, dict) else payload
    return detail if isinstance(detail, str) else str(detail)


# --------------------------------------------------------------------------- page


st.set_page_config(page_title="BikeFlow", page_icon="🚲", layout="wide")
st.title("🚲 BikeFlow — прогноз спроса на велопрокат")

with st.sidebar:
    st.text_input("Адрес API", value=get_settings().api_url, key="api_url")
    if st.button("Обновить данные", width="stretch"):
        st.rerun()

try:
    model = get("/model")
except requests.RequestException as exc:
    st.error(f"API недоступен по адресу {api_url()}: {exc}")
    st.stop()

latest_drift: dict[str, Any] | None = None
try:
    latest_drift = get("/drift/latest")
except requests.HTTPError:
    latest_drift = None

with st.sidebar:
    st.caption("Модель в работе")
    st.code(model["model_version"], language=None)
    source = "реестр MLflow" if model["source"] == "registry" else "файл"
    reference = (
        f"эталонная MAE {model['reference_mae']:.0f}"
        if model["reference_mae"]
        else "эталонная MAE неизвестна"
    )
    st.caption(f"источник: {source} · {reference}")

if latest_drift and latest_drift["concept_drift"]:
    st.error(
        f"**Дрейф модели.** На окне до {latest_drift['window_end'][:16]} ошибка MAE "
        f"{latest_drift['current_mae']:.0f} — это "
        f"{latest_drift['mae_ratio']:.2f} × эталона при пороге "
        f"{latest_drift['thresholds']['concept_drift_mae_ratio']}. Модель пора переобучить.",
        icon="⚠️",
    )
elif latest_drift:
    st.success(
        f"Проверка дрейфа до {latest_drift['window_end'][:16]}: модель в норме "
        f"(MAE {latest_drift['current_mae']:.0f} = {latest_drift['mae_ratio']:.2f} × эталона).",
        icon="✅",
    )

forecast_tab, journal_tab, monitoring_tab, experiments_tab = st.tabs(
    ["Прогноз", "Журнал прогнозов", "Мониторинг и переобучение", "Эксперименты"]
)


# --------------------------------------------------------------------------- forecast

with forecast_tab:
    st.subheader("Прогноз на один час")
    with st.form("prediction"):
        left, right = st.columns(2)
        with left:
            day = st.date_input("Дата", value=dt.date(2018, 12, 1), format="YYYY-MM-DD")
            hour = st.slider("Час", 0, 23, 18)
            holiday = st.checkbox("Праздничный день")
            functioning = st.checkbox("Прокат работает", value=True)
        with right:
            weather = {}
            for field, label in LABELS.items():
                spec = CONTRACT[field]
                weather[field] = st.number_input(
                    label,
                    min_value=float(spec["min"]),
                    max_value=float(spec["max"]),
                    value=DEFAULTS[field],
                    step=0.1,
                )
        submitted = st.form_submit_button("Спрогнозировать", type="primary")

    if submitted:
        moment = dt.datetime.combine(day, dt.time(hour), tzinfo=SEOUL)
        code, payload = post(
            "/predict",
            {
                "prediction_time": moment.isoformat(),
                "holiday": holiday,
                "functioning_day": functioning,
                **weather,
            },
        )
        if code == 200:
            st.metric(
                f"Прогноз на {moment:%Y-%m-%d %H:00}",
                f"{payload['predicted_rentals']:.0f} велосипедов",
            )
            st.caption(
                f"запись журнала №{payload['prediction_id']} · модель {payload['model_version']}"
            )
        else:
            st.error(f"API отклонил запрос ({code}): {error_text(payload)}")


# --------------------------------------------------------------------------- journal


def journal_table(records: list[dict[str, Any]], reference_mae: float | None) -> pd.DataFrame:
    """Recent predictions with an anomaly flag for unusually large errors."""
    limit = reference_mae * UI["anomaly_error_ratio"] if reference_mae else None
    rows = []
    for record in records:
        error = record["absolute_error"]
        actual = record["actual_rentals"]
        rows.append(
            {
                "№": record["prediction_id"],
                "Час": record["prediction_time"][:16].replace("T", " "),
                "Прогноз": round(record["predicted_rentals"]),
                "Факт": None if actual is None else round(actual),
                "Ошибка": None if error is None else round(error),
                "Аномалия": "🔴" if limit and error is not None and error > limit else "",
                "Модель": record["model_version"],
            }
        )
    return pd.DataFrame(rows)


with journal_tab:
    st.subheader("Последние прогнозы")
    limit = st.slider("Сколько записей показать", 10, 500, int(UI["recent_limit"]), step=10)
    records = get(f"/predictions?limit={limit}")
    if not records:
        st.info(
            "Журнал пуст: сделайте прогноз или пустите поток данных `python -m bikeflow.replay`."
        )
    else:
        reference_mae = model["reference_mae"]
        table = journal_table(records, reference_mae)
        anomalies = int((table["Аномалия"] != "").sum())
        known = table["Ошибка"].notna().sum()
        columns = st.columns(3)
        columns[0].metric("Записей", len(table))
        columns[1].metric("Из них с фактом", int(known))
        columns[2].metric("Аномальных ошибок", anomalies)
        if reference_mae:
            st.caption(
                f"Аномалия — ошибка больше {UI['anomaly_error_ratio']:g} × эталонной MAE "
                f"({reference_mae:.0f}), то есть больше "
                f"{reference_mae * UI['anomaly_error_ratio']:.0f}."
            )
        st.dataframe(table, width="stretch", hide_index=True)


# --------------------------------------------------------------------------- monitoring

with monitoring_tab:
    st.subheader("Дрейф")
    left, right = st.columns(2)
    if left.button("Проверить дрейф сейчас", width="stretch"):
        code, payload = post("/drift/check")
        if code == 200:
            st.session_state["drift_started_retraining"] = payload.get("retraining_started", False)
            st.rerun()
        else:
            st.warning(f"Проверка не выполнена ({code}): {error_text(payload)}")
    if st.session_state.pop("drift_started_retraining", False):
        st.info("Дрейф модели — переобучение запущено автоматически.", icon="🔁")

    if latest_drift:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Вид дрейфа": "данные",
                        "Есть": "да" if latest_drift["data_drift"] else "нет",
                        "Значение": f"{latest_drift['drifted_feature_share']:.0%} признаков",
                    },
                    {
                        "Вид дрейфа": "спрос",
                        "Есть": "да" if latest_drift["target_drift"] else "нет",
                        "Значение": f"{latest_drift['target_drift_score']:.2f}",
                    },
                    {
                        "Вид дрейфа": "модель (concept)",
                        "Есть": "да" if latest_drift["concept_drift"] else "нет",
                        "Значение": f"MAE {latest_drift['current_mae']:.0f} = "
                        f"{latest_drift['mae_ratio']:.2f} × эталона",
                    },
                ]
            ),
            hide_index=True,
        )
        # The page fetches the report itself: in Kubernetes the API address is internal
        # to the cluster, so a plain link would not open in the viewer's browser.
        if st.toggle("Показать отчёт Evidently"):
            report = requests.get(api_url() + "/drift/report", timeout=TIMEOUT)
            if report.status_code == 200:
                components.html(report.text, height=900, scrolling=True)
            else:
                st.info("Отчёта ещё нет: сначала проверьте дрейф.")
    else:
        st.info("Проверок дрейфа ещё не было.")

    st.subheader("Переобучение")
    status = get("/retrain/status")
    if not model["retraining_enabled"]:
        st.warning(
            "API работает с файлом модели. Чтобы переобучать, запустите его из реестра MLflow: "
            "`BIKEFLOW_MODEL_URI=models:/bikeflow-demand@champion`."
        )
    elif right.button("Переобучить модель", type="primary", width="stretch"):
        code, payload = post("/retrain")
        if code == 202:
            st.rerun()
        else:
            st.warning(f"Не запущено ({code}): {error_text(payload)}")

    if status["state"] == "running":
        st.info(
            f"Переобучение идёт ({status['trigger']}). Обновите страницу через минуту.", icon="⏳"
        )
    elif status["state"] == "failed":
        st.error(f"Последнее переобучение упало: {status['error']}")
    elif status["result"]:
        result = status["result"]
        verdict = (
            "новая модель прошла quality gate и работает"
            if result["promoted"]
            else "новая модель хуже, в работе прежняя"
        )
        st.write(
            f"Последнее переобучение ({result['trigger']}): на отложенных часах "
            f"{result['holdout_start'][:13]} … {result['holdout_end'][:13]} MAE "
            f"{result['champion_mae']:.0f} → {result['challenger_mae']:.0f} "
            f"({result['improvement']:+.1%} при пороге {result['min_improvement']:.0%}) "
            f"— {verdict}."
        )
        st.caption(f"версия в реестре MLflow: {result['registered_version']}")
    else:
        st.caption("Переобучений ещё не было.")


# --------------------------------------------------------------------------- experiments


@st.cache_data(ttl=30)
def experiment_runs() -> pd.DataFrame:
    """Runs of the MLflow experiment: the training runs and every retraining."""
    from mlflow.tracking import MlflowClient

    from bikeflow.ml.training.tracking import tracking_uri

    tracking = load_config()["tracking"]
    client = MlflowClient(tracking_uri=tracking_uri())
    experiment = client.get_experiment_by_name(tracking["experiment"])
    if experiment is None:
        return pd.DataFrame()

    rows = []
    for run in client.search_runs([experiment.experiment_id], max_results=50):
        values, tags = run.data.metrics, run.data.tags
        gate = tags.get("quality_gate")
        rows.append(
            {
                "Запуск": run.info.run_name,
                "Начат": dt.datetime.fromtimestamp(run.info.start_time / 1000).strftime(
                    "%Y-%m-%d %H:%M"
                ),
                "validation MAE": values.get("validation_mae"),
                "test MAE": values.get("test_mae"),
                "MAE на отложенных часах": (
                    f"{values['holdout_mae_champion']:.0f} → {values['holdout_mae_challenger']:.0f}"
                    if "holdout_mae_challenger" in values
                    else None
                ),
                "Quality gate": {"passed": "пройден", "rejected": "не пройден"}.get(gate),
                "Версия модели": tags.get("model_version"),
            }
        )
    return pd.DataFrame(rows)


with experiments_tab:
    st.subheader("Запуски обучения в MLflow")
    try:
        runs = experiment_runs()
    except Exception as exc:  # noqa: BLE001 - the page explains what is missing
        st.warning(f"MLflow недоступен из интерфейса: {exc}")
    else:
        if runs.empty:
            st.info("В MLflow ещё нет запусков эксперимента.")
        else:
            st.dataframe(runs, width="stretch", hide_index=True)
        st.caption(
            "Полный интерфейс с параметрами и артефактами: "
            "`mlflow ui --backend-store-uri sqlite:///mlflow.db` → http://127.0.0.1:5000"
        )
