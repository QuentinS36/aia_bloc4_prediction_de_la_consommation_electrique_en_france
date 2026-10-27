"""Dashboard : prévision J+1 (API), historique et suivi de la prévision (Neon DB).

Variables d'environnement : DATABASE_URL, API_URL (définies dans le service Render, ou dans .env en local).
Accessibilité : palette Okabe-Ito (daltonisme), styles de ligne distincts, titres explicites,
résumés textuels sous chaque graphique, aucune information portée par la couleur seule.
"""
import os
import time

try:  # en local, lit le .env de la racine ; sur Render, les variables d'environnement du service suffisent
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from sqlalchemy import create_engine, text

st.set_page_config(page_title="Prévision de consommation électrique", layout="wide")

BLUE, ORANGE = "#0072B2", "#D55E00"  # Okabe-Ito : distinguables par tous les profils de vision


def secret(name: str) -> str | None:
    """Variable d'environnement (Render, .env) ; à défaut, st.secrets (.streamlit/secrets.toml)."""
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name)
    except Exception:
        return None


@st.cache_resource
def engine():
    url = secret("DATABASE_URL")
    return create_engine(url.replace("postgres://", "postgresql://", 1), pool_pre_ping=True) if url else None


@st.cache_data(ttl=600)
def load(days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    since = (pd.Timestamp.now("UTC").tz_localize(None) - pd.Timedelta(days=days)).to_pydatetime()
    with engine().connect() as c:
        real = pd.read_sql(text("SELECT ds, y FROM dataset_hourly WHERE ds >= :s ORDER BY ds"), c, params={"s": since})
        pred = pd.read_sql(text("SELECT ds, yhat FROM predictions WHERE ds >= :s ORDER BY ds"), c, params={"s": since})
    return real, pred.drop_duplicates("ds", keep="last")


@st.cache_data(ttl=900, show_spinner="Appel de l'API (le premier appel peut être lent si le service était en veille)…")
def forecast() -> dict:
    """Appelle l'API ; tolère un démarrage à froid (service en veille) grâce à un délai long et 3 tentatives."""
    url = secret("API_URL").rstrip("/") + "/predict"
    last: Exception | None = None
    for _ in range(3):
        try:
            r = requests.get(url, timeout=90)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code < 500:
                raise  # erreur de requête : inutile de réessayer
            last = e
        except requests.RequestException as e:
            last = e
        time.sleep(5)
    raise last


def line(fig, x, y, name, color, dash="solid"):
    fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(color=color, dash=dash, width=2)))


st.title("Prévision de la consommation électrique en France")
st.caption("Consommation nationale horaire en MW. Modèle Prophet réentraîné chaque jour.")

if engine() is None:
    st.error("Base de données non configurée (DATABASE_URL).")
    st.stop()

tab_fc, tab_hist = st.tabs(["Prévision de demain", "Historique et qualité de la prévision"])

with tab_fc:
    if not secret("API_URL"):
        st.warning("API non configurée (API_URL).")
    else:
        try:
            fc = forecast()
            df = pd.DataFrame(fc["predictions"])
            df["ds"] = pd.to_datetime(df["ds"])
            peak = df.loc[df["yhat"].idxmax()]
            low = df.loc[df["yhat"].idxmin()]
            c1, c2, c3 = st.columns(3)
            c1.metric("Pic prévu", f"{peak['yhat']:,.0f} MW".replace(",", " "), f"à {peak['ds']:%H:%M} UTC", delta_color="off")
            c2.metric("Minimum prévu", f"{low['yhat']:,.0f} MW".replace(",", " "), f"à {low['ds']:%H:%M} UTC", delta_color="off")
            c3.metric("Version du modèle", fc["model_version"])
            fig = go.Figure()
            line(fig, df["ds"], df["yhat"], "Prévision", ORANGE, "dash")
            fig.update_layout(title=f"Consommation prévue le {fc['date']} (UTC)", yaxis_title="MW", xaxis_title="Heure (UTC)",
                              legend=dict(orientation="h"))
            st.plotly_chart(fig, width='stretch')
            st.caption(f"Résumé : le {fc['date']}, la consommation devrait varier entre {low['yhat']:,.0f} MW "
                       f"(à {low['ds']:%H:%M} UTC) et {peak['yhat']:,.0f} MW (à {peak['ds']:%H:%M} UTC).".replace(",", " "))
        except Exception as e:
            st.error(f"Prévision indisponible : {e}")

with tab_hist:
    days = st.slider("Période affichée (jours)", 3, 90, 14)
    real, pred = load(days)
    if real.empty:
        st.info("Aucune donnée sur la période.")
    else:
        fig = go.Figure()
        line(fig, real["ds"], real["y"], "Consommation réelle", BLUE)
        if not pred.empty:
            line(fig, pred["ds"], pred["yhat"], "Consommation prévue", ORANGE, "dash")
        fig.update_layout(title=f"Consommation réelle et prévue, {days} derniers jours", yaxis_title="MW",
                          xaxis_title="Date (UTC)", legend=dict(orientation="h"))
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig, width='stretch')
        j = pred.merge(real, on="ds")
        if len(j) >= 12:
            mape = (abs(j["y"] - j["yhat"]) / j["y"]).mean()
            st.metric("Erreur moyenne (MAPE) sur les prévisions disponibles", f"{mape:.2%}")
        st.caption(f"Résumé : sur {days} jours, la consommation moyenne est de {real['y'].mean():,.0f} MW, "
                   f"minimum {real['y'].min():,.0f} MW, maximum {real['y'].max():,.0f} MW.".replace(",", " "))
        with st.expander("Données du graphique (tableau)"):
            st.dataframe(real.tail(200), width='stretch')
