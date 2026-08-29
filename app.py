import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import yaml

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PM2.5 Air Quality Forecast",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

PROJECT_ROOT = Path(__file__).resolve().parent

# ─── Clean CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    /* Hide broken Material Symbols icon text in expanders */
    [data-testid="stExpander"] summary [data-testid="stIconMaterial"],
    [data-testid="stExpander"] summary .material-symbols-rounded,
    [data-testid="stExpander"] summary span[style*="Material"] {
        display: none !important;
    }

    /* Page title */
    .page-title {
        font-size: 1.7rem;
        font-weight: 800;
        color: #F1F5F9;
        margin-bottom: 0.15rem;
        letter-spacing: -0.02em;
    }
    .page-subtitle {
        font-size: 0.95rem;
        color: #94A3B8;
        margin-bottom: 1.8rem;
        line-height: 1.4;
    }

    /* Health Status Banner */
    .health-card {
        padding: 1.3rem 1.6rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.4rem;
    }
    .health-card .status-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.85;
    }
    .health-card .status-title {
        font-size: 1.55rem;
        font-weight: 700;
        margin: 4px 0 6px 0;
    }
    .health-card .status-advice {
        font-size: 0.92rem;
        font-weight: 400;
        opacity: 0.9;
        line-height: 1.4;
    }
    .health-good { background: linear-gradient(135deg, #10B981 0%, #059669 100%); }
    .health-moderate { background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%); color: #1E293B !important; }
    .health-moderate .status-label, .health-moderate .status-advice { color: #1E293B !important; opacity: 0.75; }
    .health-usg { background: linear-gradient(135deg, #F97316 0%, #EA580C 100%); }
    .health-unhealthy { background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); }
    .health-very-unhealthy { background: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%); }
    .health-hazardous { background: linear-gradient(135deg, #991B1B 0%, #7F1D1D 100%); }

    /* Stat Cards — dark surface */
    .stat-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .stat-card .stat-label {
        font-size: 0.75rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .stat-card .stat-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F1F5F9;
    }
    .stat-card .stat-unit {
        font-size: 0.85rem;
        font-weight: 400;
        color: #94A3B8;
    }

    /* Info callout — dark variant */
    .info-callout {
        background: rgba(14, 165, 233, 0.1);
        border-left: 3px solid #0EA5E9;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #7DD3FC;
        margin-bottom: 1rem;
        line-height: 1.5;
    }

    /* Section headers */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #E2E8F0;
        margin-bottom: 0.6rem;
        margin-top: 1rem;
    }

    /* Reduce Streamlit element spacing */
    div[data-testid="stHorizontalBlock"] {
        gap: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)



# ─── Data Loading ───────────────────────────────────────────────────────────────

FEATURE_INFO = {
    "cpm25": ("PM2.5 Concentration", "µg/m³", "Fine particles (≤2.5µm) that enter lungs — the main air quality metric."),
    "t2":    ("Temperature (2m)", "K", "Ground-level air temperature. Affects how pollutants mix and disperse."),
    "u10":   ("Wind Speed (East-West)", "m/s", "Horizontal wind — carries pollution plumes across regions."),
    "v10":   ("Wind Speed (North-South)", "m/s", "Vertical wind — carries pollution plumes across regions."),
    "rain":  ("Rainfall", "mm", "Rain washes particles out of the air, naturally cleaning pollution."),
    "pblh":  ("Boundary Layer Height", "m", "How high pollutants can mix. Low = trapped near surface."),
    "swdown": ("Solar Radiation", "W/m²", "Sunlight drives chemical reactions that create secondary pollution."),
    "psfc":  ("Surface Pressure", "Pa", "High pressure traps stagnant air, worsening pollution."),
    "q2":    ("Humidity", "kg/kg", "Moisture in air near the ground."),
    "NOx":   ("NOx Emissions", "mol/m²s", "From vehicles and power plants — a key PM2.5 precursor."),
    "SO2":   ("SO2 Emissions", "mol/m²s", "From coal and oil burning — forms sulfate particles."),
    "NH3":   ("Ammonia Emissions", "mol/m²s", "From agriculture — reacts with NOx/SO2 to form fine particles."),
    "PM25":  ("Direct PM2.5 Emissions", "kg/m²s", "Particles directly released from fires and factories."),
}


@st.cache_data
def load_params():
    p = PROJECT_ROOT / "params.yaml"
    if p.exists():
        with open(p, "r") as f:
            return yaml.safe_load(f)
    return {}


@st.cache_data
def load_predictions():
    """Load predictions: model output → raw test data → synthetic demo."""
    preds_path = PROJECT_ROOT / "outputs" / "preds.npy"
    if preds_path.exists():
        try:
            return np.load(preds_path, mmap_mode="r"), "Model Forecast"
        except Exception:
            pass

    test_cpm25 = PROJECT_ROOT / "data" / "test_in" / "cpm25.npy"
    if test_cpm25.exists():
        try:
            arr = np.load(test_cpm25, mmap_mode="r")
            if arr.ndim == 4:
                return np.transpose(arr, (0, 2, 3, 1)), "Raw Test History"
        except Exception:
            pass

    # Synthetic fallback
    np.random.seed(42)
    H, W, N, steps = 140, 124, 10, 16
    synth = np.zeros((N, H, W, steps), dtype=np.float32)
    cy, cx = H // 2, W // 2
    for s in range(N):
        for t in range(steps):
            yy, xx = np.ogrid[:H, :W]
            d = np.sqrt((yy - (cy + t * 1.2))**2 + (xx - (cx + t * 0.8))**2)
            synth[s, :, :, t] = np.clip(150 * np.exp(-d / (30 + t)) + np.random.normal(0, 3, (H, W)), 3, 400)
    return synth, "Demo Data (Synthetic)"


@st.cache_data
def load_feature(var_name):
    f = PROJECT_ROOT / "data" / "test_in" / f"{var_name}.npy"
    if f.exists():
        try:
            return np.load(f, mmap_mode="r")
        except Exception:
            return None
    return None


def health_status(peak_pm25):
    """Return (icon, title, css_class, advice) for a PM2.5 level."""
    if peak_pm25 <= 35:
        return "😊", "Good", "health-good", "Air is clean — safe for all outdoor activities."
    elif peak_pm25 <= 75:
        return "😐", "Moderate", "health-moderate", "Acceptable air quality. Sensitive individuals should take care."
    elif peak_pm25 <= 115:
        return "😷", "Unhealthy for Sensitive Groups", "health-usg", "Children, elderly, and those with lung/heart conditions should limit outdoor time."
    elif peak_pm25 <= 150:
        return "🚨", "Unhealthy", "health-unhealthy", "Everyone may feel effects. Consider wearing an N95 mask outdoors."
    elif peak_pm25 <= 250:
        return "⚠️", "Very Unhealthy", "health-very-unhealthy", "Serious health risk. Avoid outdoor exertion. Use air purifiers indoors."
    else:
        return "🆘", "Hazardous", "health-hazardous", "Emergency: stay indoors, seal windows, and run air purifiers."


# ─── Main App ───────────────────────────────────────────────────────────────────

def main():
    st.markdown('<div class="page-title">🌍 PM2.5 Air Quality Forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">AI-powered 16-hour spatiotemporal pollution forecast using ConvLSTM, wind transport modeling, and episode detection.</div>', unsafe_allow_html=True)

    # Show inference result notification (persisted across rerun)
    if "inference_msg" in st.session_state:
        msg = st.session_state.pop("inference_msg")
        if msg["status"] == "success":
            st.success(msg["text"])
        else:
            st.error(msg["text"])

    params = load_params()
    preds, source = load_predictions()
    N, H, W, T = preds.shape

    # ── Navigation ──────────────────────────────────────────────────────────
    tab_forecast, tab_data, tab_about = st.tabs(["🔮 Forecast", "📊 Explore Data", "ℹ️ About"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1: FORECAST
    # ════════════════════════════════════════════════════════════════════════
    with tab_forecast:

        # Sample selector (compact)
        sample_idx = st.slider("Select event / sample", 0, max(0, N - 1), 0, help=f"There are {N} test samples from `{source}`.")

        # Time horizon slider
        col_slider, col_play = st.columns([5, 1])
        with col_slider:
            step = st.slider("Forecast horizon", 1, T, 1, format="+%dh") - 1
        with col_play:
            st.write("")
            animate = st.button("▶ Play")

        grid = preds[sample_idx, :, :, step]
        peak = float(np.max(grid))
        avg = float(np.mean(grid))
        severe_pct = float(np.sum(grid >= 150) / (H * W)) * 100

        # Health Banner
        icon, title, css, advice = health_status(peak)
        st.markdown(f"""
        <div class="health-card {css}">
            <div class="status-label">Air Quality Status</div>
            <div class="status-title">{icon} {title}</div>
            <div class="status-advice">{advice}</div>
        </div>
        """, unsafe_allow_html=True)

        # Stat Cards
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="stat-card"><div class="stat-label">Peak PM2.5</div><div class="stat-value">{peak:.0f} <span class="stat-unit">µg/m³</span></div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-card"><div class="stat-label">Domain Average</div><div class="stat-value">{avg:.0f} <span class="stat-unit">µg/m³</span></div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-card"><div class="stat-label">Severe Area (≥150)</div><div class="stat-value">{severe_pct:.1f}<span class="stat-unit">%</span></div></div>', unsafe_allow_html=True)

        st.write("")

        # Spatial Map
        fig = px.imshow(
            grid, origin="lower",
            color_continuous_scale="YlOrRd",
            labels=dict(x="Longitude →", y="Latitude →", color="PM2.5 (µg/m³)"),
            range_color=[0, max(200, float(np.max(preds[sample_idx])))],
        )
        fig.update_layout(
            title=dict(text=f"Predicted PM2.5 at +{step + 1}h", font=dict(size=14, color="#E2E8F0")),
            height=500, margin=dict(l=10, r=10, t=35, b=10),
            coloraxis_colorbar=dict(title="µg/m³", len=0.75),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1E293B",
            font=dict(color="#CBD5E1"),
        )

        map_slot = st.empty()
        map_slot.plotly_chart(fig, width="stretch")

        # Animation
        if animate:
            for s in range(T):
                g = preds[sample_idx, :, :, s]
                f = px.imshow(
                    g, origin="lower",
                    color_continuous_scale="YlOrRd",
                    labels=dict(x="Longitude →", y="Latitude →", color="PM2.5 (µg/m³)"),
                    range_color=[0, max(200, float(np.max(preds[sample_idx])))],
                )
                f.update_layout(
                    title=dict(text=f"Predicted PM2.5 at +{s + 1}h", font=dict(size=14, color="#E2E8F0")),
                    height=500, margin=dict(l=10, r=10, t=35, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1E293B",
                    font=dict(color="#CBD5E1"),
                )
                map_slot.plotly_chart(f, width="stretch")
                time.sleep(0.35)

        # 16-hour trend line
        st.markdown('<div class="section-title">📈 How PM2.5 changes over 16 hours</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-callout">This chart shows domain-wide peak and average PM2.5 across all 16 forecast hours. The colored bands mark health risk thresholds.</div>', unsafe_allow_html=True)

        max_ts = [float(np.max(preds[sample_idx, :, :, t])) for t in range(T)]
        avg_ts = [float(np.mean(preds[sample_idx, :, :, t])) for t in range(T)]
        hours = list(range(1, T + 1))

        fig_trend = go.Figure()

        # Threshold bands (background context — higher opacity for dark bg)
        fig_trend.add_hrect(y0=0, y1=35, fillcolor="#10B981", opacity=0.15, line_width=0)
        fig_trend.add_hrect(y0=35, y1=75, fillcolor="#F59E0B", opacity=0.12, line_width=0)
        fig_trend.add_hrect(y0=75, y1=150, fillcolor="#EF4444", opacity=0.12, line_width=0)
        fig_trend.add_hrect(y0=150, y1=max(300, max(max_ts) * 1.1), fillcolor="#991B1B", opacity=0.10, line_width=0)

        # Threshold labels
        fig_trend.add_hline(y=35, line_dash="dot", line_color="#475569", line_width=1,
                            annotation_text="Good (35)", annotation_position="top left",
                            annotation_font_size=10, annotation_font_color="#64748B")
        fig_trend.add_hline(y=75, line_dash="dot", line_color="#475569", line_width=1,
                            annotation_text="Moderate (75)", annotation_position="top left",
                            annotation_font_size=10, annotation_font_color="#64748B")
        fig_trend.add_hline(y=150, line_dash="dot", line_color="#475569", line_width=1,
                            annotation_text="Unhealthy (150)", annotation_position="top left",
                            annotation_font_size=10, annotation_font_color="#64748B")

        fig_trend.add_trace(go.Scatter(x=hours, y=max_ts, mode="lines+markers", name="Peak PM2.5",
                                       line=dict(color="#EF4444", width=2.5), marker=dict(size=5)))
        fig_trend.add_trace(go.Scatter(x=hours, y=avg_ts, mode="lines+markers", name="Average PM2.5",
                                       line=dict(color="#3B82F6", width=2.5), marker=dict(size=5)))

        fig_trend.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Forecast Hour",
            yaxis_title="PM2.5 (µg/m³)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#CBD5E1")),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1E293B",
            font=dict(color="#CBD5E1"),
            yaxis=dict(gridcolor="#334155"),
            xaxis=dict(gridcolor="#334155", dtick=1),
        )
        st.plotly_chart(fig_trend, width="stretch")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2: EXPLORE DATA
    # ════════════════════════════════════════════════════════════════════════
    with tab_data:
        st.markdown('<div class="section-title">📊 Input Variables Explorer</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-callout">The model uses 15 atmospheric and emission variables to predict future PM2.5. Select any variable below to visualize its spatial distribution.</div>', unsafe_allow_html=True)

        var_keys = list(FEATURE_INFO.keys())
        selected = st.selectbox(
            "Variable",
            var_keys,
            format_func=lambda k: f"{FEATURE_INFO[k][0]} ({k})"
        )

        name, unit, desc = FEATURE_INFO[selected]
        st.info(f"**{name}** — {desc}  \nUnit: `{unit}`")

        feat = load_feature(selected)

        if feat is not None:
            ndim = feat.ndim
            N_f = feat.shape[0]

            col1, col2 = st.columns(2)
            with col1:
                f_sample = st.slider("Sample", 0, max(0, N_f - 1), 0, key="feat_sample")
            with col2:
                t_max = feat.shape[1] - 1 if ndim == 4 else 0
                f_time = st.slider("Time step", 0, max(0, t_max), 0, key="feat_time")

            if ndim == 4:
                s2d = feat[f_sample, f_time, :, :]
            elif ndim == 3:
                s2d = feat[f_sample, :, :]
            else:
                s2d = feat

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Min", f"{float(np.min(s2d)):.4g} {unit}")
            mc2.metric("Max", f"{float(np.max(s2d)):.4g} {unit}")
            mc3.metric("Mean", f"{float(np.mean(s2d)):.4g} {unit}")

            fig_f = px.imshow(
                s2d, origin="lower",
                color_continuous_scale="Viridis",
                labels=dict(x="Longitude →", y="Latitude →", color=f"{selected} ({unit})"),
            )
            fig_f.update_layout(
                title=dict(text=f"{name} — Sample {f_sample}, Time {f_time}", font=dict(size=13, color="#E2E8F0")),
                height=480, margin=dict(l=10, r=10, t=35, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1E293B",
                font=dict(color="#CBD5E1"),
            )
            st.plotly_chart(fig_f, width="stretch")
        else:
            st.warning(f"File `data/test_in/{selected}.npy` not found.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3: ABOUT
    # ════════════════════════════════════════════════════════════════════════
    with tab_about:
        st.markdown('<div class="section-title">About This System</div>', unsafe_allow_html=True)

        st.markdown("""
This dashboard visualizes predictions from a **physics-informed deep learning model** that forecasts fine particulate matter (PM2.5) concentrations 16 hours into the future across a 140 × 124 spatial grid.

#### How it works

The model takes **10 hours of historical observations** (PM2.5 concentrations + 14 weather and emission variables) and produces **16 hourly spatial forecasts** using:

1. **ConvLSTM Encoder** — Learns spatiotemporal patterns from past observations
2. **WindWarp Module** — Simulates how wind physically transports pollution plumes
3. **Episode Detector** — Identifies extreme pollution event regions
4. **Spatial Attention** — Focuses on the most informative grid areas
5. **Autoregressive Decoder** — Generates forecasts one step at a time

#### Health Thresholds (PM2.5)

| Level | Range | Meaning |
|-------|-------|---------|
| 🟢 Good | 0–35 µg/m³ | Safe for everyone |
| 🟡 Moderate | 35–75 µg/m³ | Acceptable; sensitive groups take care |
| 🟠 Unhealthy (Sensitive) | 75–115 µg/m³ | At-risk groups should limit outdoor time |
| 🔴 Unhealthy | 115–150 µg/m³ | Everyone may feel effects |
| 🟣 Very Unhealthy | 150–250 µg/m³ | Serious health risk for all |
| 🟤 Hazardous | 250+ µg/m³ | Emergency conditions |
""")

        st.markdown("---")

        with st.expander("⚙️ Model Configuration (params.yaml)"):
            if params:
                st.json(params)
            else:
                st.warning("params.yaml not found.")

        with st.expander("🚀 Run Model Inference"):
            st.write("Generate predictions on test data in `data/test_in/`.")
            ckpt = PROJECT_ROOT / "models" / "best_model_p2.pt"
            if ckpt.exists():
                st.success(f"Model checkpoint: `{ckpt.name}` ({ckpt.stat().st_size / (1024*1024):.1f} MB)")
            else:
                st.warning("No checkpoint found.")

            if st.button("🚀 Run Inference"):
                with st.spinner("Running model..."):
                    try:
                        import subprocess
                        r = subprocess.run(
                            [sys.executable, "-m", "src.inference.predict"],
                            capture_output=True, text=True, cwd=str(PROJECT_ROOT)
                        )
                        if r.returncode == 0:
                            st.session_state["inference_msg"] = {
                                "status": "success",
                                "text": "✅ Inference complete! Predictions updated and reloaded."
                            }
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.session_state["inference_msg"] = {
                                "status": "error",
                                "text": f"Inference failed (exit {r.returncode}): {r.stderr[:300]}"
                            }
                            st.rerun()
                    except Exception as e:
                        st.error(str(e))


if __name__ == "__main__":
    main()
