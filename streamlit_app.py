import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import welch, csd

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="BARABAR v2.1 | Simulation Ic",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS & THEME
# ============================================================================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── BASE ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    .stApp {
        background: #070b14;
        color: #dde3f0;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1400px !important;
    }

    /* ── SIDEBAR ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1120 0%, #0f1929 100%) !important;
        border-right: 1px solid rgba(99, 157, 255, 0.12) !important;
    }
    section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }
    section[data-testid="stSidebar"] label {
        color: #8da0c0 !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
    }

    /* ── SLIDER ── */
    .stSlider [data-baseweb="slider"] { padding: 4px 6px; }
    .stSlider [data-testid="stTickBar"] { color: #4a5568; }

    /* ── PRIMARY BUTTON ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3a7ef4 0%, #6b4ef7 100%) !important;
        border: none !important;
        color: #fff !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.06em !important;
        padding: 0.65rem 1.2rem !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 20px rgba(58, 126, 244, 0.35) !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 28px rgba(107, 78, 247, 0.55) !important;
        transform: translateY(-2px) !important;
    }
    .stButton > button[kind="primary"]:active {
        transform: translateY(0px) !important;
    }

    /* ── METRICS ── */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 14px !important;
        padding: 1.1rem 1.3rem !important;
        backdrop-filter: blur(8px) !important;
    }
    [data-testid="stMetricLabel"] { color: #8da0c0 !important; font-size: 0.78rem !important; }
    [data-testid="stMetricValue"] {
        color: #639dff !important;
        font-weight: 700 !important;
        font-size: 1.6rem !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    [data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

    /* ── PROGRESS BAR ── */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #3a7ef4, #6b4ef7) !important;
        border-radius: 6px !important;
    }
    .stProgress > div > div {
        background: rgba(255,255,255,0.06) !important;
        border-radius: 6px !important;
    }

    /* ── ALERTS ── */
    .stSuccess { background: rgba(0,212,136,0.1) !important; border-color: rgba(0,212,136,0.3) !important; border-radius: 10px !important; }
    .stInfo    { background: rgba(58,126,244,0.1) !important; border-color: rgba(58,126,244,0.3) !important; border-radius: 10px !important; }
    .stWarning { background: rgba(255,159,28,0.1) !important; border-color: rgba(255,159,28,0.3) !important; border-radius: 10px !important; }
    .stError   { background: rgba(255,72,66,0.1)  !important; border-color: rgba(255,72,66,0.3)  !important; border-radius: 10px !important; }

    /* ── DIVIDER ── */
    hr {
        border: none !important;
        border-top: 1px solid rgba(255,255,255,0.07) !important;
        margin: 1.8rem 0 !important;
    }

    /* ── CAPTION ── */
    .stCaption { color: #4a5568 !important; font-size: 0.73rem !important; }

    /* ── SPINNER ── */
    .stSpinner > div { border-top-color: #3a7ef4 !important; }

    /* ── HIDE STREAMLIT CHROME ── */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: hidden; }

    /* ── CARD COMPONENT ── */
    .bb-card {
        background: rgba(255,255,255,0.025);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        position: relative;
        overflow: hidden;
        transition: border-color 0.25s ease;
    }
    .bb-card:hover { border-color: rgba(99,157,255,0.25); }
    .bb-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 3px; height: 100%;
        border-radius: 16px 0 0 16px;
    }
    .bb-card-blue::before  { background: linear-gradient(180deg,#3a7ef4,#1a56c4); }
    .bb-card-purple::before{ background: linear-gradient(180deg,#9b59b6,#6b4ef7); }
    .bb-card-orange::before{ background: linear-gradient(180deg,#ff7043,#e91e63); }

    .bb-card-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.6rem;
    }
    .bb-card-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 0.25rem;
    }
    .bb-card-unit {
        font-size: 0.8rem;
        color: #8da0c0;
        font-weight: 400;
    }
    .bb-card-sub {
        font-size: 0.8rem;
        color: #8da0c0;
        margin-top: 0.7rem;
        padding-top: 0.7rem;
        border-top: 1px solid rgba(255,255,255,0.06);
    }
    .bb-card-sub-value {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        color: #b0bfd8;
    }

    /* ── SECTION HEADER ── */
    .bb-section {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1.2rem;
    }
    .bb-section-num {
        width: 30px; height: 30px;
        background: linear-gradient(135deg,#3a7ef4,#6b4ef7);
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.8rem; font-weight: 700; color: #fff;
        flex-shrink: 0;
    }
    .bb-section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #dde3f0;
        letter-spacing: -0.01em;
    }
    .bb-section-sub {
        font-size: 0.8rem;
        color: #8da0c0;
        margin-left: auto;
        font-weight: 400;
    }

    /* ── HEADER BLOCK ── */
    .bb-header {
        background: linear-gradient(135deg, rgba(58,126,244,0.08) 0%, rgba(107,78,247,0.06) 100%);
        border: 1px solid rgba(99,157,255,0.1);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .bb-header-badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        background: rgba(58,126,244,0.15);
        border: 1px solid rgba(58,126,244,0.3);
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        color: #639dff;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }

    /* ── DECISION BOX ── */
    .bb-decision {
        border-radius: 16px;
        padding: 1.6rem;
        text-align: center;
        backdrop-filter: blur(8px);
    }
    .bb-decision-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 3.5rem;
        font-weight: 700;
        line-height: 1;
        margin: 0.5rem 0;
    }
    .bb-decision-label {
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-top: 0.6rem;
    }
    .bb-decision-thresholds {
        font-size: 0.7rem;
        color: rgba(255,255,255,0.4);
        margin-top: 1rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── METRICS TABLE ── */
    .bb-metrics-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.55rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        font-size: 0.85rem;
    }
    .bb-metrics-row:last-child { border-bottom: none; }
    .bb-metric-key  { color: #8da0c0; }
    .bb-metric-val  {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        color: #c8d4e8;
    }
    .bb-metric-note { font-size: 0.72rem; color: #4a6080; margin-left: 0.4rem; }

    /* ── SIDEBAR SECTION HEADER ── */
    .sb-header {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: #4a5568;
        margin: 1.2rem 0 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# MATPLOTLIB THEME
# ============================================================================
def setup_mpl_theme():
    plt.rcParams.update({
        'figure.facecolor':    '#0d1521',
        'axes.facecolor':      '#0f1b2d',
        'axes.edgecolor':      '#1e2d45',
        'axes.labelcolor':     '#8da0c0',
        'axes.titlecolor':     '#dde3f0',
        'axes.titlesize':      11,
        'axes.labelsize':      9.5,
        'axes.titleweight':    '600',
        'axes.grid':           True,
        'grid.color':          '#1a2a40',
        'grid.linewidth':      0.8,
        'xtick.color':         '#5a7090',
        'ytick.color':         '#5a7090',
        'xtick.labelsize':     8.5,
        'ytick.labelsize':     8.5,
        'text.color':          '#dde3f0',
        'font.family':         'DejaVu Sans',
        'font.size':           9.5,
        'lines.linewidth':     2.0,
        'axes.spines.top':     False,
        'axes.spines.right':   False,
        'axes.spines.left':    True,
        'axes.spines.bottom':  True,
        'figure.dpi':          120,
        'savefig.dpi':         120,
        'savefig.facecolor':   '#0d1521',
        'legend.facecolor':    '#0f1b2d',
        'legend.edgecolor':    '#1e2d45',
        'legend.framealpha':   0.9,
        'legend.fontsize':     8.5,
    })


# ============================================================================
# PHYSICS ENGINE
# ============================================================================
class PhysicsEngine:
    """Conversion piézoélectrique et transmission osseuse."""

    def __init__(self):
        self.mu_0      = 4 * np.pi * 1e-7
        self.eps_0     = 8.854e-12
        self.d33       = 2.31e-12      # C/N
        self.f_qz      = 0.32
        self.eps_eff   = 6.65 * self.eps_0
        self.sigma_skull  = 0.006
        self.sigma_brain  = 0.33
        self.thickness    = 0.007      # 7 mm

    def orientation_factor(self, odf):
        f_tex = 1 - np.exp(-(odf - 1) / 2)
        return f_tex * 0.978           # cos²(8.6°)

    def piezo_conversion(self, P_pa, odf, f_hz):
        F     = self.orientation_factor(odf)
        sigma = P_pa * 0.8 * F
        D3    = self.d33 * self.f_qz * (odf ** 2) * sigma * 0.7
        E     = D3 / self.eps_eff
        dDdt  = D3 * 2 * np.pi * f_hz
        B     = (self.mu_0 / (4 * np.pi)) * (dDdt * 1.0) / (0.3 ** 3)
        return E, B

    def skull_transmission(self, f_hz):
        w    = 2 * np.pi * f_hz
        Z_s  = np.sqrt(1j * w * self.mu_0 / self.sigma_skull)
        Z_b  = np.sqrt(1j * w * self.mu_0 / self.sigma_brain)
        G    = (Z_s - Z_b) / (Z_s + Z_b)
        delta = 0.12 * np.sqrt(40 / max(f_hz, 1))
        att  = np.exp(-self.thickness / delta)
        T    = np.abs(((1 - G ** 2) / (1 + G ** 2 * np.exp(-0.014 / delta))) * att)
        return T, delta

    def ellipse_gain(self, Q):
        return np.sqrt(Q * 2), 5.4


# ============================================================================
# NEURAL SIMULATION  (Kuramoto mean-field, N=10 000)
# ============================================================================
class NeuralSim:
    def __init__(self, N=10000):
        self.N  = N
        self.dt = 0.0005

    def kuramoto(self, f_hz, E_uv, noise_mv, T=2.0):
        n_steps = int(T / self.dt)
        t       = np.arange(0, T, self.dt)
        omega   = 2 * np.pi * np.random.normal(f_hz, 2.0, self.N)
        theta   = np.random.uniform(0, 2 * np.pi, self.N)
        I_em    = 0.5 * (E_uv * 1e-6) * 0.33 * 1e-4
        D       = (noise_mv * 1e-3) ** 2
        noise   = np.sqrt(2 * D * self.dt) * np.random.randn(n_steps, self.N)
        history = np.zeros((n_steps, self.N))

        for i in range(n_steps):
            history[i] = theta
            mean_p  = np.angle(np.mean(np.exp(1j * theta)))
            dtheta  = (omega
                       + 2.0 * np.sin(mean_p - theta)
                       + I_em * np.sin(2 * np.pi * f_hz * t[i])
                       + noise[i] / self.dt)
            theta = np.mod(theta + dtheta * self.dt, 2 * np.pi)

        return t, history

    def plv(self, history, f_hz):
        n      = history.shape[0]
        t      = np.arange(n) * self.dt
        target = 2 * np.pi * f_hz * t
        mean_th = np.angle(np.mean(np.exp(1j * history), axis=1))
        return np.abs(np.mean(np.exp(1j * (mean_th - target))))

    def sr_curve(self, E_uv, f_hz, noises):
        snrs = []
        for n in noises:
            _, h = self.kuramoto(f_hz, E_uv, n, T=1.5)
            p    = self.plv(h, f_hz)
            snrs.append(p ** 2 / (n ** 2 + 0.01))
        snrs = np.array(snrs)
        return noises, snrs, noises[np.argmax(snrs)], np.max(snrs)


# ============================================================================
# Ic ALGORITHM v2.0
# ============================================================================
class IcAlgorithm:
    def __init__(self):
        self.w       = [0.4, 0.4, 0.2]
        self.s_low   = 0.25
        self.s_high  = 0.65

    def coherence(self, s1, s2, fs=1000):
        n       = min(256, len(s1) // 4)
        f, P1   = welch(s1, fs, nperseg=n)
        f, P2   = welch(s2, fs, nperseg=n)
        f, Pc   = csd(s1, s2, fs, nperseg=n)
        g2      = np.abs(Pc) ** 2 / (P1 * P2 + 1e-20)
        w_snr   = 1 - np.exp(-P2 / (np.mean(P2[:10]) + 1e-10) / 10)
        return float(np.clip(np.mean(g2 * w_snr), 0, 1))

    def power_law(self, Press, Bfields):
        if len(Press) < 2 or np.any(Press <= 0):
            return 0.0, 0.0, 0.0
        lp, lb        = np.log(Press), np.log(Bfields)
        slope, _, r, p, _ = stats.linregress(lp, lb)
        if p >= 0.05:
            return 0.0, float(slope), float(r ** 2)
        qual = r ** 2 * np.exp(-abs(slope - 0.5) / 0.5)
        return float(np.clip(qual, 0, 1)), float(slope), float(r ** 2)

    def geo_factor(self, odf):
        x  = np.linspace(0, 10, 500)
        p  = stats.norm.pdf(x, odf, 0.5)
        p  = p / np.sum(p)
        pc = 1 / (1 + np.exp(-2 * (x - 3)))
        return float(np.clip(np.sum(pc * p) * 0.85, 0, 1))

    def fusion(self, C, T, G):
        scores  = [C, T, G]
        std     = float(np.std(scores))
        if std > 0.2:
            Ic       = np.mean([C, T]) * 0.5
            conflict = True
        else:
            Ic       = float(np.dot(self.w, scores))
            conflict = False
        Ic = float(np.clip(Ic, 0, 1))

        if Ic < self.s_low:
            return Ic, "H₀ VALIDÉE",   conflict, std, "#ff4d4d", "#2d0a0a"
        elif Ic < self.s_high:
            return Ic, "INDÉTERMINÉ",  conflict, std, "#ffa726", "#2d1b00"
        else:
            return Ic, "H₁ VALIDÉE",   conflict, std, "#00d48a", "#002d1b"


# ============================================================================
# HELPER: HTML COMPONENTS
# ============================================================================
def section_header(num: str, title: str, subtitle: str = ""):
    sub_html = f'<span class="bb-section-sub">{subtitle}</span>' if subtitle else ""
    return f"""
    <div class="bb-section">
        <div class="bb-section-num">{num}</div>
        <span class="bb-section-title">{title}</span>
        {sub_html}
    </div>"""


def phys_card(label: str, value: str, unit: str,
              sub_label: str, sub_value: str, sub_unit: str,
              color: str, css_class: str) -> str:
    return f"""
    <div class="bb-card {css_class}">
        <div class="bb-card-label" style="color:{color};">{label}</div>
        <div class="bb-card-value" style="color:{color};">{value}
            <span class="bb-card-unit">{unit}</span>
        </div>
        <div class="bb-card-sub">
            {sub_label}&nbsp;&nbsp;
            <span class="bb-card-sub-value">{sub_value}</span>
            <span style="font-size:0.72rem;color:#4a6080;"> {sub_unit}</span>
        </div>
    </div>"""


def metrics_table(rows: list) -> str:
    html = '<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:14px;padding:1rem 1.2rem;">'
    for key, val, note in rows:
        html += f"""
        <div class="bb-metrics-row">
            <span class="bb-metric-key">{key}</span>
            <span>
                <span class="bb-metric-val">{val}</span>
                <span class="bb-metric-note">{note}</span>
            </span>
        </div>"""
    html += "</div>"
    return html


def decision_box(Ic: float, decision: str, color: str, bg: str) -> str:
    return f"""
    <div class="bb-decision" style="background:{bg};border:1px solid {color}40;">
        <div style="font-size:0.7rem;font-weight:600;color:{color};text-transform:uppercase;letter-spacing:0.12em;opacity:0.8;">
            Indice Ic
        </div>
        <div class="bb-decision-value" style="color:{color};">{Ic:.3f}</div>
        <div class="bb-decision-label" style="color:{color};">{decision}</div>
        <div class="bb-decision-thresholds">seuils · 0.25 | 0.65</div>
    </div>"""


# ============================================================================
# MAIN
# ============================================================================
def main():
    inject_css()
    setup_mpl_theme()

    phys   = PhysicsEngine()
    neural = NeuralSim(N=10000)
    algo   = IcAlgorithm()

    # ── SIDEBAR ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="padding:0 0.2rem 1.5rem;">
            <div style="font-size:1.1rem;font-weight:700;color:#dde3f0;letter-spacing:-0.02em;">
                ◉ BARABAR
            </div>
            <div style="font-size:0.72rem;color:#4a5568;margin-top:0.15rem;font-family:'JetBrains Mono',monospace;">
                v2.1 · Algorithme Ic
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sb-header">Acoustique</div>', unsafe_allow_html=True)
        freq     = st.slider("Fréquence (Hz)",      30.0, 80.0, 40.0, 0.5)
        press_db = st.slider("Pression acoustique (dBSPL)", 80, 120, 102, 1)

        st.markdown('<div class="sb-header">Matériau</div>', unsafe_allow_html=True)
        odf = st.slider("Texture ODF",  1.0, 5.0, 4.2, 0.1)
        Q   = st.slider("Facteur Q",   20,  1000, 950, 10)

        st.markdown('<div class="sb-header">Neurophysique</div>', unsafe_allow_html=True)
        noise = st.slider("Bruit synaptique (mV)", 0.1, 3.0, 1.85, 0.05)

        st.markdown("---")
        run = st.button("▶  Lancer la simulation", type="primary", use_container_width=True)

        st.markdown("""
        <div style="margin-top:2rem;padding:1rem;background:rgba(255,255,255,0.02);
                    border:1px solid rgba(255,255,255,0.05);border-radius:10px;">
            <div style="font-size:0.68rem;color:#3a5a80;font-weight:600;
                        text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.6rem;">
                Chaîne causale
            </div>
            <div style="font-size:0.75rem;color:#5a7590;line-height:1.8;">
                Acoustique<br>
                <span style="color:#1e3a5a;margin-left:0.3rem;">↓</span><br>
                Piézoélectrique<br>
                <span style="color:#1e3a5a;margin-left:0.3rem;">↓</span><br>
                Transmission osseuse<br>
                <span style="color:#1e3a5a;margin-left:0.3rem;">↓</span><br>
                Neuromodulation<br>
                <span style="color:#1e3a5a;margin-left:0.3rem;">↓</span><br>
                Décision Ic
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── PRE-COMPUTE ─────────────────────────────────────────────────────────
    P_pa      = 20e-6 * (10 ** (press_db / 20))
    E_gran, B_gran = phys.piezo_conversion(P_pa, odf, freq)
    T_skull, delta = phys.skull_transmission(freq)
    E_cortex  = E_gran * T_skull
    p_gain, e_gain = phys.ellipse_gain(Q)
    E_foyer   = E_gran * e_gain

    # ── HEADER ──────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="bb-header">
        <div>
            <div class="bb-header-badge">Simulation Ic · {freq:.1f} Hz</div>
            <h1 style="margin:0;font-size:1.9rem;font-weight:700;
                        letter-spacing:-0.03em;color:#dde3f0;line-height:1.2;">
                Simulation BARABAR
                <span style="font-weight:300;color:#4a6080;"> v2.1</span>
            </h1>
            <p style="margin:0.4rem 0 0;color:#5a7090;font-size:0.85rem;">
                Modélisation bio-EM · Réseau de Kuramoto N=10 000 · Décision bayésienne
            </p>
        </div>
        <div style="text-align:right;font-family:'JetBrains Mono',monospace;
                    font-size:0.78rem;color:#2a3a50;line-height:2.0;display:none;">
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 1 — PHYSIQUE
    # ════════════════════════════════════════════════════════════════════════
    st.markdown(section_header("1", "Conversion Physique",
                               f"P = {P_pa*1e3:.2f} mPa · δ = {delta*1e3:.1f} mm"),
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(phys_card(
            "Champ Électromagnétique", f"{E_gran:.2f}", "µV/m",
            "Induction magnétique B", f"{B_gran*1e9:.2f}", "nT",
            "#4f8ef7", "bb-card-blue"
        ), unsafe_allow_html=True)

    with c2:
        st.markdown(phys_card(
            "Transmission Osseuse", f"{T_skull*100:.1f}", "%",
            "Champ au cortex E", f"{E_cortex:.3f}", "µV/m",
            "#a78bfa", "bb-card-purple"
        ), unsafe_allow_html=True)

    with c3:
        st.markdown(phys_card(
            "Focalisation Elliptique", f"×{e_gain:.1f}", "",
            "Champ au foyer E", f"{E_foyer:.2f}", "µV/m",
            "#fb923c", "bb-card-orange"
        ), unsafe_allow_html=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 2 — NEURONALE
    # ════════════════════════════════════════════════════════════════════════
    st.markdown(section_header("2", "Dynamique Neuronale",
                               "Réseau de Kuramoto · N = 10 000"),
                unsafe_allow_html=True)

    if run:
        with st.spinner("Simulation du réseau neuronal en cours…"):
            n_range                           = np.linspace(0.1, 3.0, 12)
            n_arr, snr_arr, n_opt, snr_max    = neural.sr_curve(E_cortex, freq, n_range)
            t, theta                          = neural.kuramoto(freq, E_cortex, noise, T=2.0)
            plv_val                           = neural.plv(theta, freq)

        left, right = st.columns([3, 1])

        with left:
            # ── SR curve ──
            fig, axes = plt.subplots(1, 2, figsize=(13, 4),
                                     gridspec_kw={'wspace': 0.35})
            ax = axes[0]
            ax.plot(n_arr, snr_arr, 'o-', color='#4f8ef7',
                    linewidth=2.2, markersize=6, markerfacecolor='#0f1b2d',
                    markeredgewidth=1.8, zorder=3)
            ax.fill_between(n_arr, snr_arr, alpha=0.12, color='#4f8ef7')
            ax.axvline(n_opt, color='#00d48a', linestyle='--',
                       linewidth=1.5, alpha=0.9, label=f'Optimal  {n_opt:.2f} mV')
            ax.axvline(noise, color='#ff6b6b', linestyle=':',
                       linewidth=1.5, alpha=0.8, label=f'Sélection  {noise:.2f} mV')
            ax.set_xlabel('Bruit synaptique (mV)')
            ax.set_ylabel('SNR')
            ax.set_title('Résonance Stochastique')
            ax.legend(frameon=True)
            ax.set_xlim(n_arr[0], n_arr[-1])

            # ── Phase distribution ──
            ax2 = axes[1]
            counts, bins, patches = ax2.hist(
                theta[-400:].flatten(), bins=36,
                density=True, edgecolor='none', alpha=0.0
            )
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            cmap_vals   = (bin_centers / (2 * np.pi))
            colors_grad = plt.cm.plasma(cmap_vals)
            for patch, color in zip(patches, colors_grad):
                patch.set_facecolor(color)
                patch.set_alpha(0.85)
            ax2.set_xlabel('Phase (rad)')
            ax2.set_ylabel('Densité')
            ax2.set_title(f'Distribution des phases  ·  PLV = {plv_val:.3f}')
            ax2.set_xlim(0, 2 * np.pi)
            ax2.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
            ax2.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with right:
            st.metric("Synchronisation PLV", f"{plv_val:.3f}",
                      help="0 = phases aléatoires · 1 = synchronisation parfaite")
            st.metric("Bruit optimal",   f"{n_opt:.2f} mV")
            st.metric("Gain SR max",     f"{snr_max:.2f} ×")

            st.markdown("<br>", unsafe_allow_html=True)

            if plv_val > 0.5:
                st.success("**Synchronisation forte**")
            elif plv_val > 0.3:
                st.info("**Synchronisation modérée**")
            else:
                st.error("**Synchronisation faible**")

    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem 2rem;
                    background:rgba(255,255,255,0.015);
                    border:1px dashed rgba(255,255,255,0.08);
                    border-radius:16px;color:#3a5a80;">
            <div style="font-size:2rem;margin-bottom:0.6rem;opacity:0.4;">◉</div>
            <div style="font-size:0.9rem;font-weight:500;">
                Ajustez les paramètres puis cliquez sur
                <strong style="color:#639dff;">▶ Lancer la simulation</strong>
            </div>
            <div style="font-size:0.78rem;margin-top:0.4rem;opacity:0.6;">
                Réseau de Kuramoto N=10 000 · ~5–10 s de calcul
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ════════════════════════════════════════════════════════════════════════
    # SECTION 3 — ALGORITHME Ic
    # ════════════════════════════════════════════════════════════════════════
    st.markdown(section_header("3", "Algorithme Ic v2.0",
                               "Fusion bayésienne · 3 sous-scores"),
                unsafe_allow_html=True)

    # compute scores
    fs     = 1000
    t_sig  = np.linspace(0, 2, int(fs * 2))
    s_ref  = np.sin(2 * np.pi * freq * t_sig)
    s_mag  = 0.8 * np.sin(2 * np.pi * freq * t_sig + 0.5) + 0.15 * np.random.randn(len(t_sig))

    C      = algo.coherence(s_ref, s_mag)
    P_test = np.linspace(P_pa * 0.6, P_pa * 1.4, 5)
    B_test = np.array([phys.piezo_conversion(p, odf, freq)[1] for p in P_test])
    T, alpha, r2 = algo.power_law(P_test, B_test)
    G      = algo.geo_factor(odf)
    Ic, decision, conflict, std_s, color, bg = algo.fusion(C, T, G)

    col_chart, col_metrics, col_ic = st.columns([3, 2, 2])

    # ── Bar chart ──
    with col_chart:
        fig, ax = plt.subplots(figsize=(7, 4))

        cats    = ['Cohérence\nSpectrale  γ²', 'Transfert\nPiézo  (√P)', 'Facteur\nGéologique']
        vals    = [C, T, G]
        palette = ['#4f8ef7', '#a78bfa', '#00d48a']

        bars = ax.bar(cats, vals, color=palette, alpha=0.85,
                      width=0.55, edgecolor='none', zorder=3)

        # gradient overlay via twin
        for bar, col_ in zip(bars, palette):
            ax.bar(bar.get_x() + bar.get_width() / 2,
                   bar.get_height(), width=bar.get_width(),
                   color=col_, alpha=0.15, zorder=2)

        # threshold lines
        ax.axhline(y=0.65, color='#00d48a', linestyle='--',
                   linewidth=1.2, alpha=0.6, zorder=1)
        ax.axhline(y=0.25, color='#ff6b6b', linestyle='--',
                   linewidth=1.2, alpha=0.6, zorder=1)
        ax.text(2.42, 0.66, 'H₁', fontsize=7.5, color='#00d48a', alpha=0.7, va='bottom')
        ax.text(2.42, 0.26, 'H₀', fontsize=7.5, color='#ff6b6b', alpha=0.7, va='bottom')

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.025,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=10, fontweight='700',
                    color='#dde3f0')

        if conflict:
            ax.text(0.5, 0.5, '⚠ CONFLIT DÉTECTÉ',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=13, color='#ffa726', alpha=0.4,
                    fontweight='700', rotation=12)

        ax.set_ylim(0, 1.12)
        ax.set_ylabel('Score normalisé')
        ax.set_title('Composition de l\'indice Ic')
        ax.tick_params(axis='x', which='both', bottom=False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Metrics table ──
    with col_metrics:
        st.markdown(metrics_table([
            ("γ² Cohérence spectrale",  f"{C:.4f}", ""),
            ("α Pente (cible 0.50)",    f"{alpha:+.3f}", ""),
            ("R² Qualité fit",          f"{r2:.4f}", ""),
            ("G Facteur ODF",           f"{G:.4f}", ""),
            ("σ Dispersion scores",     f"{std_s:.4f}",
             "⚠ conflit" if conflict else "✓ cohérent"),
            ("Poids fusion (C/T/G)",    "0.40 / 0.40 / 0.20", ""),
        ]), unsafe_allow_html=True)

    # ── Decision box ──
    with col_ic:
        st.markdown(decision_box(Ic, decision, color, bg),
                    unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(float(Ic))

        # Ic breakdown mini-bar
        st.markdown(f"""
        <div style="margin-top:1rem;font-size:0.72rem;color:#4a5568;
                    font-family:'JetBrains Mono',monospace;">
            <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem;">
                <span>C</span><span>T</span><span>G</span>
            </div>
            <div style="display:flex;gap:3px;height:6px;border-radius:4px;overflow:hidden;">
                <div style="flex:{C:.2f};background:#4f8ef7;border-radius:4px 0 0 4px;"></div>
                <div style="flex:{T:.2f};background:#a78bfa;"></div>
                <div style="flex:{G:.2f};background:#00d48a;border-radius:0 4px 4px 0;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
