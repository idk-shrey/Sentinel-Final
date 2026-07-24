"""
SENTINEL - Streamlit Demo App
Malware Classification with SHAP Explainability

HOW TO RUN:
    streamlit run SENTINEL_app.py

FILES NEEDED IN SAME FOLDER:
    sentinel_final_model.pkl
    sentinel_shap_explainer.pkl
    sentinel_shap_background.npy
    sentinel_shap_values.npy
    sentinel_shap_X.npy
    sentinel_shap_y.npy
    sentinel_sample_indices.json
    sentinel_all_results.json
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import joblib
import json
import time
import warnings
warnings.filterwarnings('ignore')

try:
    import ember_features
except ImportError:
    ember_features = None
warnings.filterwarnings('ignore')

# ===================================================================
#  PAGE CONFIG
# ===================================================================
st.set_page_config(
    page_title="SENTINEL",
    page_icon="\U0001f6e1\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Google Font - loaded via separate call for reliability
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

# Main theme CSS - NOTE: avoid CSS ">" child selectors as st.markdown parses them as HTML
st.markdown("""<style>
html, body, .main, .block-container, .stApp,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stMainBlockContainer"],
[data-testid="stVerticalBlock"],
[data-testid="stBottom"],
div.appview-container {
    background-color: #EFF6FB !important;
    color: #1B2A3D !important;
}
section[data-testid="stSidebar"] {
    background-color: #F4F9FD !important;
    border-right: 1px solid #CDDEED !important;
}
#MainMenu, footer { visibility: hidden !important; display: none !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #B3D8F0; border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #4DB6F0; }
</style>""", unsafe_allow_html=True)

# Component styles
st.markdown("""<style>
.hero {
    background: linear-gradient(135deg, #00C6FF 0%, #0072FF 100%);
    border-radius: 20px; padding: 2.5rem 2.5rem;
    margin-bottom: 2rem; position: relative; overflow: hidden;
    box-shadow: 0 15px 35px rgba(0, 114, 255, 0.25), 0 5px 15px rgba(0, 114, 255, 0.15);
    border: 1px solid rgba(255, 255, 255, 0.2);
}
.hero::after {
    content:''; position:absolute; top:-30%; right:-10%;
    width:400px; height:400px; border-radius:50%;
    background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 60%);
    pointer-events: none;
}
.hero::before {
    content:''; position:absolute; bottom:-40%; left:-10%;
    width:300px; height:300px; border-radius:50%;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
    pointer-events: none;
}
.hero h1 { color:#ffffff !important; font-size:2.8rem !important; font-weight:900 !important; margin:0 !important; letter-spacing:-0.5px !important; position:relative; text-shadow: 0 2px 10px rgba(0,0,0,0.15) !important; }
.hero p  { color:rgba(255,255,255,0.95) !important; font-size:1.15rem !important; margin:.5rem 0 0 !important; font-weight:500 !important; position:relative; text-shadow: 0 1px 5px rgba(0,0,0,0.1) !important;}
.hero .sub { color:rgba(255,255,255,0.75) !important; font-size:.9rem !important; margin-top:.4rem !important; font-weight:400 !important; position:relative; }
.stat-row { display:flex; gap:.8rem; margin-bottom:1.5rem; }
.stat-card {
    flex:1; background:#FFFFFF; border:1px solid #E2E8F0; border-radius:16px;
    padding:1.3rem 1rem; text-align:center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    transition: transform .3s ease, box-shadow .3s ease, border-color .3s ease;
}
.stat-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 30px rgba(0, 114, 255, 0.15);
    border-color: #00C6FF;
}
.stat-val { color:#0072FF; font-size:1.6rem; font-weight:900; }
.stat-lbl { color:#64748B; font-size:.75rem; margin-top:5px; font-weight:600; letter-spacing:1px; text-transform:uppercase; }
.m-card {
    background: #FFFFFF; border-radius: 16px; padding: 1.2rem 1.4rem;
    border: none; margin-bottom: .8rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04), 0 1px 3px rgba(0,0,0,0.02);
    transition: transform .25s ease, box-shadow .25s ease;
}
.m-card:hover { 
    transform: translateY(-3px); 
    box-shadow: 0 8px 25px rgba(0, 114, 255, 0.12); 
}
.m-lbl { color:#6B8FAD; font-size:.68rem; font-weight:600; text-transform:uppercase; letter-spacing:.7px; }
.m-val { color:#01579B; font-size:1.35rem; font-weight:700; margin-top:2px; }
</style>""", unsafe_allow_html=True)

# Verdict, badge, and utility styles
st.markdown("""<style>
.verdict-mal {
    background: linear-gradient(135deg, #FFF0F0 0%, #FFE0E0 100%);
    border: 1px solid #F9A8A8; border-left: 5px solid #EF4444;
    border-radius: 14px; padding: 1.8rem; text-align: center;
    box-shadow: 0 4px 16px rgba(239,68,68,0.08);
}
.verdict-ben {
    background: linear-gradient(135deg, #F0FFF4 0%, #DCFCE7 100%);
    border: 1px solid #86EFAC; border-left: 5px solid #22C55E;
    border-radius: 14px; padding: 1.8rem; text-align: center;
    box-shadow: 0 4px 16px rgba(34,197,94,0.08);
}
.v-title { font-size:2rem; font-weight:900; color:#1B2A3D; margin:0; letter-spacing:-.3px; }
.v-conf  { font-size:.95rem; color:#5A6B7F; margin-top:.4rem; }
.pill { display:inline-block; padding:5px 18px; border-radius:99px; font-size:.75rem; font-weight:700; letter-spacing:.4px; }
.pill-crit { background:#FEE2E2; color:#991B1B; }
.pill-high { background:#FEF3C7; color:#92400E; }
.pill-med  { background:#FEF9C3; color:#854D0E; }
.pill-low  { background:#DCFCE7; color:#166534; }
.info-box {
    background:#E8F4FD; border:1px solid #B3D8F0;
    border-left:4px solid #29B6F6; border-radius:10px;
    padding:.9rem 1.1rem; color:#1B2A3D; font-size:.92rem; margin-bottom:1rem;
}
.report-box {
    background:#F8FBFE; border:1px solid #D4E6F1; border-radius:12px;
    padding:1.3rem 1.5rem; font-family:'Consolas','SFMono-Regular',monospace !important;
    font-size:.8rem; color:#2C3E50; white-space:pre-wrap; line-height:1.65;
}
.conf-track { background:#E1F0FA; border-radius:99px; height:12px; overflow:hidden; margin:.4rem 0 1rem; }
.conf-fill  { height:100%; border-radius:99px; transition: width .5s ease; }
.sec-lbl { color:#6B8FAD; font-size:.7rem; font-weight:600; text-transform:uppercase; letter-spacing:.7px; margin:1.2rem 0 .6rem; }
</style>""", unsafe_allow_html=True)

# Widget and interaction styles
st.markdown("""<style>
.stButton button,
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #00C6FF, #0072FF) !important;
    color: #ffffff !important; border: none !important;
    border-radius: 12px !important; font-weight: 700 !important;
    padding: .6rem 1.5rem !important; font-size: .95rem !important;
    box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3) !important;
    transition: all .3s ease !important; letter-spacing: .3px !important;
}
.stButton button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    box-shadow: 0 8px 25px rgba(0, 114, 255, 0.45) !important;
    transform: translateY(-2px) !important;
}
.stButton button:active { transform: translateY(0) !important; }
.stDownloadButton button {
    background: #FFFFFF !important; color: #0072FF !important;
    border: 2px solid #00C6FF !important; border-radius: 12px !important;
    font-weight: 700 !important; transition: all .3s ease !important;
}
.stDownloadButton button:hover {
    background: #F0FBFF !important;
    box-shadow: 0 6px 20px rgba(0, 198, 255, 0.2) !important;
    transform: translateY(-2px) !important;
}
.stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 2px solid #D4E6F1; }
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #6B8FAD !important;
    font-weight: 500 !important; padding: .65rem 1.1rem !important;
    transition: all .15s ease !important; border-radius: 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #0277BD !important;
    font-weight: 700 !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #29B6F6 !important;
    height: 3px !important;
}
div.stMarkdown p, div.stText p { color: #2C3E50 !important; }
div.stMarkdown h1, div.stMarkdown h2, div.stMarkdown h3 { color: #1B2A3D !important; font-weight: 700 !important; }
.stRadio label { color: #1B2A3D !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] { background: #29B6F6 !important; }
.stAlert { border-radius: 10px !important; }
hr { border-color: #D4E6F1 !important; opacity: .4; }
@keyframes fadeUp {
    from { opacity:0; transform:translateY(10px); }
    to   { opacity:1; transform:translateY(0); }
}
.hero, .stat-row, .m-card, .verdict-mal, .verdict-ben {
    animation: fadeUp .35s ease forwards;
}
.footer-text { color: #8FA7BC; font-size: .78rem; }
.footer-text b { color: #5A7A94; }
</style>""", unsafe_allow_html=True)

# ===================================================================
#  CONSTANTS
# ===================================================================
FEATURE_GROUPS = {
    range(0, 256):    'Byte Histogram',
    range(256, 512):  'Byte Entropy',
    range(512, 556):  'String Features',
    range(556, 563):  'General File Info',
    range(563, 575):  'Header Info',
    range(575, 611):  'Section Info',
    range(611, 745):  'Import Features',
    range(745, 2381): 'Export & Other',
}


def get_group(i):
    for r, n in FEATURE_GROUPS.items():
        if i in r:
            return n
    return 'Other'


N_FEATURES = 2381
feature_names = [f'Feature_{i}' for i in range(N_FEATURES)]

# ===================================================================
#  DATA LOADING
# ===================================================================
@st.cache_resource
def load_all():
    m   = joblib.load('sentinel_final_model.pkl')
    exp = joblib.load('sentinel_shap_explainer.pkl')
    bg  = np.load('sentinel_shap_background.npy')
    sv  = np.load('sentinel_shap_values.npy')
    sX  = np.load('sentinel_shap_X.npy')
    sy  = np.load('sentinel_shap_y.npy')
    with open('sentinel_sample_indices.json') as f:
        idx = json.load(f)
    with open('sentinel_all_results.json') as f:
        res = json.load(f)
    return m, exp, bg, sv, sX, sy, idx, res

# ===================================================================
#  HELPER FUNCTIONS
# ===================================================================
def risk(p):
    if p >= .9:
        return 'CRITICAL'
    if p >= .7:
        return 'HIGH'
    if p >= .5:
        return 'MEDIUM'
    return 'LOW'


def risk_color(r):
    return {
        'CRITICAL': '#EF4444',
        'HIGH': '#F59E0B',
        'MEDIUM': '#EAB308',
        'LOW': '#22C55E',
    }[r]


def risk_pill_class(r):
    return {
        'CRITICAL': 'pill-crit',
        'HIGH': 'pill-high',
        'MEDIUM': 'pill-med',
        'LOW': 'pill-low',
    }[r]


def classify(sample, mdl, exp):
    x = sample.reshape(1, -1)
    prob = mdl.predict_proba(x)[0][1]
    verdict = 'MALICIOUS' if prob >= .5 else 'BENIGN'
    r = risk(prob)
    sv = exp.shap_values(x)[0]
    return prob, verdict, r, sv


def make_waterfall(sv, sample, base_value, title, n=15):
    explanation = shap.Explanation(
        values=sv, base_values=base_value,
        data=sample, feature_names=feature_names,
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor('#F5FAFD')
    ax.set_facecolor('#F5FAFD')
    
    # Workaround for SHAP + Matplotlib 3.8+ bug (tick labels not populated before draw)
    old_get = ax.yaxis.get_majorticklabels
    def new_get():
        fig.canvas.draw()
        return old_get()
    ax.yaxis.get_majorticklabels = new_get
    
    shap.waterfall_plot(explanation, max_display=n, show=False)
    plt.title(title, fontsize=13, fontweight='bold', pad=14, color='#1B2A3D')
    for t in ax.texts:
        t.set_color('#2C3E50')
    ax.tick_params(colors='#2C3E50')
    for spine in ax.spines.values():
        spine.set_color('#CDDEED')
    plt.tight_layout()
    return fig


def make_bar_chart(sv):
    gs = {}
    for i, v in enumerate(sv):
        g = get_group(i)
        gs[g] = gs.get(g, 0) + abs(v)
    sg = sorted(gs.items(), key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#F5FAFD')
    ax.set_facecolor('#F5FAFD')
    colors = ['#0288D1' if i == 0 else '#B3E5FC' for i in range(len(sg))]
    ax.barh(
        [g for g, _ in sg][::-1],
        [v for _, v in sg][::-1],
        color=colors[::-1], edgecolor='none', height=0.55,
    )
    ax.set_xlabel('Total |SHAP|', color='#2C3E50', fontsize=11)
    ax.tick_params(colors='#2C3E50', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CDDEED')
    ax.spines['bottom'].set_color('#CDDEED')
    plt.tight_layout()
    return fig


def build_report(prob, verdict, r, sv):
    tp = np.argsort(sv)[::-1][:5]
    tn = np.argsort(sv)[:5]
    lines = [
        '=' * 60,
        '     SENTINEL THREAT INTELLIGENCE REPORT',
        '=' * 60,
        '  Verdict    : ' + verdict,
        '  Risk Level : ' + r,
        '  Confidence : {:.2f}% malicious  |  {:.2f}% benign'.format(
            prob * 100, (1 - prob) * 100),
        '-' * 60,
        '  INDICATORS -> MALICIOUS:',
    ]
    for i, fi in enumerate(tp, 1):
        name = feature_names[fi]
        group = get_group(fi)
        val = sv[fi]
        lines.append('    {}. {:<18}| {:<22}| +{:.5f}'.format(i, name, group, val))
    lines += ['', '  INDICATORS -> BENIGN:']
    for i, fi in enumerate(tn, 1):
        name = feature_names[fi]
        group = get_group(fi)
        val = sv[fi]
        lines.append('    {}. {:<18}| {:<22}| {:.5f}'.format(i, name, group, val))
    lines += ['-' * 60, '  ANALYST RECOMMENDATION:']
    if verdict == 'MALICIOUS' and r in ('CRITICAL', 'HIGH'):
        lines.append('  >> Quarantine immediately. Do not execute.')
        lines.append('  >> Escalate to incident response team.')
    elif verdict == 'MALICIOUS':
        lines.append('  >> Flag for manual review.')
    else:
        lines.append('  >> File appears safe. Continue normal processing.')
    lines.append('=' * 60)
    return '\n'.join(lines)


# ===================================================================
#  LOAD DATA
# ===================================================================
try:
    model, explainer, bg, shap_vals, shap_X, shap_y, indices, results = load_all()
except Exception as e:
    st.error('Could not load data files: ' + str(e))
    st.stop()

final = results.get('final_model', {})
acc  = final.get('accuracy',  0.972)
rec  = final.get('recall',    0.970)
prec = final.get('precision', 0.973)
auc  = final.get('auc_roc',   0.997)
shap_proba = model.predict_proba(shap_X)[:, 1]

# ===================================================================
#  SIDEBAR
# ===================================================================
with st.sidebar:
    st.markdown(
        '<div style="text-align:center; padding:1.2rem 0 .6rem">'
        '<div style="width:54px; height:54px; margin:0 auto; border-radius:16px;'
        ' background:#FFFFFF;'
        ' display:flex; align-items:center; justify-content:center;'
        ' box-shadow:0 8px 24px rgba(0, 114, 255, 0.15); font-size:1.6rem">'
        '\U0001f6e1\ufe0f</div>'
        '<div style="color:#1B2A3D; font-size:1.25rem; font-weight:800;'
        ' letter-spacing:1px; margin-top:.5rem">SENTINEL</div>'
        '<div style="color:#6B8FAD; font-size:.7rem; margin-top:2px; line-height:1.5">'
        'Static Executable Analysis<br>ML-Powered Threat Intelligence</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('---')

    st.markdown('##### \u2699\ufe0f Input Mode')
    mode = st.radio(
        'Select input mode',
        ['\U0001f3af Demo Samples', '\U0001f4c4 Upload PE (.exe, .dll)', '\U0001f4c1 Upload .npy', '\U0001f39a\ufe0f Manual Sliders'],
        label_visibility='collapsed',
    )

    st.markdown('---')

    st.markdown('##### \U0001f4ca Model Performance')
    for label, val, fmt in [
        ('Accuracy', acc, '{:.2%}'),
        ('Recall', rec, '{:.2%}'),
        ('Precision', prec, '{:.2%}'),
        ('AUC-ROC', auc, '{:.4f}'),
    ]:
        st.markdown(
            '<div class="m-card"><div class="m-lbl">{}</div>'
            '<div class="m-val">{}</div></div>'.format(label, fmt.format(val)),
            unsafe_allow_html=True,
        )

    st.markdown('---')

    st.markdown(
        '<div style="color:#6B8FAD; font-size:.76rem; line-height:1.8">'
        '<b style="color:#1B2A3D">Dataset:</b> EMBER 2018<br>'
        '<b style="color:#1B2A3D">Training:</b> 600k+ PE samples<br>'
        '<b style="color:#1B2A3D">Algorithm:</b> XGBoost (tuned)<br>'
        '<b style="color:#1B2A3D">Features:</b> 2,381 static PE features<br>'
        '<b style="color:#1B2A3D">Approach:</b> Pre-execution static analysis'
        '</div>',
        unsafe_allow_html=True,
    )

# ===================================================================
#  HERO BANNER
# ===================================================================
st.markdown(
    '<div class="hero">'
    '<h1 style="display: flex; align-items: center; gap: 14px;">'
    '<svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="filter: drop-shadow(0 2px 6px rgba(0,0,0,0.15));">'
    '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>'
    'SENTINEL</h1>'
    '<p>Static Executable Analysis Using ML for Threat Intelligence</p>'
    '<div class="sub">Pre-execution malware classification with SHAP '
    'explainability \u2014 no file execution required</div>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="stat-row">'
    '<div class="stat-card"><div class="stat-val">{:.1%}</div>'
    '<div class="stat-lbl">Accuracy</div></div>'
    '<div class="stat-card"><div class="stat-val">600k+</div>'
    '<div class="stat-lbl">Training Samples</div></div>'
    '<div class="stat-card"><div class="stat-val">2,381</div>'
    '<div class="stat-lbl">PE Features</div></div>'
    '<div class="stat-card"><div class="stat-val">{:.4f}</div>'
    '<div class="stat-lbl">AUC-ROC</div></div>'
    '<div class="stat-card"><div class="stat-val">0 ms</div>'
    '<div class="stat-lbl">Execution Needed</div></div>'
    '</div>'.format(acc, auc),
    unsafe_allow_html=True,
)

# ===================================================================
#  INPUT SECTION
# ===================================================================
st.markdown('## \U0001f50e Analyze a File')
sample = None

if mode == '\U0001f3af Demo Samples':
    st.markdown(
        '<div class="info-box">\U0001f4cc <b>Demo mode</b> \u2014 real EMBER '
        'validation samples (features only, no live malware).</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button('\U0001f534 High-Confidence Malware'):
            st.session_state.update({
                'sample': shap_X[indices['best_malware']],
                'psv':    shap_vals[indices['best_malware']],
                'pprob':  float(shap_proba[indices['best_malware']]),
            })
    with c2:
        if st.button('\U0001f7e2 High-Confidence Benign'):
            st.session_state.update({
                'sample': shap_X[indices['best_benign']],
                'psv':    shap_vals[indices['best_benign']],
                'pprob':  float(shap_proba[indices['best_benign']]),
            })
    with c3:
        if st.button('\U0001f7e1 Borderline Case'):
            st.session_state.update({
                'sample': shap_X[indices['borderline']],
                'psv':    shap_vals[indices['borderline']],
                'pprob':  float(shap_proba[indices['borderline']]),
            })
    if 'sample' in st.session_state:
        sample = st.session_state['sample']

elif mode == '\U0001f4c1 Upload .npy':
    st.markdown(
        '<div class="info-box">\U0001f4cc Upload a <b>.npy file</b> with a '
        '2381-dimensional EMBER feature vector.</div>',
        unsafe_allow_html=True,
    )
    up = st.file_uploader('Upload feature vector (.npy)', type=['npy'])
    if up:
        try:
            sample = np.load(up)
            if sample.shape[0] != N_FEATURES:
                st.error('Expected {} features, got {}.'.format(N_FEATURES, sample.shape[0]))
                sample = None
            else:
                st.success('Loaded \u2014 {} features'.format(N_FEATURES))
                for k in ('psv', 'pprob'):
                    st.session_state.pop(k, None)
        except Exception as e:
            st.error('Error: ' + str(e))

elif mode == '\U0001f4c4 Upload PE (.exe, .dll)':
    st.markdown(
        '<div class="info-box">\U0001f4cc Upload a raw <b>Windows Executable</b> (.exe, .dll). '
        'Features will be extracted on the fly.</div>',
        unsafe_allow_html=True,
    )
    if ember_features is None:
        st.error("The `ember_features` module or `lief` library is missing. PE extraction disabled.")
    else:
        up_pe = st.file_uploader('Upload Executable (.exe, .dll)', type=['exe', 'dll', 'bin', 'sys'])
        if up_pe:
            try:
                with st.spinner("Extracting features using LIEF..."):
                    bytez = up_pe.read()
                    extractor = ember_features.PEFeatureExtractor(2, print_feature_warning=False)
                    sample = np.array(extractor.feature_vector(bytez), dtype=np.float32)
                    st.success('Extracted 2381 features successfully!')
                    for k in ('psv', 'pprob'):
                        st.session_state.pop(k, None)
            except Exception as e:
                st.error('Error parsing PE file: ' + str(e))

elif mode == '\U0001f39a\ufe0f Manual Sliders':
    st.markdown(
        '<div class="info-box">\U0001f4cc Set PE file feature values manually '
        'to explore how each affects the verdict.</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        be  = st.slider('Byte Entropy',   0.0, 1.0, 0.5, 0.01,
                         help='High = packed/encrypted = often malware')
        id_ = st.slider('Import Density', 0.0, 1.0, 0.3, 0.01,
                         help='Unusual imports suggest malice')
    with c2:
        se = st.slider('Section Entropy', 0.0, 1.0, 0.4, 0.01,
                        help='Abnormal sections = obfuscation')
        sf = st.slider('String Score',    0.0, 1.0, 0.2, 0.01,
                        help='Suspicious strings in file')
    with c3:
        ha = st.slider('Header Anomaly',  0.0, 1.0, 0.1, 0.01,
                        help='Unusual PE header values')
    s = shap_X[indices['borderline']].copy()
    b = shap_X[indices['best_benign']].copy()
    m = shap_X[indices['best_malware']].copy()
    
    # Interpolate sections between benign (0.0) and malicious (1.0)
    s[256:512] = b[256:512] + (m[256:512] - b[256:512]) * be
    s[611:745] = b[611:745] + (m[611:745] - b[611:745]) * id_
    s[575:611] = b[575:611] + (m[575:611] - b[575:611]) * se
    s[512:556] = b[512:556] + (m[512:556] - b[512:556]) * sf
    s[563:575] = b[563:575] + (m[563:575] - b[563:575]) * ha
    st.session_state['sample'] = s
    for k in ('psv', 'pprob'):
        st.session_state.pop(k, None)
    if 'sample' in st.session_state:
        sample = st.session_state['sample']

# ===================================================================
#  RESULTS
# ===================================================================
if sample is not None:
    st.markdown('---')

    with st.spinner('\U0001f50d Running SENTINEL analysis...'):
        if 'psv' in st.session_state and 'pprob' in st.session_state:
            prob = st.session_state['pprob']
            sv   = st.session_state['psv']
            verdict = 'MALICIOUS' if prob >= .5 else 'BENIGN'
            r = risk(prob)
        else:
            prob, verdict, r, sv = classify(sample, model, explainer)

    if verdict == 'MALICIOUS':
        bc = '#EF4444'
        icon = '\U0001f6a8'
        vc = 'verdict-mal'
    else:
        bc = '#22C55E'
        icon = '\u2705'
        vc = 'verdict-ben'

    rc = risk_color(r)
    pill_cls = risk_pill_class(r)

    st.markdown('## \U0001f4ca Analysis Results')
    col1, col2 = st.columns([1.6, 2.4])

    with col1:
        st.markdown(
            '<div class="{vc}">'
            '<div class="v-title">{icon} {verdict}</div>'
            '<div class="v-conf">{conf:.2f}% confidence</div>'
            '<div style="margin-top:.7rem">'
            '<span class="pill {pill}">{r} RISK</span></div>'
            '</div>'.format(
                vc=vc, icon=icon, verdict=verdict,
                conf=prob * 100, pill=pill_cls, r=r),
            unsafe_allow_html=True,
        )

        st.markdown('<br>', unsafe_allow_html=True)

        st.markdown(
            '<div class="m-card">'
            '<div class="m-lbl">Malicious Probability</div>'
            '<div class="m-val" style="color:{bc}">{val:.1f}%</div>'
            '</div>'
            '<div class="m-card">'
            '<div class="m-lbl">Benign Probability</div>'
            '<div class="m-val">{ben:.1f}%</div>'
            '</div>'
            '<div class="m-card">'
            '<div class="m-lbl">Risk Level</div>'
            '<div class="m-val" style="color:{rc}">{r}</div>'
            '</div>'.format(
                bc=bc, val=prob * 100, ben=(1 - prob) * 100, rc=rc, r=r),
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="sec-lbl">Confidence Meter</div>'
            '<div class="conf-track">'
            '<div class="conf-fill" style="width:{w:.1f}%; background:{bc}"></div>'
            '</div>'.format(w=prob * 100, bc=bc),
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sec-lbl">Analyst Recommendation</div>',
                    unsafe_allow_html=True)
        if verdict == 'MALICIOUS' and r in ('CRITICAL', 'HIGH'):
            st.error('\u26a0\ufe0f Quarantine immediately. Escalate to IR team.')
        elif verdict == 'MALICIOUS':
            st.warning('\u26a0\ufe0f Flag for manual review.')
        else:
            st.success('\u2705 File appears safe. Continue normal processing.')

    with col2:
        t1, t2, t3 = st.tabs([
            '\U0001f50d SHAP Explanation',
            '\U0001f4cb Threat Report',
            '\U0001f4ca Feature Groups',
        ])

        with t1:
            st.markdown('**What drove this prediction?**')
            st.caption(
                'Red = toward malicious \u00b7 Blue = toward benign '
                '\u00b7 Longer = stronger influence')
            with st.spinner('Generating explanation...'):
                fig = make_waterfall(
                    sv, sample, explainer.expected_value,
                    'SENTINEL \u2014 {} ({:.1f}%)'.format(verdict, prob * 100))
            st.pyplot(fig)
            plt.close()

        with t2:
            st.markdown('**Full threat intelligence report**')
            rep = build_report(prob, verdict, r, sv)
            import html
            safe_rep = html.escape(rep).replace('\n', '<br>')
            st.markdown(
                '<div class="report-box">{}</div>'.format(safe_rep),
                unsafe_allow_html=True,
            )
            st.download_button(
                '\U0001f4e5 Download Report',
                data=rep,
                file_name='sentinel_{}.txt'.format(int(time.time())),
                mime='text/plain',
            )

        with t3:
            st.markdown('**Which PE section contributed most?**')
            fig2 = make_bar_chart(sv)
            st.pyplot(fig2)
            plt.close()

# ===================================================================
#  FOOTER
# ===================================================================
st.markdown('---')
fc1, fc2, fc3, fc4 = st.columns(4)
footer_items = [
    '\U0001f6e1\ufe0f <b>SENTINEL</b> \u2014 Malware Classification',
    '\u26a1 XGBoost + SHAP Explainability',
    '\U0001f4ca EMBER Dataset \u2014 600k+ samples',
    '\U0001f52c Static analysis \u2014 zero execution',
]
for col, txt in zip([fc1, fc2, fc3, fc4], footer_items):
    col.markdown(
        '<span class="footer-text">{}</span>'.format(txt),
        unsafe_allow_html=True,
    )
