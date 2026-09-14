# ============================================================================
# streamlit_app.py — FULL CORRECTED VERSION
# Fixes KeyError: 'FeaturePipeline' in _inject_into_main()
# ============================================================================

import warnings
warnings.filterwarnings("ignore")
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

import streamlit as st
import json
import pandas as pd
import numpy as np
import os
import re
import glob
import sys
import joblib
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix


# ============================================================================
# TOKENIZER
# ============================================================================
class GujaratiTokenizer:
    GUJ = re.compile(r'[\u0A80-\u0AFF]+')
    ENG = re.compile(r'[a-zA-Z]+')
    NUM = re.compile(r'[0-9]+')

    @classmethod
    def words(cls, text):
        if not text or not isinstance(text, str):
            return []
        return cls.GUJ.findall(text) + cls.ENG.findall(text) + cls.NUM.findall(text)

    @classmethod
    def sentences(cls, text):
        if not text or not isinstance(text, str):
            return []
        t = text.replace('।', '.').replace('?', '.').replace('!', '.')
        return [s.strip() for s in re.split(r'(?<=[.])\s+', t) if len(s.strip()) > 2]


# ============================================================================
# STYLE MATRIX EXTRACTOR — produces 41 features
# ============================================================================
class StyleMatrixExtractor:
    def __init__(self):
        self.tk = GujaratiTokenizer()
        self.v_markers = [
            'તથા', 'વળી', 'આથી', 'ગણાય', 'પ્રચલિત', 'આવાં', 'કેટલાંક', 'અલબત્ત',
            'તદુપરાંત', 'દા.ત.', 'દા. ત.', 'જુઓ', 'એટલે કે', 'કહેવાય છે', 'તેમજ',
            'ઉપરાંત', 'વિશેષ', 'એટલે', 'કહેવાય', 'કરાય છે', 'થાય છે', 'ઓળખાય છે',
            'ગણાય છે', 'સ્વયંસંચાલિત', 'અંકીય', 'ગણનયંત્ર', 'ભૌતિકવિજ્ઞાન'
        ]
        self.w_markers = [
            'શામેલ', 'ઘણીવાર', 'કોઈપણ', 'વ્યાખ્યાયિત', 'ઉદાહરણ તરીકે', 'મોડેલ',
            'સોફ્ટવેર', 'મુખ્ય લેખ', 'આ પણ જુઓ', 'જો કે', 'દ્વારા', 'સંદર્ભ',
            'બાહ્ય કડીઓ', 'સ્રોત', 'ટીકા', 'વિવાદ', 'સક્ષમ', 'સમાવેશ', 'ઉલ્લેખ',
            'પ્રોગ્રામ', 'ક્લસ્ટર', 'કરવામાં આવે છે', 'આપવામાં આવે છે',
            'બનાવવામાં આવે છે', 'માનવામાં આવે છે', 'હતું', 'હતા', 'હતી',
            'એપ્રિલ', 'મે', 'જૂન', 'ઓગસ્ટ', 'નવેમ્બર', 'ડિસેમ્બર', 'તારીખ'
        ]
        self.traditional_translit = ['ૉ', 'ૅ', 'ઑ', 'ઍ']
        self.modern_translit = ['ો', 'ે', 'ૈ']

    def extract(self, text: str) -> dict:
        if not text or not isinstance(text, str) or len(text) < 20:
            return self._empty()
        words = self.tk.words(text)
        sentences = self.tk.sentences(text)
        if len(words) < 5:
            return self._empty()

        wc = len(words)
        cc = len(text)
        sc = max(len(sentences), 1)

        feats = {
            'word_count': wc,
            'log_word_count': float(np.log1p(wc)),
            'char_count': cc,
            'log_char_count': float(np.log1p(cc)),
            'sentence_count': sc,
            'avg_word_length': float(np.mean([len(w) for w in words])) if words else 0.0,
        }

        sl = [len(self.tk.words(s)) for s in sentences]
        sl = [l for l in sl if l > 0]
        feats['avg_sentence_length'] = float(np.mean(sl)) if sl else 0.0
        feats['std_sentence_length'] = float(np.std(sl)) if len(sl) > 1 else 0.0
        feats['max_sentence_length'] = float(max(sl)) if sl else 0.0
        feats['min_sentence_length'] = float(min(sl)) if sl else 0.0

        uniq = set(words)
        feats['type_token_ratio'] = len(uniq) / wc if wc > 0 else 0.0
        feats['hapax_ratio'] = sum(1 for c in Counter(words).values() if c == 1) / max(len(uniq), 1)

        v_c = sum(text.count(m) for m in self.v_markers)
        w_c = sum(text.count(m) for m in self.w_markers)
        feats['v_markers_per_1000'] = (v_c / wc) * 1000 if wc > 0 else 0.0
        feats['w_markers_per_1000'] = (w_c / wc) * 1000 if wc > 0 else 0.0
        feats['marker_diff_per_1000'] = ((v_c - w_c) / wc) * 1000 if wc > 0 else 0.0
        feats['marker_ratio'] = v_c / (v_c + w_c) if (v_c + w_c) > 0 else 0.5

        passive_markers = ['થાય છે', 'થયું', 'થયા', 'થઈ', 'આવે છે', 'આવ્યું',
                           'બને છે', 'કરાય છે']
        p_c = sum(text.count(m) for m in passive_markers)
        feats['passive_per_1000'] = (p_c / wc) * 1000 if wc > 0 else 0.0

        eng_chars = len(re.findall(r'[a-zA-Z]', text))
        guj_chars = len(re.findall(r'[\u0A80-\u0AFF]', text))
        feats['english_char_ratio'] = eng_chars / cc if cc > 0 else 0.0
        feats['gujarati_char_ratio'] = guj_chars / cc if cc > 0 else 0.0
        feats['script_ratio'] = guj_chars / (guj_chars + eng_chars + 1)

        feats['colon_per_1000'] = (text.count(':') / wc) * 1000 if wc > 0 else 0.0
        feats['comma_per_1000'] = (text.count(',') / wc) * 1000 if wc > 0 else 0.0
        feats['paren_per_1000'] = ((text.count('(') + text.count(')')) / wc) * 1000 if wc > 0 else 0.0
        feats['space_comma_per_1000'] = (len(re.findall(r'\s,', text)) / wc) * 1000 if wc > 0 else 0.0
        feats['hyphen_per_1000'] = (text.count('-') / wc) * 1000 if wc > 0 else 0.0
        feats['danda_per_1000'] = (text.count('।') / wc) * 1000 if wc > 0 else 0.0

        feats['citation_count'] = len(re.findall(r'\[\d+\]', text))
        feats['wiki_heading_count'] = len(re.findall(r'==+.*?==+', text))
        first_200 = text[:200]
        feats['colon_in_first_200'] = 1.0 if ':' in first_200 else 0.0
        feats['def_in_first_200'] = 1.0 if any(m in first_200 for m in ['એટલે', 'કહેવાય', 'ગણાય']) else 0.0

        for m in ['તથા', 'વળી', 'કહેવાય છે', 'એટલે', 'કરાય છે',
                  'શામેલ', 'દ્વારા', 'સક્ષમ', 'ઉલ્લેખ', 'કરવામાં આવે છે']:
            feats['cnt_' + m.replace(' ', '_')] = text.count(m)

        trad = sum(text.count(c) for c in self.traditional_translit)
        mod = sum(text.count(c) for c in self.modern_translit)
        feats['translit_traditional_count'] = float(trad)
        feats['translit_modern_count'] = float(mod)
        feats['translit_style_ratio'] = trad / (trad + mod) if (trad + mod) > 0 else 0.5

        return feats

    def _empty(self) -> dict:
        return {k: 0.0 for k in self._names()}

    def _names(self):
        names = [
            'word_count', 'log_word_count', 'char_count', 'log_char_count',
            'sentence_count', 'avg_word_length',
            'avg_sentence_length', 'std_sentence_length',
            'max_sentence_length', 'min_sentence_length',
            'type_token_ratio', 'hapax_ratio',
            'v_markers_per_1000', 'w_markers_per_1000', 'marker_diff_per_1000',
            'marker_ratio',
            'passive_per_1000',
            'english_char_ratio', 'gujarati_char_ratio', 'script_ratio',
            'colon_per_1000', 'comma_per_1000', 'paren_per_1000',
            'space_comma_per_1000', 'hyphen_per_1000', 'danda_per_1000',
            'citation_count', 'wiki_heading_count',
            'colon_in_first_200', 'def_in_first_200',
        ]
        for m in ['તથા', 'વળી', 'કહેવાય છે', 'એટલે', 'કરાય છે',
                  'શામેલ', 'દ્વારા', 'સક્ષમ', 'ઉલ્લેખ', 'કરવામાં આવે છે']:
            names.append('cnt_' + m.replace(' ', '_'))
        names.extend([
            'translit_traditional_count',
            'translit_modern_count',
            'translit_style_ratio',
        ])
        return names


# ============================================================================
# SAFE PIPELINE — pads/truncates style features to match scaler
# ============================================================================
class SafePipeline:
    def __init__(self, raw_pipeline):
        self.raw = raw_pipeline
        self.extractor = StyleMatrixExtractor()

        self.expected_style_features = None
        try:
            if hasattr(raw_pipeline, "scaler") and hasattr(raw_pipeline.scaler, "n_features_in_"):
                self.expected_style_features = int(raw_pipeline.scaler.n_features_in_)
        except Exception:
            pass
        if self.expected_style_features is None:
            try:
                self.expected_style_features = len(raw_pipeline.scaler.mean_)
            except Exception:
                self.expected_style_features = 41

    def _style_matrix(self, texts):
        rows = []
        names = self.extractor._names()
        n_expect = self.expected_style_features

        for t in texts:
            f = self.extractor.extract(t)
            vals = [float(f.get(k, 0.0)) for k in names]
            if len(vals) < n_expect:
                vals = vals + [0.0] * (n_expect - len(vals))
            elif len(vals) > n_expect:
                vals = vals[:n_expect]
            rows.append(vals)

        arr = np.array(rows, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def transform(self, texts):
        try:
            return self.raw.transform(texts)
        except Exception:
            pass

        style_feats = self._style_matrix(texts)

        try:
            scaled = self.raw.scaler.transform(style_feats)
        except Exception:
            scaled = style_feats

        try:
            word_feats = self.raw.word_tfidf.transform(texts)
        except Exception:
            word_feats = None
        try:
            char_feats = self.raw.char_tfidf.transform(texts)
        except Exception:
            char_feats = None

        parts = [csr_matrix(scaled)]
        if word_feats is not None:
            parts.append(word_feats)
        if char_feats is not None:
            parts.append(char_feats)

        return hstack(parts).tocsr()


# ============================================================================
# Aliases — so pickle can find `main.Fpipe`, `main.StyleExt`, `main.Tk`, etc.
# ============================================================================
Fpipe = SafePipeline
StyleExt = StyleMatrixExtractor
Tk = GujaratiTokenizer
FeaturePipeline = SafePipeline
GujaratiStyleMatrixExtractor = StyleMatrixExtractor


# ============================================================================
# Inject classes into sys.modules['main'] BEFORE loading any pickle.
# Uses .get() with a safe fallback to avoid KeyError.
# ============================================================================
def _inject_into_main():
    injected = []

    candidates = {
        # Modern names used in this app
        "SafePipeline": SafePipeline,
        "StyleMatrixExtractor": StyleMatrixExtractor,
        "GujaratiTokenizer": GujaratiTokenizer,
        # Aliases for training-script names
        "Fpipe": SafePipeline,
        "StyleExt": StyleMatrixExtractor,
        "Tk": GujaratiTokenizer,
        # Aliases for intermediate versions
        "FeaturePipeline": SafePipeline,
        "GujaratiStyleMatrixExtractor": StyleMatrixExtractor,
    }
    # Drop any that are None
    candidates = {k: v for k, v in candidates.items() if v is not None}

    for mod_name in ["main", "__main__"]:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        for cls_name, cls in candidates.items():
            setattr(mod, cls_name, cls)
        injected.append(mod_name)
    return injected


INJECTED_MODULES = _inject_into_main()

if "main" not in sys.modules:
    sys.modules["main"] = sys.modules[__name__]
    for cls_name, cls in {
        "SafePipeline": SafePipeline,
        "StyleMatrixExtractor": StyleMatrixExtractor,
        "GujaratiTokenizer": GujaratiTokenizer,
        "Fpipe": SafePipeline,
        "StyleExt": StyleMatrixExtractor,
        "Tk": GujaratiTokenizer,
        "FeaturePipeline": SafePipeline,
        "GujaratiStyleMatrixExtractor": StyleMatrixExtractor,
    }.items():
        if cls is not None:
            setattr(sys.modules["main"], cls_name, cls)
    INJECTED_MODULES.append("main (newly created)")


# ============================================================================
# Safe import of rule-based classifier
# ============================================================================
try:
    from style_matrix_classifier import analyze_text
    STYLE_IMPORT_OK = True
except ImportError as e:
    STYLE_IMPORT_OK = False
    STYLE_IMPORT_ERR = str(e)

    def analyze_text(text):
        wc = len(text.split())
        return {
            'prediction': 'Unknown', 'confidence': 0.0,
            'votes': {'visvakosh_total': 0, 'wikipedia_total': 0, 'visvakosh_ratio': 0.5},
            'rule_results': [], 'visvakosh_satisfied': [], 'wikipedia_satisfied': [],
            'quantitative_analysis': {
                'length_metrics': {'word_count': wc, 'character_count': len(text),
                                   'sentence_count': text.count('.') + 1},
                'sentence_metrics': {'avg_sentence_length': 0, 'std_sentence_length': 0,
                                     'interpretation': 'style_matrix_classifier.py missing'},
                'vocabulary_metrics': {'type_token_ratio': 0, 'hapax_ratio': 0, 'interpretation': 'N/A'},
                'marker_metrics': {'visvakosh_markers_per_1000': 0, 'wikipedia_markers_per_1000': 0, 'interpretation': 'N/A'},
                'passive_metrics': {'visvakosh_passive_per_1000': 0, 'wikipedia_passive_per_1000': 0, 'interpretation': 'N/A'},
                'transliteration_metrics': {'traditional_count': 0, 'modern_count': 0, 'interpretation': 'N/A'},
                'punctuation_metrics': {'colon_per_1000': 0, 'semicolon_per_1000': 0, 'parentheses_per_1000': 0},
                'gloss_metrics': {'english_gloss_count': 0, 'english_glosses_per_1000': 0},
                'structural_metrics': {'citation_count': 0, 'wiki_heading_count': 0},
            },
            'qualitative_analysis': {
                'tone': {'definition_first': False, 'definition_style': 'N/A'},
                'structure': {'has_citations': False, 'has_wiki_headings': False},
                'vocabulary_style': {'marker_style': 'N/A', 'transliteration_style': 'N/A'},
                'glossing_style': {'gloss_density': 'N/A'},
            },
            'raw_features': {}
        }


# ============================================================================
# STREAMLIT SETUP
# ============================================================================
st.set_page_config(page_title="Visvakosh vs Wikipedia Classifier", page_icon="📚", layout="wide")
st.markdown("""
<style>
    .main-title { font-size: 2.5rem; font-weight: bold; color: #1f4e79;
                  text-align: center; margin-bottom: 0.5rem; }
    .subtitle { font-size: 1.1rem; color: #555; text-align: center; margin-bottom: 2rem; }
    .prediction-box { padding: 1.5rem; border-radius: 10px; margin: 1rem 0;
                      text-align: center; font-size: 1.5rem; font-weight: bold; }
    .visvakosh-pred { background-color: #d4edda; color: #155724; border: 2px solid #28a745; }
    .wikipedia-pred { background-color: #cce5ff; color: #004085; border: 2px solid #0066cc; }
    .unknown-pred { background-color: #fff3cd; color: #856404; border: 2px solid #ffc107; }
    .satisfied-tag { display: inline-block; background-color: #28a745; color: white;
                     padding: 3px 10px; border-radius: 12px; margin: 3px; font-size: 0.85rem; }
    .wikipedia-tag { display: inline-block; background-color: #0066cc; color: white;
                     padding: 3px 10px; border-radius: 12px; margin: 3px; font-size: 0.85rem; }
    .model-card { padding: 12px; border-radius: 8px; margin: 8px 0;
                  border-left: 5px solid #999; background: #f9f9f9; }
    .model-card-v { border-left-color: #28a745; background: #f0f9f2; }
    .model-card-w { border-left-color: #0066cc; background: #eff6ff; }
    .model-card-err { border-left-color: #dc3545; background: #fff5f5; }
    .model-name { font-weight: bold; font-size: 1.05rem; }
    .model-reason { color: #555; font-size: 0.92rem; margin-top: 6px; }
    .model-signal { display: inline-block; background: #e9ecef; color: #333;
                    padding: 2px 8px; border-radius: 10px; margin: 2px;
                    font-size: 0.8rem; font-family: monospace; }
    .stTextArea textarea { font-family: 'Noto Sans Gujarati', 'Shruti', sans-serif; font-size: 15px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📚 Gujarati Source Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Rule-based Visvakosh vs Wikipedia classification '
            'using style matrix + all ML models ensemble</div>', unsafe_allow_html=True)

if not STYLE_IMPORT_OK:
    st.warning(f"⚠️ `style_matrix_classifier.py` not found — rule-based classifier disabled. "
               f"({STYLE_IMPORT_ERR})")


# ============================================================================
# CONFIG + LOAD
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKIP_FILES = {"best_model.pkl", "visvakosh_classifier.pkl", "streamlit_app.pkl"}
DENSE_ONLY = {'SVC_RBF', 'KNN', 'MLP', 'LDA', 'DecisionTree'}


def discover_pkl_files():
    found = set()
    for p in glob.glob(os.path.join(SCRIPT_DIR, "*.pkl")):
        found.add(os.path.abspath(p))
    tm = os.path.join(SCRIPT_DIR, "trained_models")
    if os.path.isdir(tm):
        for p in glob.glob(os.path.join(tm, "*.pkl")):
            found.add(os.path.abspath(p))
    for root, dirs, files in os.walk(SCRIPT_DIR):
        dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'venv', '.venv',
                                                  'node_modules', '.streamlit'}]
        for f in files:
            if f.endswith(".pkl"):
                found.add(os.path.abspath(os.path.join(root, f)))
    return sorted(found)


def load_all_ml_models():
    models, status = {}, []
    pkl_paths = discover_pkl_files()
    if not pkl_paths:
        return models, status

    for path in pkl_paths:
        fname = os.path.relpath(path, SCRIPT_DIR)
        size = os.path.getsize(path)
        base = os.path.basename(path)

        if base in SKIP_FILES:
            status.append((fname, size, False, "skipped (in SKIP_FILES)"))
            continue
        if size < 200:
            status.append((fname, size, False, f"too small ({size} B)"))
            continue

        try:
            data = joblib.load(path)
            if not isinstance(data, dict):
                status.append((fname, size, False, f"expected dict, got {type(data).__name__}"))
                continue

            name = data.get("model_name", base.replace(".pkl", ""))
            if "model" not in data or "feature_pipeline" not in data:
                status.append((fname, size, False, f"missing keys: {list(data.keys())[:5]}"))
                continue

            safe_pipe = SafePipeline(data["feature_pipeline"])

            models[name] = {
                "model": data["model"],
                "pipeline": safe_pipe,
                "cv_f1": data.get("metrics", {}).get("cv_mean", 0.0),
                "test_acc": data.get("metrics", {}).get("accuracy", 0.0),
                "val_v_ok": data.get("val_v_ok", False),
                "val_w_ok": data.get("val_w_ok", False),
                "source": "local",
                "expected_features": safe_pipe.expected_style_features,
            }
            status.append((fname, size, True,
                           f"{name} (style features: {safe_pipe.expected_style_features})"))
        except Exception as e:
            status.append((fname, size, False, f"{type(e).__name__}: {str(e)[:80]}"))
    return models, status


if "ml_models" not in st.session_state:
    _m, _s = load_all_ml_models()
    st.session_state["ml_models"] = _m
    st.session_state["ml_status"] = _s

ALL_ML_MODELS = st.session_state["ml_models"]
LOAD_STATUS   = st.session_state["ml_status"]


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("⚙️ About")
    st.info("**Rule-Based Classifier** + **ML Models Ensemble**")
    st.markdown("---")
    st.header("📝 Sample Texts")
    sample_v = """કોમ્પ્યૂટર : વિવિધ કાર્યક્રમમાં આપેલી સૂચના અનુસાર માહિતીસંગ્રહ અને માહિતીપ્રક્રમણ માટેનું વીજાણુસાધન. તે સંજ્ઞાઓનું ઝડપથી અને ચોકસાઈપૂર્વક રૂપાંતર કરી શકતું મશીન છે."""
    sample_w = """કમ્પ્યુટર એ એક ઇલેક્ટ્રોનિક ઉપકરણ છે જે માહિતીને સંગ્રહિત કરી શકે છે. આ ઉપકરણનો ઉપયોગ વિવિધ ક્ષેત્રોમાં કરવામાં આવે છે. મુખ્ય લેખ: કમ્પ્યુટરનો ઇતિહાસ [1][2]"""
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📖 Visvakosh", use_container_width=True):
            st.session_state['sample_text'] = sample_v
    with c2:
        if st.button("🌐 Wikipedia", use_container_width=True):
            st.session_state['sample_text'] = sample_w
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state['sample_text'] = ""

    st.markdown("---")
    st.header("🤖 ML Models Status")
    st.caption(f"Injected: {INJECTED_MODULES}")
    st.caption(f"Scanning: `{SCRIPT_DIR}`")

    if st.button("🔄 Reload Models", use_container_width=True, key="reload_btn"):
        st.session_state.pop("ml_models", None)
        st.session_state.pop("ml_status", None)
        st.rerun()

    ok_count  = sum(1 for _, _, ok, _ in LOAD_STATUS if ok)
    err_count = sum(1 for _, _, ok, _ in LOAD_STATUS if not ok)
    if ok_count:  st.success(f"✅ {ok_count} models loaded")
    if err_count: st.error(f"❌ {err_count} not loaded")

    with st.expander(f"✅ Loaded ({ok_count})", expanded=(ok_count > 0)):
        for fname, size, ok, msg in LOAD_STATUS:
            if ok: st.write(f"✓ `{fname}` ({size:,} B) → **{msg}**")

    if err_count:
        with st.expander(f"❌ Failed ({err_count})", expanded=(ok_count == 0)):
            for fname, size, ok, msg in LOAD_STATUS:
                if not ok:
                    st.write(f"❌ `{fname}` ({size:,} B)")
                    st.caption(f"↳ {msg}")

    with st.expander("🔍 Debug: Discovered .pkl files", expanded=(ok_count == 0)):
        pkls = discover_pkl_files()
        if pkls:
            for p in pkls:
                rel = os.path.relpath(p, SCRIPT_DIR)
                st.write(f"• `{rel}` ({os.path.getsize(p):,} B)")
        else:
            st.write("No .pkl files found anywhere in the repo.")


# ============================================================================
# ML PREDICTION
# ============================================================================
def ml_predict_one(text, name, bundle):
    model, pipeline = bundle["model"], bundle["pipeline"]
    out = {"model": name, "prediction": None, "confidence": None,
           "proba_v": None, "proba_w": None,
           "reason": "", "signals": [], "error": None,
           "cv_f1": bundle.get("cv_f1", 0.0),
           "val_v_ok": bundle.get("val_v_ok", False),
           "val_w_ok": bundle.get("val_w_ok", False),
           "source": bundle.get("source", "?"),
           "expected_features": bundle.get("expected_features", "?")}

    try:
        X = pipeline.transform([text])
        if name in DENSE_ONLY:
            X = X.toarray()
        pred = model.predict(X)[0]
        out["prediction"] = "Wikipedia" if pred == 1 else "Visvakosh"

        if hasattr(model, "predict_proba"):
            try:
                p = model.predict_proba(X)[0]
                out["proba_v"] = float(p[0])
                out["proba_w"] = float(p[1])
                out["confidence"] = float(max(p))
            except Exception:
                pass

        if out["confidence"] is None and hasattr(model, "decision_function"):
            try:
                d = float(model.decision_function(X)[0])
                out["confidence"] = float(1 / (1 + np.exp(-abs(d))))
                out["proba_w"] = float(1 / (1 + np.exp(-d)))
                out["proba_v"] = 1.0 - out["proba_w"]
            except Exception:
                out["confidence"] = 1.0
                out["proba_v"] = 1.0 if pred == 0 else 0.0
                out["proba_w"] = 1.0 if pred == 1 else 0.0

        if out["confidence"] is None:
            out["confidence"] = 0.5
            out["proba_v"] = 0.5
            out["proba_w"] = 0.5

        extractor = StyleMatrixExtractor()
        feats = extractor.extract(text)
        reasons, signals = [], []

        if pred == 1:
            if feats.get("w_markers_per_1000", 0) > feats.get("v_markers_per_1000", 0):
                reasons.append(f"Wikipedia markers dominate "
                               f"({feats['w_markers_per_1000']:.1f}/1000 vs "
                               f"{feats['v_markers_per_1000']:.1f}/1000)")
                signals.append(f"w_markers={feats['w_markers_per_1000']:.1f}")
            if feats.get("english_char_ratio", 0) > 0.02:
                reasons.append(f"English glosses ({feats['english_char_ratio']:.1%})")
                signals.append(f"eng={feats['english_char_ratio']:.1%}")
            if feats.get("citation_count", 0) > 0:
                reasons.append(f"Citations found ({feats['citation_count']})")
                signals.append(f"citations={feats['citation_count']}")
            if not reasons:
                reasons.append("Statistical profile matches Wikipedia training")
        else:
            if feats.get("v_markers_per_1000", 0) > feats.get("w_markers_per_1000", 0):
                reasons.append(f"Visvakosh markers dominate "
                               f"({feats['v_markers_per_1000']:.1f}/1000 vs "
                               f"{feats['w_markers_per_1000']:.1f}/1000)")
                signals.append(f"v_markers={feats['v_markers_per_1000']:.1f}")
            if feats.get("colon_in_first_200", 0) == 1:
                reasons.append("Definition-first pattern")
                signals.append("def_colon")
            if not reasons:
                reasons.append("Statistical profile matches Visvakosh training")

        conf = out["confidence"]
        cw = "high" if conf > 0.85 else "moderate" if conf > 0.65 else "low"
        reasons.append(f"Confidence: {conf:.1%} ({cw})")
        if out["proba_v"] is not None:
            reasons.append(f"Probs — V: {out['proba_v']:.1%}, W: {out['proba_w']:.1%}")
        out["reason"] = " • ".join(reasons)
        out["signals"] = signals
        return out
    except Exception as e:
        out["error"] = str(e)[:200]
        out["prediction"] = "ERROR"
        out["reason"] = f"Model error: {out['error']}"
        return out


def ml_predict_all(text):
    return [ml_predict_one(text, n, b) for n, b in ALL_ML_MODELS.items()]


# ============================================================================
# MAIN INPUT
# ============================================================================
st.header("📝 Enter Gujarati Text")
text_input = st.text_area(
    "Paste Gujarati paragraph here:",
    value=st.session_state.get('sample_text', ''),
    height=300,
    placeholder="અહીં તમારું ગુજરાતી લખાણ પેસ્ટ કરો...",
    key="main_text"
)

if text_input:
    c1, c2, c3 = st.columns(3)
    c1.metric("Characters", f"{len(text_input):,}")
    c2.metric("Words", f"{len(text_input.split()):,}")
    c3.metric("Ready", "✅" if len(text_input) > 50 else "⚠️")

c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    analyze_btn = st.button(
        "🔍 ANALYZE TEXT", type="primary", use_container_width=True,
        disabled=(not text_input or len(text_input) < 30)
    )


# ============================================================================
# RESULTS
# ============================================================================
if analyze_btn:
    with st.spinner("Analyzing..."):
        result = analyze_text(text_input)

    st.success("✅ Analysis complete")
    st.markdown("---")
    st.header("🎯 Prediction")

    pred = result['prediction']
    conf = result['confidence']
    if pred == "Visvakosh":   css, emoji = "visvakosh-pred", "📖"
    elif pred == "Wikipedia": css, emoji = "wikipedia-pred", "🌐"
    else:                     css, emoji = "unknown-pred", "❓"

    st.markdown(
        f'<div class="prediction-box {css}">{emoji} Likely Source: '
        f'<strong>{pred}</strong><br>'
        f'<span style="font-size:1rem;">Confidence: {conf:.1%}</span></div>',
        unsafe_allow_html=True
    )

    v = result['votes']
    c1, c2, c3 = st.columns(3)
    c1.metric("📖 V Votes", v['visvakosh_total'])
    c2.metric("🌐 W Votes", v['wikipedia_total'])
    c3.metric("V Ratio", f"{v['visvakosh_ratio']:.1%}")
    st.progress(v['visvakosh_ratio'])

    st.markdown("---")
    st.header("📋 Rule-by-Rule Breakdown")
    for r in result['rule_results']:
        vv, wv = r['visvakosh_votes'], r['wikipedia_votes']
        if vv == 0 and wv == 0: continue
        if vv > 0: st.markdown(f"**+{vv} Visvakosh**  {r['reason']}")
        if wv > 0: st.markdown(f"**+{wv} Wikipedia**  {r['reason']}")

    st.markdown("---")
    st.header("✅ Satisfied Style Properties")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📖 Visvakosh Indicators")
        if result['visvakosh_satisfied']:
            for r in result['visvakosh_satisfied']:
                st.markdown(f'<span class="satisfied-tag">{r["rule"]}</span>', unsafe_allow_html=True)
        else:
            st.info("None met")
    with c2:
        st.subheader("🌐 Wikipedia Indicators")
        if result['wikipedia_satisfied']:
            for r in result['wikipedia_satisfied']:
                st.markdown(f'<span class="wikipedia-tag">{r["rule"]}</span>', unsafe_allow_html=True)
        else:
            st.info("None met")

    st.markdown("---")
    st.header("📈 Quantitative Analysis")
    quant = result['quantitative_analysis']

    with st.expander("📏 Length Metrics", expanded=True):
        lm = quant['length_metrics']
        c1, c2, c3 = st.columns(3)
        c1.metric("Words", f"{lm['word_count']:,}")
        c2.metric("Characters", f"{lm['character_count']:,}")
        c3.metric("Sentences", lm['sentence_count'])
    with st.expander("📝 Sentence Metrics", expanded=True):
        sm = quant['sentence_metrics']
        c1, c2 = st.columns(2)
        c1.metric("Avg Sentence Length", f"{sm['avg_sentence_length']:.2f}")
        c2.metric("Std Dev", f"{sm['std_sentence_length']:.2f}")
        st.info(f"💡 {sm['interpretation']}")
    with st.expander("📚 Vocabulary Metrics", expanded=True):
        vm = quant['vocabulary_metrics']
        c1, c2 = st.columns(2)
        c1.metric("Type-Token Ratio", f"{vm['type_token_ratio']:.4f}")
        c2.metric("Hapax Ratio", f"{vm['hapax_ratio']:.4f}")
        st.info(f"💡 {vm['interpretation']}")
    with st.expander("🏷️ Marker Metrics", expanded=True):
        mm = quant['marker_metrics']
        c1, c2 = st.columns(2)
        c1.metric("V Markers/1000", f"{mm['visvakosh_markers_per_1000']:.2f}")
        c2.metric("W Markers/1000", f"{mm['wikipedia_markers_per_1000']:.2f}")
        st.info(f"💡 {mm['interpretation']}")
    with st.expander("🔊 Passive Voice Metrics", expanded=True):
        pm = quant['passive_metrics']
        c1, c2 = st.columns(2)
        c1.metric("V Passive/1000", f"{pm['visvakosh_passive_per_1000']:.2f}")
        c2.metric("W Passive/1000", f"{pm['wikipedia_passive_per_1000']:.2f}")
        st.info(f"💡 {pm['interpretation']}")
    with st.expander("🔤 Transliteration Metrics", expanded=True):
        tm = quant['transliteration_metrics']
        c1, c2 = st.columns(2)
        c1.metric("Traditional", tm['traditional_count'])
        c2.metric("Modern", tm['modern_count'])
        st.info(f"💡 {tm['interpretation']}")
    with st.expander("❕ Punctuation Metrics", expanded=True):
        pum = quant['punctuation_metrics']
        c1, c2, c3 = st.columns(3)
        c1.metric("Colons/1000", f"{pum['colon_per_1000']:.2f}")
        c2.metric("Semicolons/1000", f"{pum['semicolon_per_1000']:.2f}")
        c3.metric("Parentheses/1000", f"{pum['parentheses_per_1000']:.2f}")
    with st.expander("🔠 English Glosses", expanded=True):
        gm = quant['gloss_metrics']
        c1, c2 = st.columns(2)
        c1.metric("English Glosses", gm['english_gloss_count'])
        c2.metric("Glosses/1000", f"{gm['english_glosses_per_1000']:.2f}")
    with st.expander("🏗️ Structural Metrics", expanded=True):
        stm = quant['structural_metrics']
        c1, c2 = st.columns(2)
        c1.metric("Citation Markers", stm['citation_count'])
        c2.metric("Wiki Headings", stm['wiki_heading_count'])

    st.markdown("---")
    st.header("🎭 Qualitative Analysis")
    qual = result['qualitative_analysis']
    with st.expander("🎵 Tone", expanded=True):
        t = qual['tone']
        st.markdown(f"**Definition-first:** {'Yes ✅' if t['definition_first'] else 'No ❌'}")
        st.markdown(f"**Style:** {t['definition_style']}")
    with st.expander("🏗️ Structure", expanded=True):
        s = qual['structure']
        st.markdown(f"**Citations:** {'Yes ✅' if s['has_citations'] else 'No ❌'}")
        st.markdown(f"**Wiki headings:** {'Yes ✅' if s['has_wiki_headings'] else 'No ❌'}")
    with st.expander("📖 Vocabulary Style", expanded=True):
        v_qual = qual['vocabulary_style']
        st.markdown(f"**Function words:** {v_qual['marker_style']}")
        st.markdown(f"**Transliteration:** {v_qual['transliteration_style']}")
    with st.expander("🔤 Glossing Style", expanded=True):
        g = qual['glossing_style']
        st.markdown(f"**Gloss density:** {g['gloss_density']}")
    with st.expander("🔬 Raw Feature Values"):
        df = pd.DataFrame([{"Feature": k, "Value": v}
                           for k, v in result['raw_features'].items()])
        st.dataframe(df, use_container_width=True, height=400)

    st.markdown("---")
    st.header("📥 Download Report")
    report = {
        "prediction": result['prediction'],
        "confidence": result['confidence'],
        "votes": result['votes'],
        "rule_results": result['rule_results'],
        "quantitative_analysis": result['quantitative_analysis'],
        "qualitative_analysis": result['qualitative_analysis'],
        "raw_features": result['raw_features']
    }
    st.download_button(
        "⬇️ Download JSON Report",
        data=json.dumps(report, indent=2, ensure_ascii=False, default=str),
        file_name=f"analysis_{result['prediction'].lower()}.json",
        mime="application/json",
        use_container_width=True
    )

    # ========================================================================
    # ML MODELS
    # ========================================================================
    st.markdown("---")
    st.header("🤖 ML Models — Individual Predictions & Reasoning")

    if not ALL_ML_MODELS:
        st.error("⚠️ 0 ML models loaded. Check sidebar ❌ Failed section and "
                 "🔍 Debug: Discovered .pkl files section.")
    else:
        with st.spinner(f"Running {len(ALL_ML_MODELS)} ML models..."):
            ml_results = ml_predict_all(text_input)

        n_v = n_w = 0
        cv_sum = cw_sum = 0.0
        for r in ml_results:
            if r["error"]: continue
            if r["prediction"] == "Visvakosh":
                n_v += 1; cv_sum += r["confidence"] or 0
            else:
                n_w += 1; cw_sum += r["confidence"] or 0

        total = n_v + n_w
        avg_v = cv_sum / n_v if n_v else 0
        avg_w = cw_sum / n_w if n_w else 0

        if n_v > n_w:   ens_verdict, ens_css, ens_emoji = "Visvakosh", "visvakosh-pred", "📖"
        elif n_w > n_v: ens_verdict, ens_css, ens_emoji = "Wikipedia", "wikipedia-pred", "🌐"
        else:           ens_verdict, ens_css, ens_emoji = "Tie", "unknown-pred", "⚖️"

        st.markdown(
            f'<div class="prediction-box {ens_css}">{ens_emoji} ML Ensemble Verdict: '
            f'<strong>{ens_verdict}</strong><br>'
            f'<span style="font-size:1rem;">{n_v} V / {n_w} W out of {total} models</span></div>',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📖 V Votes", n_v)
        c2.metric("🌐 W Votes", n_w)
        c3.metric("Avg V Conf", f"{avg_v:.1%}" if avg_v else "—")
        c4.metric("Avg W Conf", f"{avg_w:.1%}" if avg_w else "—")

        st.markdown("### 🔍 Per-Model Predictions & Reasoning")
        sorted_results = sorted(
            ml_results,
            key=lambda r: (not (r["val_v_ok"] and r["val_w_ok"]), -r["cv_f1"])
        )

        for r in sorted_results:
            if r["error"]:
                st.markdown(
                    f'<div class="model-card model-card-err">'
                    f'<div class="model-name">❌ {r["model"]}</div>'
                    f'<div class="model-reason">{r["reason"]}</div></div>',
                    unsafe_allow_html=True
                )
                continue

            cc = "model-card-v" if r["prediction"] == "Visvakosh" else "model-card-w"
            icon = "📖" if r["prediction"] == "Visvakosh" else "🌐"
            conf = r["confidence"] or 0.5

            badges = []
            if r["val_v_ok"] and r["val_w_ok"]:
                badges.append("✅ both validations passed")
            badges.append(f"CV F1 = {r['cv_f1']:.4f}")
            badges.append(f"style feats = {r.get('expected_features','?')}")

            sig_html = "".join(
                f'<span class="model-signal">{s}</span>' for s in r.get("signals", [])
            )

            proba_html = ""
            if r["proba_v"] is not None and r["proba_w"] is not None:
                pv = r["proba_v"] * 100; pw = r["proba_w"] * 100
                proba_html = (
                    f'<div style="margin-top:8px;font-size:0.85rem;">'
                    f'<div>📖 Visvakosh: <b>{pv:.1f}%</b> '
                    f'<div style="background:#e9ecef;border-radius:4px;height:8px;overflow:hidden;margin-top:2px;">'
                    f'<div style="background:#28a745;width:{pv:.1f}%;height:100%;"></div></div></div>'
                    f'<div style="margin-top:4px;">🌐 Wikipedia: <b>{pw:.1f}%</b> '
                    f'<div style="background:#e9ecef;border-radius:4px;height:8px;overflow:hidden;margin-top:2px;">'
                    f'<div style="background:#0066cc;width:{pw:.1f}%;height:100%;"></div></div></div></div>'
                )

            color = "#155724" if r["prediction"] == "Visvakosh" else "#004085"
            st.markdown(
                f'<div class="model-card {cc}"><div class="model-name">{icon} {r["model"]} → '
                f'<span style="color:{color};">{r["prediction"]}</span> '
                f'<span style="font-size:0.85rem;color:#666;">(confidence: {conf:.1%})</span></div>'
                f'<div style="font-size:0.85rem;color:#666;margin-top:4px;">{" • ".join(badges)}</div>'
                f'<div style="margin-top:6px;">{sig_html}</div>'
                f'<div class="model-reason">💡 <b>Why?</b> {r["reason"]}</div>{proba_html}</div>',
                unsafe_allow_html=True
            )

        st.markdown("### 📊 Summary Table")
        rows = []
        for r in sorted_results:
            if r["error"]:
                rows.append({
                    "Model": r["model"], "Prediction": "ERROR",
                    "Confidence": "—", "V %": "—", "W %": "—",
                    "CV F1": f"{r['cv_f1']:.4f}", "V-val": "—", "W-val": "—"
                })
            else:
                rows.append({
                    "Model": r["model"],
                    "Prediction": ("📖 " if r["prediction"] == "Visvakosh" else "🌐 ") + r["prediction"],
                    "Confidence": f"{r['confidence']:.1%}" if r["confidence"] else "—",
                    "V %": f"{r['proba_v']:.1%}" if r["proba_v"] is not None else "—",
                    "W %": f"{r['proba_w']:.1%}" if r["proba_w"] is not None else "—",
                    "CV F1": f"{r['cv_f1']:.4f}",
                    "V-val": "✓" if r["val_v_ok"] else "·",
                    "W-val": "✓" if r["val_w_ok"] else "·"
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     height=min(600, 40 + 35 * len(rows)))

        ml_report = {
            "ensemble_verdict": ens_verdict,
            "votes": {"visvakosh": n_v, "wikipedia": n_w, "total": total},
            "models": [
                {
                    "model": r["model"],
                    "prediction": r["prediction"],
                    "confidence": r["confidence"],
                    "proba_visvakosh": r["proba_v"],
                    "proba_wikipedia": r["proba_w"],
                    "cv_f1": r["cv_f1"],
                    "reason": r["reason"],
                    "signals": r["signals"],
                    "error": r["error"]
                }
                for r in sorted_results
            ]
        }
        st.download_button(
            "⬇️ Download ML Predictions JSON",
            data=json.dumps(ml_report, indent=2, ensure_ascii=False, default=str),
            file_name=f"ml_predictions_{ens_verdict.lower()}.json",
            mime="application/json",
            use_container_width=True,
            key="download_ml_report"
        )
