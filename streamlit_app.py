# ============================================================================
# streamlit_app.py — Rule-Based + All ML Models (LOCAL .pkl files)
# ============================================================================

import streamlit as st
import json
import pandas as pd
import numpy as np
import os
import re
import glob
import io
import sys
import types
import joblib
import pickle
from collections import Counter

from style_matrix_classifier import analyze_text


st.set_page_config(
    page_title="Visvakosh vs Wikipedia Classifier",
    page_icon="📚",
    layout="wide"
)


# ============================================================================
# ⚠️ CRITICAL — Define all classes that were in the training script
#    These must match the training script's class definitions EXACTLY
#    (same names, same methods, same attributes) so unpickling works.
# ============================================================================

class GujaratiTokenizer:
    GUJ = re.compile(r'[\u0A80-\u0AFF]+')
    ENG = re.compile(r'[a-zA-Z]+')
    NUM = re.compile(r'[0-9]+')

    @classmethod
    def words(cls, text):
        if not text:
            return []
        return cls.GUJ.findall(text) + cls.ENG.findall(text) + cls.NUM.findall(text)

    @classmethod
    def sentences(cls, text):
        if not text:
            return []
        t = text.replace('।', '.')
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', t)
                if len(s.strip()) > 2]


class StyleMatrixExtractor:
    def __init__(self):
        self.tk = GujaratiTokenizer()
        self.v_markers = [
            'તથા', 'વળી', 'આથી', 'ગણાય', 'પ્રચલિત', 'આવાં', 'કેટલાંક',
            'અલબત્ત', 'તદુપરાંત', 'દા.ત.', 'દા. ત.', 'જુઓ', 'એટલે કે',
            'કહેવાય છે', 'તેમજ', 'ઉપરાંત', 'વિશેષ', 'એટલે', 'કહેવાય',
            'કરાય છે', 'થાય છે', 'ઓળખાય છે', 'ગણાય છે'
        ]
        self.w_markers = [
            'શામેલ', 'ઘણીવાર', 'કોઈપણ', 'વ્યાખ્યાયિત', 'ઉદાહરણ તરીકે',
            'મોડેલ', 'સોફ્ટવેર', 'મુખ્ય લેખ', 'આ પણ જુઓ', 'જો કે',
            'દ્વારા', 'સંદર્ભ', 'બાહ્ય કડીઓ', 'સ્રોત', 'ટીકા', 'વિવાદ',
            'સક્ષમ', 'સમાવેશ', 'ઉલ્લેખ', 'પ્રોગ્રામ', 'ક્લસ્ટર',
            'કરવામાં આવે છે', 'આપવામાં આવે છે', 'બનાવવામાં આવે છે',
            'માનવામાં આવે છે', 'હતું', 'હતા', 'હતી'
        ]

    def extract(self, text: str) -> dict:
        if not text or len(text) < 20:
            return self._empty()
        words = self.tk.words(text)
        sentences = self.tk.sentences(text)
        if len(words) < 5:
            return self._empty()

        wc = len(words)
        cc = len(text)
        sc = max(len(sentences), 1)

        feats = {
            'word_count': wc, 'log_word_count': np.log1p(wc),
            'char_count': cc, 'log_char_count': np.log1p(cc),
            'sentence_count': sc,
            'avg_word_length': float(np.mean([len(w) for w in words])),
        }

        sl = [len(self.tk.words(s)) for s in sentences]
        sl = [l for l in sl if l > 0]
        feats['avg_sentence_length'] = float(np.mean(sl)) if sl else 0
        feats['std_sentence_length'] = float(np.std(sl)) if len(sl) > 1 else 0
        feats['max_sentence_length'] = float(max(sl)) if sl else 0
        feats['min_sentence_length'] = float(min(sl)) if sl else 0

        uniq = set(words)
        feats['type_token_ratio'] = len(uniq) / wc
        word_freq = Counter(words)
        feats['hapax_ratio'] = (
            sum(1 for c in word_freq.values() if c == 1) / max(len(uniq), 1)
        )
        feats['dis_ratio'] = (
            sum(1 for c in word_freq.values() if c == 2) / max(len(uniq), 1)
        )

        v_c = sum(text.count(m) for m in self.v_markers)
        w_c = sum(text.count(m) for m in self.w_markers)
        feats['v_markers_per_1000'] = (v_c / wc) * 1000
        feats['w_markers_per_1000'] = (w_c / wc) * 1000
        feats['marker_diff_per_1000'] = ((v_c - w_c) / wc) * 1000
        feats['marker_ratio_v'] = (v_c / (v_c + w_c)) if (v_c + w_c) > 0 else 0.5

        for m in ['તથા', 'વળી', 'કહેવાય છે', 'એટલે', 'કરાય છે',
                  'શામેલ', 'દ્વારા', 'સક્ષમ', 'ઉલ્લેખ', 'કરવામાં આવે છે']:
            key = 'm_' + m.replace(' ', '_').replace('.', '')
            feats[key] = text.count(m)

        eng_chars = len(re.findall(r'[a-zA-Z]', text))
        guj_chars = len(re.findall(r'[\u0A80-\u0AFF]', text))
        feats['english_char_ratio'] = eng_chars / cc if cc > 0 else 0
        feats['gujarati_char_ratio'] = guj_chars / cc if cc > 0 else 0
        feats['script_ratio'] = guj_chars / (guj_chars + eng_chars + 1)

        feats['colon_per_1000'] = (text.count(':') / wc) * 1000
        feats['comma_per_1000'] = (text.count(',') / wc) * 1000
        feats['paren_per_1000'] = ((text.count('(') + text.count(')')) / wc) * 1000
        feats['danda_per_1000'] = (text.count('।') / wc) * 1000

        first_200 = text[:200]
        feats['colon_in_first_200'] = 1.0 if ':' in first_200 else 0.0
        feats['def_in_first_200'] = 1.0 if any(
            m in first_200 for m in ['એટલે', 'કહેવાય', 'ગણાય']
        ) else 0.0

        feats['hyphen_count'] = text.count('-')
        feats['hyphen_per_1000'] = (text.count('-') / wc) * 1000
        feats['space_comma_count'] = len(re.findall(r'\s,', text))
        feats['space_comma_per_1000'] = (feats['space_comma_count'] / wc) * 1000
        feats['comma_ratio'] = text.count(',') / max(text.count('.'), 1)

        return feats

    def _empty(self) -> dict:
        return {
            'word_count': 0, 'log_word_count': 0, 'char_count': 0,
            'log_char_count': 0, 'sentence_count': 1, 'avg_word_length': 0,
            'avg_sentence_length': 0, 'std_sentence_length': 0,
            'max_sentence_length': 0, 'min_sentence_length': 0,
            'type_token_ratio': 0, 'hapax_ratio': 0, 'dis_ratio': 0,
            'v_markers_per_1000': 0, 'w_markers_per_1000': 0,
            'marker_diff_per_1000': 0, 'marker_ratio_v': 0.5,
            'm_તથા': 0, 'm_વળી': 0, 'm_કહેવાય_છે': 0, 'm_એટલે': 0,
            'm_કરાય_છે': 0, 'm_શામેલ': 0, 'm_દ્વારા': 0, 'm_સક્ષમ': 0,
            'm_ઉલ્લેખ': 0, 'm_કરવામાં_આવે_છે': 0,
            'english_char_ratio': 0, 'gujarati_char_ratio': 0, 'script_ratio': 0,
            'colon_per_1000': 0, 'comma_per_1000': 0, 'paren_per_1000': 0,
            'danda_per_1000': 0, 'colon_in_first_200': 0, 'def_in_first_200': 0,
            'hyphen_count': 0, 'hyphen_per_1000': 0,
            'space_comma_count': 0, 'space_comma_per_1000': 0,
            'comma_ratio': 0,
        }


class FeaturePipeline:
    """Must match training script's FeaturePipeline."""
    def __init__(self, max_word_features: int = 3000,
                 max_char_features: int = 3000):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import StandardScaler
        self.extractor = StyleMatrixExtractor()
        self.scaler = StandardScaler()
        self.word_tfidf = TfidfVectorizer(
            max_features=max_word_features,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            token_pattern=r'[\u0A80-\u0AFF]+|[a-zA-Z]+'
        )
        self.char_tfidf = TfidfVectorizer(
            analyzer='char_wb',
            max_features=max_char_features,
            ngram_range=(3, 5),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True
        )
        self.fitted = False
        self.style_names = None

    def fit_transform(self, texts):
        from scipy.sparse import hstack, csr_matrix
        sf = self._style(texts)
        ss = self.scaler.fit_transform(sf)
        wf = self.word_tfidf.fit_transform(texts)
        cf = self.char_tfidf.fit_transform(texts)
        self.fitted = True
        return hstack([csr_matrix(ss), wf, cf]).tocsr()

    def transform(self, texts):
        from scipy.sparse import hstack, csr_matrix
        sf = self._style(texts)
        ss = self.scaler.transform(sf)
        wf = self.word_tfidf.transform(texts)
        cf = self.char_tfidf.transform(texts)
        return hstack([csr_matrix(ss), wf, cf]).tocsr()

    def _style(self, texts):
        rows = [self.extractor.extract(t) for t in texts]
        df = pd.DataFrame(rows).fillna(0).replace([np.inf, -np.inf], 0)
        if self.style_names is None:
            self.style_names = df.columns.tolist()
        return df.values

    def get_dims(self):
        return {
            'style': len(self.style_names) if self.style_names else 0,
            'word': len(self.word_tfidf.vocabulary_) if hasattr(self.word_tfidf, 'vocabulary_') else 0,
            'char': len(self.char_tfidf.vocabulary_) if hasattr(self.char_tfidf, 'vocabulary_') else 0,
        }


# ============================================================================
# ⚠️ PATCH __main__ SO UNPICKLING FINDS OUR CLASSES
#    The .pkl files reference classes as `__main__.ClassName` (training script)
#    We register our versions under BOTH `__main__` and `main`
# ============================================================================

def _register_classes_in_main():
    """Make our classes available under both __main__ and main module names."""
    # Get the current module (streamlit_app)
    current = sys.modules[__name__]

    # Register under "__main__"
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        for cls_name in ["GujaratiTokenizer", "StyleMatrixExtractor",
                         "FeaturePipeline"]:
            if hasattr(current, cls_name):
                setattr(main_module, cls_name, getattr(current, cls_name))
        # Also register the module itself as "main" if not present
        if "main" not in sys.modules:
            sys.modules["main"] = main_module

    # If a "main" module already exists, populate it too
    main_alias = sys.modules.get("main")
    if main_alias is not None:
        for cls_name in ["GujaratiTokenizer", "StyleMatrixExtractor",
                         "FeaturePipeline"]:
            if hasattr(current, cls_name):
                setattr(main_alias, cls_name, getattr(current, cls_name))


_register_classes_in_main()


# Also inject as top-level names so pickle can find them
# by simple name (some pickle formats store just "FeaturePipeline")
for _cls_name in ["GujaratiTokenizer", "StyleMatrixExtractor", "FeaturePipeline"]:
    globals()[_cls_name] = globals()[_cls_name]


# ============================================================================
# CUSTOM UNPICKLER — Remap __main__ / main → streamlit_app
# ============================================================================

class _RemapUnpickler(pickle.Unpickler):
    """Remap any reference to __main__ or main module to our module."""
    def find_class(self, module, name):
        # If the pickle wants a class from __main__ or main,
        # redirect to our module where we defined those classes
        if module in ("__main__", "main", "streamlit_app"):
            # Try our module first
            this_module = sys.modules.get(__name__)
            if this_module is not None and hasattr(this_module, name):
                return getattr(this_module, name)
            # Fall back to __main__
            main_mod = sys.modules.get("__main__")
            if main_mod is not None and hasattr(main_mod, name):
                return getattr(main_mod, name)
        # Default behavior for everything else
        return super().find_class(module, name)


def safe_joblib_load(path):
    """Load a pickle file with module remapping to handle __main__ refs."""
    with open(path, "rb") as f:
        try:
            return _RemapUnpickler(f).load()
        except Exception:
            # Retry with standard joblib.load as fallback
            f.seek(0)
            return joblib.load(f)


# ============================================================================
# CONFIG
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKIP_FILES = {"best_model.pkl", "visvakosh_classifier.pkl"}
DENSE_ONLY = {'KNN', 'SVC_RBF', 'MLP', 'LDA', 'DecisionTree'}


# ============================================================================
# LOAD ALL MODELS
# ============================================================================

def load_all_ml_models():
    models = {}
    status = []

    pkl_paths = sorted(glob.glob(os.path.join(SCRIPT_DIR, "*.pkl")))
    if not pkl_paths:
        return models, status

    for path in pkl_paths:
        fname = os.path.basename(path)
        size = os.path.getsize(path)

        if fname in SKIP_FILES:
            status.append((fname, size, False, "skipped (not a classifier)"))
            continue

        if fname.startswith("GradientBoosting (1)"):
            status.append((fname, size, False, "skipped (duplicate)"))
            continue

        if size < 200:
            status.append((fname, size, False,
                           f"too small ({size} bytes — LFS pointer?)"))
            continue

        try:
            data = safe_joblib_load(path)
            if not isinstance(data, dict):
                status.append((fname, size, False,
                               f"expected dict, got {type(data).__name__}"))
                continue

            name = data.get("model_name", fname.replace(".pkl", ""))
            if "model" not in data or "feature_pipeline" not in data:
                status.append((fname, size, False,
                               f"missing keys. Got: {list(data.keys())}"))
                continue

            models[name] = {
                "model": data["model"],
                "pipeline": data["feature_pipeline"],
                "cv_f1": data.get("metrics", {}).get("cv_mean", 0.0),
                "test_acc": data.get("metrics", {}).get("accuracy", 0.0),
                "val_v_ok": data.get("val_v_ok", False),
                "val_w_ok": data.get("val_w_ok", False),
                "source": "local",
            }
            status.append((fname, size, True, name))

        except Exception as e:
            status.append((fname, size, False,
                           f"{type(e).__name__}: {str(e)[:80]}"))

    return models, status


if "ml_models" not in st.session_state:
    _m, _s = load_all_ml_models()
    st.session_state["ml_models"] = _m
    st.session_state["ml_status"] = _s

ALL_ML_MODELS = st.session_state["ml_models"]
LOAD_STATUS   = st.session_state["ml_status"]


# ============================================================================
# STYLES + UI
# ============================================================================

st.markdown("""
<style>
    .main-title { font-size: 2.5rem; font-weight: bold; color: #1f4e79;
                  text-align: center; margin-bottom: 0.5rem; }
    .subtitle { font-size: 1.1rem; color: #555; text-align: center;
                margin-bottom: 2rem; }
    .prediction-box { padding: 1.5rem; border-radius: 10px; margin: 1rem 0;
                      text-align: center; font-size: 1.5rem; font-weight: bold; }
    .visvakosh-pred { background-color: #d4edda; color: #155724;
                      border: 2px solid #28a745; }
    .wikipedia-pred { background-color: #cce5ff; color: #004085;
                      border: 2px solid #0066cc; }
    .unknown-pred { background-color: #fff3cd; color: #856404;
                    border: 2px solid #ffc107; }
    .satisfied-tag { display: inline-block; background-color: #28a745;
                     color: white; padding: 3px 10px; border-radius: 12px;
                     margin: 3px; font-size: 0.85rem; }
    .wikipedia-tag { display: inline-block; background-color: #0066cc;
                     color: white; padding: 3px 10px; border-radius: 12px;
                     margin: 3px; font-size: 0.85rem; }
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
    .stTextArea textarea { font-family: 'Noto Sans Gujarati', 'Shruti', sans-serif;
                           font-size: 15px; }
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-title">📚 Gujarati Source Classifier</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Rule-based Visvakosh vs Wikipedia classification '
    'using style matrix (qualitative + quantitative) + ML models ensemble</div>',
    unsafe_allow_html=True
)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("⚙️ About")
    st.info(
        "**Rule-Based Classifier** + **ML Models Ensemble**\n\n"
        "Uses 14 style matrix rules + all trained ML models."
    )

    st.markdown("---")
    st.header("📝 Sample Texts")

    sample_v = """કોમ્પ્યૂટર : વિવિધ કાર્યક્રમમાં આપેલી સૂચના અનુસાર માહિતીસંગ્રહ અને માહિતીપ્રક્રમણ માટેનું વીજાણુસાધન. તે સંજ્ઞાઓનું ઝડપથી અને ચોકસાઈપૂર્વક રૂપાંતર કરી શકતું મશીન છે. આથી તેને ગણાય છે. કોમ્પ્યુટરમાં દ્વિઅંકી સંજ્ઞા (binary code) 0 અને 1 વપરાય છે. વળી, ઍનાલિટિક એન્જિન (analytical engine) નામે ગણનયંત્ર ચાર્લ્સ બેબેજે બનાવ્યું હતું. તથા તે 1837માં બનાવવામાં આવ્યું હતું."""

    sample_w = """કમ્પ્યુટર એ એક ઇલેક્ટ્રોનિક ઉપકરણ છે જે માહિતીને સંગ્રહિત કરી શકે છે અને પ્રક્રિયા કરી શકે છે. આ ઉપકરણનો ઉપયોગ વિવિધ ક્ષેત્રોમાં કરવામાં આવે છે. ઉદાહરણ તરીકે, શિક્ષણ, આરોગ્ય સંભાળ, વ્યાપાર વગેરેમાં કમ્પ્યુટરનો ઉપયોગ કરવામાં આવે છે. કમ્પ્યુટરની શોધ ઘણા વૈજ્ઞાનિકો દ્વારા કરવામાં આવી હતી. જો કે, ચાર્લ્સ બેબેજને કમ્પ્યુટરના પિતા ગણવામાં આવે છે. મુખ્ય લેખ: કમ્પ્યુટરનો ઇતિહાસ [1][2]"""

    c1, c2 = st.columns(2)
    with c1:
        if st.button("📖 Visvakosh Sample", use_container_width=True):
            st.session_state['sample_text'] = sample_v
    with c2:
        if st.button("🌐 Wikipedia Sample", use_container_width=True):
            st.session_state['sample_text'] = sample_w

    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state['sample_text'] = ""

    st.markdown("---")
    st.header("🤖 ML Models Status")

    if st.button("🔄 Reload Models", use_container_width=True, key="reload_btn"):
        st.session_state.pop("ml_models", None)
        st.session_state.pop("ml_status", None)
        st.rerun()

    ok_count  = sum(1 for _, _, ok, _ in LOAD_STATUS if ok)
    err_count = sum(1 for _, _, ok, _ in LOAD_STATUS if not ok)

    if ok_count:
        st.success(f"✅ {ok_count} models loaded")
    if err_count:
        st.error(f"❌ {err_count} not loaded")

    with st.expander(f"✅ Loaded ({ok_count})", expanded=(ok_count > 0)):
        for fname, size, ok, msg in LOAD_STATUS:
            if ok:
                st.write(f"✓ `{fname}` ({size:,} B) → **{msg}**")

    if err_count:
        with st.expander(f"❌ Failed ({err_count})", expanded=(ok_count == 0)):
            for fname, size, ok, msg in LOAD_STATUS:
                if not ok:
                    st.write(f"❌ `{fname}` ({size:,} B)")
                    st.caption(f"↳ {msg}")

    st.caption(f"📁 Looking in: `{SCRIPT_DIR}`")


# ============================================================================
# ML PREDICTION + REASONING
# ============================================================================

def ml_predict_one(text, name, bundle):
    model = bundle["model"]
    pipeline = bundle["pipeline"]

    out = {
        "model": name, "prediction": None, "confidence": None,
        "proba_v": None, "proba_w": None,
        "reason": "", "signals": [], "error": None,
        "cv_f1": bundle.get("cv_f1", 0.0),
        "val_v_ok": bundle.get("val_v_ok", False),
        "val_w_ok": bundle.get("val_w_ok", False),
        "source": bundle.get("source", "?"),
    }

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

        extractor = StyleMatrixExtractor()
        feats = extractor.extract(text)
        reasons, signals = [], []

        if pred == 1:
            if feats.get("w_markers_per_1000", 0) > feats.get("v_markers_per_1000", 0):
                reasons.append(
                    f"Wikipedia markers dominate "
                    f"({feats['w_markers_per_1000']:.1f}/1000 vs "
                    f"{feats['v_markers_per_1000']:.1f}/1000)"
                )
                signals.append(f"w_markers={feats['w_markers_per_1000']:.1f}")
            if feats.get("space_comma_per_1000", 0) > 5:
                reasons.append(
                    f"Wiki-style space before commas "
                    f"({feats['space_comma_per_1000']:.1f}/1000)"
                )
                signals.append(f"space_comma={feats['space_comma_per_1000']:.1f}")
            if feats.get("english_char_ratio", 0) > 0.02:
                reasons.append(
                    f"Frequent English glosses "
                    f"({feats['english_char_ratio']:.1%})"
                )
                signals.append(f"eng={feats['english_char_ratio']:.1%}")
            if feats.get("hyphen_per_1000", 0) > 3:
                reasons.append(
                    f"Hyphenated neologisms "
                    f"({feats['hyphen_per_1000']:.1f}/1000)"
                )
                signals.append(f"hyphen={feats['hyphen_per_1000']:.1f}")
            if feats.get("colon_in_first_200", 0) == 0:
                reasons.append("No definition-first colon")
                signals.append("no_def_colon")
            if not reasons:
                reasons.append("Statistical profile matches Wikipedia training")
        else:
            if feats.get("v_markers_per_1000", 0) > feats.get("w_markers_per_1000", 0):
                reasons.append(
                    f"Visvakosh markers dominate "
                    f"({feats['v_markers_per_1000']:.1f}/1000 vs "
                    f"{feats['w_markers_per_1000']:.1f}/1000)"
                )
                signals.append(f"v_markers={feats['v_markers_per_1000']:.1f}")
            if feats.get("colon_in_first_200", 0) == 1:
                reasons.append("Definition-first pattern (colon in opening)")
                signals.append("def_colon")
            if feats.get("def_in_first_200", 0) == 1:
                reasons.append("Encyclopedic opener (એટલે/કહેવાય/ગણાય)")
                signals.append("def_opener")
            if feats.get("danda_per_1000", 0) > 5:
                reasons.append(
                    f"High danda (।) usage "
                    f"({feats['danda_per_1000']:.1f}/1000)"
                )
                signals.append(f"danda={feats['danda_per_1000']:.1f}")
            if feats.get("english_char_ratio", 0) < 0.02:
                reasons.append(
                    f"Low English ratio ({feats['english_char_ratio']:.1%})"
                )
                signals.append(f"eng={feats['english_char_ratio']:.1%}")
            if feats.get("hyphen_per_1000", 0) < 2:
                reasons.append("Rare hyphenated neologisms")
                signals.append("low_hyphen")
            if not reasons:
                reasons.append("Statistical profile matches Visvakosh training")

        conf = out["confidence"] or 0.5
        conf_word = "high" if conf > 0.85 else "moderate" if conf > 0.65 else "low"
        reasons.append(f"Confidence: {conf:.1%} ({conf_word})")

        if out["proba_v"] is not None and out["proba_w"] is not None:
            reasons.append(
                f"Probs — V: {out['proba_v']:.1%}, W: {out['proba_w']:.1%}"
            )

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
    c2.metric("Words (approx)", f"{len(text_input.split()):,}")
    c3.metric("Ready", "✅" if len(text_input) > 50 else "⚠️ Too short")


c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    analyze_btn = st.button(
        "🔍 ANALYZE TEXT",
        type="primary",
        use_container_width=True,
        disabled=(not text_input or len(text_input) < 30)
    )


# ============================================================================
# RESULTS
# ============================================================================

if analyze_btn:
    with st.spinner("Analyzing style matrix..."):
        result = analyze_text(text_input)

    st.success("✅ Analysis complete")
    st.markdown("---")

    st.header("🎯 Prediction")
    pred = result['prediction']
    conf = result['confidence']

    if pred == "Visvakosh":
        css, emoji = "visvakosh-pred", "📖"
    elif pred == "Wikipedia":
        css, emoji = "wikipedia-pred", "🌐"
    else:
        css, emoji = "unknown-pred", "❓"

    st.markdown(
        f'<div class="prediction-box {css}">'
        f'{emoji} Likely Source: <strong>{pred}</strong><br>'
        f'<span style="font-size:1rem;">Confidence: {conf:.1%}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    v = result['votes']
    c1, c2, c3 = st.columns(3)
    c1.metric("📖 Visvakosh Votes", v['visvakosh_total'])
    c2.metric("🌐 Wikipedia Votes", v['wikipedia_total'])
    c3.metric("Visvakosh Ratio", f"{v['visvakosh_ratio']:.1%}")
    st.progress(v['visvakosh_ratio'])

    st.markdown("---")
    st.header("📋 Rule-by-Rule Breakdown")
    st.caption("Each rule contributes votes to Visvakosh or Wikipedia")

    for r in result['rule_results']:
        v_votes = r['visvakosh_votes']
        w_votes = r['wikipedia_votes']
        if v_votes == 0 and w_votes == 0:
            continue
        if v_votes > 0:
            st.markdown(f"**+{v_votes} Visvakosh**  {r['reason']}")
        if w_votes > 0:
            st.markdown(f"**+{w_votes} Wikipedia**  {r['reason']}")

    st.markdown("---")
    st.header("✅ Satisfied Style Properties")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📖 Visvakosh Indicators")
        if result['visvakosh_satisfied']:
            for r in result['visvakosh_satisfied']:
                st.markdown(
                    f'<span class="satisfied-tag">{r["rule"]}</span>',
                    unsafe_allow_html=True
                )
        else:
            st.info("None met")
    with c2:
        st.subheader("🌐 Wikipedia Indicators")
        if result['wikipedia_satisfied']:
            for r in result['wikipedia_satisfied']:
                st.markdown(
                    f'<span class="wikipedia-tag">{r["rule"]}</span>',
                    unsafe_allow_html=True
                )
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
        st.markdown(f"**Note:** {s['citation_note']}")

    with st.expander("📖 Vocabulary Style", expanded=True):
        v = qual['vocabulary_style']
        st.markdown(f"**Function words:** {v['marker_style']}")
        st.markdown(f"**Transliteration:** {v['transliteration_style']}")

    with st.expander("🔤 Glossing Style", expanded=True):
        g = qual['glossing_style']
        st.markdown(f"**Gloss density:** {g['gloss_density']}")

    with st.expander("🔬 Raw Feature Values"):
        df = pd.DataFrame([
            {"Feature": k, "Value": v}
            for k, v in result['raw_features'].items()
        ])
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
    # ML MODELS SECTION
    # ========================================================================
    st.markdown("---")
    st.header("🤖 ML Models — Individual Predictions & Reasoning")
    st.caption(
        "Each trained model predicts independently. "
        "Below each model, you can see **why** it made that prediction."
    )

    if not ALL_ML_MODELS:
        st.error(
            "⚠️ **0 ML models loaded.**\n\n"
            f"Looking in: `{SCRIPT_DIR}`\n\n"
            "Check the sidebar **❌ Failed** expander for details."
        )
    else:
        with st.spinner(f"Running {len(ALL_ML_MODELS)} ML models..."):
            ml_results = ml_predict_all(text_input)

        conf_v_sum = 0.0
        conf_w_sum = 0.0
        n_v, n_w = 0, 0
        for r in ml_results:
            if r["error"]:
                continue
            if r["prediction"] == "Visvakosh":
                n_v += 1
                if r["confidence"]:
                    conf_v_sum += r["confidence"]
            else:
                n_w += 1
                if r["confidence"]:
                    conf_w_sum += r["confidence"]

        total = n_v + n_w
        avg_v = (conf_v_sum / n_v) if n_v else 0
        avg_w = (conf_w_sum / n_w) if n_w else 0

        if n_v > n_w:
            ens_verdict, ens_css, ens_emoji = "Visvakosh", "visvakosh-pred", "📖"
        elif n_w > n_v:
            ens_verdict, ens_css, ens_emoji = "Wikipedia", "wikipedia-pred", "🌐"
        else:
            ens_verdict, ens_css, ens_emoji = "Tie", "unknown-pred", "⚖️"

        st.markdown(
            f'<div class="prediction-box {ens_css}">'
            f'{ens_emoji} ML Ensemble Verdict: <strong>{ens_verdict}</strong><br>'
            f'<span style="font-size:1rem;">'
            f'{n_v} Visvakosh / {n_w} Wikipedia out of {total} models'
            f'</span></div>',
            unsafe_allow_html=True
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📖 Visvakosh Votes", n_v)
        c2.metric("🌐 Wikipedia Votes", n_w)
        c3.metric("Avg V Conf", f"{avg_v:.1%}" if avg_v else "—")
        c4.metric("Avg W Conf", f"{avg_w:.1%}" if avg_w else "—")

        st.markdown("### 🔍 Per-Model Predictions & Reasoning")

        sorted_results = sorted(
            ml_results,
            key=lambda r: (
                not (r["val_v_ok"] and r["val_w_ok"]),
                -r["cv_f1"],
            )
        )

        for r in sorted_results:
            if r["error"]:
                st.markdown(
                    f'<div class="model-card model-card-err">'
                    f'<div class="model-name">❌ {r["model"]}</div>'
                    f'<div class="model-reason">{r["reason"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                continue

            card_class = "model-card-v" if r["prediction"] == "Visvakosh" else "model-card-w"
            icon = "📖" if r["prediction"] == "Visvakosh" else "🌐"
            conf = r["confidence"] if r["confidence"] is not None else 0.5

            badges = []
            if r["val_v_ok"] and r["val_w_ok"]:
                badges.append("✅ both validations passed")
            badges.append(f"CV F1 = {r['cv_f1']:.4f}")

            signals_html = "".join(
                f'<span class="model-signal">{s}</span>' for s in r.get("signals", [])
            )

            proba_html = ""
            if r["proba_v"] is not None and r["proba_w"] is not None:
                pv = r["proba_v"] * 100
                pw = r["proba_w"] * 100
                proba_html = (
                    f'<div style="margin-top:8px;font-size:0.85rem;">'
                    f'<div>📖 Visvakosh: <b>{pv:.1f}%</b> '
                    f'<div style="background:#e9ecef;border-radius:4px;height:8px;'
                    f'overflow:hidden;margin-top:2px;">'
                    f'<div style="background:#28a745;width:{pv:.1f}%;height:100%;">'
                    f'</div></div></div>'
                    f'<div style="margin-top:4px;">🌐 Wikipedia: <b>{pw:.1f}%</b> '
                    f'<div style="background:#e9ecef;border-radius:4px;height:8px;'
                    f'overflow:hidden;margin-top:2px;">'
                    f'<div style="background:#0066cc;width:{pw:.1f}%;height:100%;">'
                    f'</div></div></div>'
                    f'</div>'
                )

            color = "#155724" if r["prediction"] == "Visvakosh" else "#004085"
            st.markdown(
                f'<div class="model-card {card_class}">'
                f'<div class="model-name">{icon} {r["model"]} → '
                f'<span style="color:{color};">{r["prediction"]}</span> '
                f'<span style="font-size:0.85rem;color:#666;">'
                f'(confidence: {conf:.1%})</span></div>'
                f'<div style="font-size:0.85rem;color:#666;margin-top:4px;">'
                f'{" • ".join(badges)}</div>'
                f'<div style="margin-top:6px;">{signals_html}</div>'
                f'<div class="model-reason">💡 <b>Why?</b> {r["reason"]}</div>'
                f'{proba_html}'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("### 📊 Summary Table")
        summary_rows = []
        for r in sorted_results:
            if r["error"]:
                summary_rows.append({
                    "Model": r["model"], "Prediction": "ERROR",
                    "Confidence": "—", "Visvakosh %": "—",
                    "Wikipedia %": "—", "CV F1": f"{r['cv_f1']:.4f}",
                    "V-val": "—", "W-val": "—",
                })
            else:
                summary_rows.append({
                    "Model": r["model"],
                    "Prediction": ("📖 " if r["prediction"] == "Visvakosh" else "🌐 ")
                                  + r["prediction"],
                    "Confidence": (f"{r['confidence']:.1%}"
                                   if r["confidence"] is not None else "—"),
                    "Visvakosh %": (f"{r['proba_v']:.1%}"
                                    if r["proba_v"] is not None else "—"),
                    "Wikipedia %": (f"{r['proba_w']:.1%}"
                                    if r["proba_w"] is not None else "—"),
                    "CV F1": f"{r['cv_f1']:.4f}",
                    "V-val": "✓" if r["val_v_ok"] else "·",
                    "W-val": "✓" if r["val_w_ok"] else "·",
                })

        st.dataframe(
            pd.DataFrame(summary_rows),
            use_container_width=True,
            height=min(600, 40 + 35 * len(summary_rows))
        )

        st.markdown("### 📥 Download ML Report")
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
                    "val_v_ok": r["val_v_ok"],
                    "val_w_ok": r["val_w_ok"],
                    "reason": r["reason"],
                    "signals": r["signals"],
                    "error": r["error"],
                }
                for r in sorted_results
            ],
        }
        st.download_button(
            "⬇️ Download ML Predictions JSON",
            data=json.dumps(ml_report, indent=2, ensure_ascii=False, default=str),
            file_name=f"ml_predictions_{ens_verdict.lower()}.json",
            mime="application/json",
            use_container_width=True,
            key="download_ml_report",
        )
