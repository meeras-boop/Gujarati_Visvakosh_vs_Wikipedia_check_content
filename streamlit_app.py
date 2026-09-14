# ============================================================================
# streamlit_app.py — FULL CORRECTED VERSION with numbered calculation traces
#   + CATEGORY DETECTION using category-specific keyword sets
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
from typing import Dict, List, Tuple, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix


# ============================================================================
# CATEGORY-SPECIFIC KEYWORD SETS  (from Category-specific keyword sets.txt)
# ============================================================================
CATEGORY_KEYWORDS = {
    "Person Biography": {
        "visvakosh": [
            'જન્મ', 'અવસાન', 'પદવી', 'શિક્ષણ', 'યુનિવર્સિટી', 'પ્રોફેસર',
            'વૈજ્ઞાનિક', 'ગણિતશાસ્ત્રી', 'ઇજનેર', 'સંશોધક', 'સંશોધન',
            'કારકિર્દી', 'પ્રદાન', 'પુરસ્કાર', 'ઍવૉર્ડ', 'મેડલ', 'એનાયત',
            'ફેલો', 'સભ્ય', 'સ્થાપક', 'ડિરેક્ટર', 'પીએચ.ડી.', 'હિન્ટન',
            'એકર્ટ', 'એડા', 'ટ્યૂરિંગ', 'બૅબેજ', 'ચિદંબરમ્',
        ],
        "wikipedia": [
            'જન્મ', 'મૃત્યુ', 'જીવન', 'જીવનચરિત્ર', 'કારકિર્દી', 'શિક્ષણ',
            'કાર્ય', 'યોગદાન', 'સંશોધન', 'વૈજ્ઞાનિક', 'ગણિતશાસ્ત્રી',
            'ઇજનેર', 'પ્રોગ્રામર', 'પ્રોફેસર', 'યુનિવર્સિટી', 'પુરસ્કાર',
            'સન્માન', 'પદવી', 'જાણીતા', 'સ્થાપક',
            'researcher', 'scientist', 'engineer', 'mathematician',
        ],
    },
    "Computer Science & Information Technology": {
        "visvakosh": [
            'કોમ્પ્યૂટર', 'કમ્પ્યુટર', 'સૉફ્ટવૅર', 'હાર્ડવેર', 'પ્રોગ્રામ',
            'પ્રોગ્રામિંગ', 'અલ્ગોરિધમ', 'માહિતી', 'ડેટા', 'નિવેશ',
            'ઇનપુટ', 'નિર્ગમ', 'આઉટપુટ', 'CPU', 'મેમરી', 'સ્ટોરેજ',
            'ડિસ્ક', 'ઇન્ટરનેટ', 'નેટવર્ક', 'વેબ', 'વેબસાઇટ', 'સર્વર',
            'ક્લાયન્ટ', 'DNS', 'ઇ-મેઇલ', 'કૃત્રિમ બુદ્ધિમત્તા', 'ચેટબોટ',
            'મશીન લર્નિંગ', 'ડેટાબેઇઝ', 'દ્વિઅંકી',
        ],
        "wikipedia": [
            'કમ્પ્યુટર', 'computer', 'software', 'hardware', 'program',
            'programming', 'algorithm', 'data', 'database', 'memory',
            'processor', 'CPU', 'internet', 'network', 'web', 'server',
            'browser', 'email', 'ઇ-મેઇલ', 'AI', 'કૃત્રિમ બુદ્ધિમત્તા',
            'machine learning', 'deep learning', 'ChatGPT', 'OpenAI',
            'chatbot', 'information technology', 'IT', 'code',
        ],
    },
    "Engineering & Technology": {
        "visvakosh": [
            'ઇજનેરી', 'યંત્ર', 'સાધન', 'ડિઝાઇન', 'ઇલેકટ્રોનિક્સ',
            'ઇલેક્ટ્રોનિક', 'વિદ્યુત', 'પરિપથ', 'ટ્રાન્ઝિસ્ટર',
            'અર્ધવાહક', 'IC', 'માઇક્રોવેવ', 'તરંગ', 'આવૃત્તિ', 'રેડિયો',
            'સિગ્નલ', 'યાન', 'યાન-નયન', 'કૉકપિટ', 'મોટર', 'વાહન',
            'ઑટોમોબાઇલ', 'ઉત્પાદન', 'બીબું', 'વૉશિંગ મશીન',
            'સંદેશાવ્યવહાર', 'પ્રસારણ', 'ઉપગ્રહ', 'નિયંત્રણ',
        ],
        "wikipedia": [
            'engineering', 'technology', 'ઇજનેરી', 'યંત્ર', 'machine',
            'device', 'system', 'design', 'electronic', 'electronics',
            'electrical', 'circuit', 'semiconductor', 'transistor',
            'microwave', 'frequency', 'signal', 'navigation', 'vehicle',
            'automobile', 'cockpit', 'motor', 'manufacturing',
            'production', 'communication', 'transmission', 'satellite',
            'control system',
        ],
    },
    "Medical & Health Sciences": {
        "visvakosh": [
            'ઔષધ', 'ઔષધો', 'દવા', 'ફાર્મસી', 'રોગ', 'દર્દી', 'નિદાન',
            'સારવાર', 'શરીર', 'શ્રવણ', 'શ્રવણસહાયક', 'કાન', 'વિકિરણ',
            'કિરણોત્સર્ગી', 'વિકિરણશીલ', 'સમસ્થાનિક', 'રેડિયો સમસ્થાનિક',
            'ગૅમા', 'થાઇરૉઇડ', 'ચિત્રણ', 'સ્કૅન', 'PET', 'સ્મૃતિ',
            'સ્મૃતિલોપ', 'વિસ્મૃતિ', 'યાદ', 'મગજ', 'દીર્ઘકાલીન',
            'અલ્પકાલીન', 'ઈજા',
        ],
        "wikipedia": [
            'medical', 'medicine', 'health', 'ઔષધ', 'દવા', 'રોગ', 'disease',
            'patient', 'દર્દી', 'diagnosis', 'નિદાન', 'treatment', 'સારવાર',
            'pharmacy', 'hearing', 'hearing aid', 'કાન', 'radioisotope',
            'isotope', 'radiation', 'scan', 'imaging', 'PET', 'thyroid',
            'memory', 'સ્મૃતિ', 'amnesia', 'સ્મૃતિલોપ', 'brain', 'મગજ',
        ],
    },
    "Education & Research": {
        "visvakosh": [
            'શિક્ષણ', 'પ્રાથમિક', 'શાળા', 'વિદ્યાર્થી', 'શિક્ષક',
            'અધ્યાપક', 'અધ્યાપન', 'બોધન', 'અધ્યયન', 'અભ્યાસ',
            'અભ્યાસક્રમ', 'પરીક્ષા', 'તાલીમ', 'વિશ્વવિદ્યાલય',
            'ઉચ્ચ શિક્ષણ', 'અનુદાન', 'આયોગ', 'UGC', 'શિક્ષણનીતિ',
            'સંશોધન', 'પ્રયોગશાળા', 'PRL', 'સંસ્થા', 'વ્યવસ્થાપન',
            'IIM', 'ભૌતિકવિજ્ઞાન', 'યુનિવર્સિટી', 'શૈક્ષણિક',
        ],
        "wikipedia": [
            'education', 'શિક્ષણ', 'school', 'શાળા', 'student',
            'વિદ્યાર્થી', 'teacher', 'શિક્ષક', 'teaching', 'બોધન',
            'learning', 'અધ્યયન', 'curriculum', 'પાઠ્યક્રમ', 'university',
            'વિશ્વવિદ્યાલય', 'higher education', 'research', 'સંશોધન',
            'laboratory', 'પ્રયોગશાળા', 'UGC', 'grant', 'અનુદાન',
            'commission', 'institute', 'PRL', 'IIM', 'academic',
        ],
    },
    "Language, Literature & Communication": {
        "visvakosh": [
            'લેખન', 'લખાણ', 'શબ્દ', 'ભાષા', 'લિપિ', 'અક્ષર', 'શૈલી',
            'પ્રૂફ', 'પ્રૂફરીડિંગ', 'ભૂલો', 'સંપાદન', 'પ્રકાશન', 'કોશ',
            'પર્યાયકોશ', 'પર્યાય', 'નિઘંટુ', 'દસ્તાવેજ', 'ડૉક્યુમેન્ટેશન',
            'લેખ્યસૂચિ', 'લેખ્યસૂચીકરણ', 'ગ્રંથાલય', 'સુલેખન', 'લેખિની',
            'કલમ', 'શાહી', 'કાગળ', 'પ્રતિલિપિ', 'અક્ષરમાળા',
        ],
        "wikipedia": [
            'language', 'ભાષા', 'literature', 'સાહિત્ય', 'writing',
            'લેખન', 'text', 'લખાણ', 'word', 'શબ્દ', 'script', 'લિપિ',
            'letter', 'અક્ષર', 'proofreading', 'પ્રૂફરીડિંગ', 'editing',
            'સંપાદન', 'dictionary', 'શબ્દકોશ', 'thesaurus', 'પર્યાયકોશ',
            'synonym', 'પર્યાય', 'documentation', 'દસ્તાવેજ',
            'calligraphy', 'સુલેખન', 'publication', 'પ્રકાશન',
        ],
    },
    "Science, Industry & Society": {
        "visvakosh": [
            'વિજ્ઞાન', 'ઔદ્યોગિક', 'ઉદ્યોગ', 'વિકાસ', 'ઉત્પાદન',
            'સેવા-ઉદ્યોગ', 'આંકડાશાસ્ત્ર', 'આંકડા', 'આંકડાશાસ્ત્રીય',
            'નમૂના', 'પ્રમાણ', 'હીરા', 'હીરો', 'હીરાઉદ્યોગ', 'કૅરેટ',
            'ખાણ', 'કિમ્બરલાઇટ', 'કાર્બન', 'વૃદ્ધિ', 'જનસંખ્યા',
            'સંસાધન', 'પર્યાવરણ', 'પગરખાં', 'ચામડું', 'હવામાન',
            'વાતાવરણ', 'તાપમાન', 'દબાણ', 'ભેજ', 'વરસાદ', 'પવન',
            'ચક્રવાત', 'આગાહી',
        ],
        "wikipedia": [
            'science', 'વિજ્ઞાન', 'industry', 'ઉદ્યોગ', 'industrial',
            'ઔદ્યોગિક', 'development', 'વિકાસ', 'statistics',
            'આંકડાશાસ્ત્ર', 'statistical', 'service industry',
            'diamond', 'હીરા', 'carat', 'કૅરેટ', 'mine', 'ખાણ', 'growth',
            'વૃદ્ધિ', 'population', 'જનસંખ્યા', 'resources', 'environment',
            'પર્યાવરણ', 'weather', 'હવામાન', 'temperature', 'તાપમાન',
            'rainfall', 'વરસાદ', 'climate', 'forecast',
        ],
    },
}


# ============================================================================
# TOKENIZER  (matches training script exactly)
# ============================================================================
class GujaratiTokenizer:
    GUJARATI_PATTERN = re.compile(r'[\u0A80-\u0AFF]+')
    ENGLISH_PATTERN = re.compile(r'[a-zA-Z]+')
    DIGIT_PATTERN = re.compile(r'[0-9]+')

    @classmethod
    def tokenize_words(cls, text: str) -> List[str]:
        if not text or not isinstance(text, str):
            return []
        return (cls.GUJARATI_PATTERN.findall(text)
                + cls.ENGLISH_PATTERN.findall(text)
                + cls.DIGIT_PATTERN.findall(text))

    @classmethod
    def tokenize_sentences(cls, text: str) -> List[str]:
        if not text or not isinstance(text, str):
            return []
        t = text.replace('।', '.')
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', t)
                if s.strip() and len(s.strip()) > 2]

    @classmethod
    def get_ngrams(cls, tokens: List[str], n: int) -> List[Tuple]:
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


# ============================================================================
# CATEGORY DETECTOR
# ============================================================================
def detect_category(text: str, source: str = None) -> Dict[str, Any]:
    """
    Detect which category the text belongs to, using the category keyword sets.
    `source` can be "Visvakosh" or "Wikipedia" to prioritize that source's
    keyword list. If None, both lists are used and the source is auto-detected.
    Returns:
        {
            "category": "<best category name or 'General / Unknown'>",
            "scores": {"<cat>": {"v_hits": N, "w_hits": N, "total": N, "words": [...]}}
            "best_source": "Visvakosh" | "Wikipedia" | "Mixed",
            "all_matches": [...],
        }
    """
    if not text or not isinstance(text, str):
        return {"category": "General / Unknown", "scores": {},
                "best_source": "Unknown", "all_matches": []}

    text_lower = text  # Gujarati has no case; keep as-is for English too

    scores: Dict[str, Dict[str, Any]] = {}
    all_matches: List[Dict[str, Any]] = []

    for cat, kw in CATEGORY_KEYWORDS.items():
        v_words = [w for w in kw["visvakosh"] if w and w in text]
        w_words = [w for w in kw["wikipedia"] if w and w in text]

        # Count actual occurrences (not just presence) for a better score
        v_hits = sum(text.count(w) for w in v_words)
        w_hits = sum(text.count(w) for w in w_words)

        # Bonus for the source-specific list
        if source == "Visvakosh":
            score = v_hits * 2 + w_hits
        elif source == "Wikipedia":
            score = w_hits * 2 + v_hits
        else:
            score = v_hits + w_hits

        scores[cat] = {
            "v_hits": v_hits,
            "w_hits": w_hits,
            "total": v_hits + w_hits,
            "score": score,
            "v_words": v_words,
            "w_words": w_words,
        }

        for w in v_words:
            all_matches.append({"category": cat, "word": w,
                                "count": text.count(w), "source": "Visvakosh"})
        for w in w_words:
            all_matches.append({"category": cat, "word": w,
                                "count": text.count(w), "source": "Wikipedia"})

    if not scores or all(s["score"] == 0 for s in scores.values()):
        return {"category": "General / Unknown", "scores": scores,
                "best_source": source or "Unknown", "all_matches": []}

    best_cat = max(scores.items(), key=lambda kv: kv[1]["score"])[0]
    best_info = scores[best_cat]

    if best_info["v_hits"] > best_info["w_hits"]:
        best_source = "Visvakosh"
    elif best_info["w_hits"] > best_info["v_hits"]:
        best_source = "Wikipedia"
    else:
        best_source = source or "Mixed"

    return {
        "category": best_cat,
        "scores": scores,
        "best_source": best_source,
        "all_matches": all_matches,
        "best_info": best_info,
    }


# ============================================================================
# STYLE MATRIX EXTRACTOR — 41 features (matches pickled scaler)
# ============================================================================
class GujaratiStyleMatrixExtractor:
    """
    Extracts the exact 41 style features that the pickled `StandardScaler`
    was fitted on. Feature ORDER matters and MUST match the training script.
    """
    def __init__(self):
        self.tokenizer = GujaratiTokenizer()

        self.v_markers = [
            'તથા', 'વળી', 'આથી', 'ગણાય', 'પ્રચલિત', 'આવાં', 'કેટલાંક',
            'અલબત્ત', 'તદુપરાંત', 'દા.ત.', 'જુઓ', 'એટલે કે', 'કહેવાય છે',
            'દા. ત.', 'તેમજ', 'ઉપરાંત', 'વિશેષ', 'અત્રે', 'તેવી જ રીતે',
            'એટલે', 'કહેવાય', 'કરાય છે', 'થાય છે', 'ઓળખાય છે', 'ગણાય છે',
            'સ્વયંસંચાલિત', 'અંકીય', 'ગણનયંત્ર', 'ભૌતિકવિજ્ઞાન',
        ]
        self.w_markers = [
            'શામેલ', 'ઘણીવાર', 'કોઈપણ', 'વ્યાખ્યાયિત', 'ઉદાહરણ તરીકે',
            'મોડેલ', 'સોફ્ટવેર', 'મુખ્ય લેખ', 'આ પણ જુઓ', 'જો કે',
            'દ્વારા', 'સંદર્ભ', 'બાહ્ય કડીઓ', 'સ્રોત', 'ટીકા', 'વિવાદ',
            'સક્ષમ', 'સમાવેશ', 'ઉલ્લેખ', 'પ્રોગ્રામ', 'ક્લસ્ટર',
            'કરવામાં આવે છે', 'આપવામાં આવે છે', 'બનાવવામાં આવે છે',
            'માનવામાં આવે છે', 'એપ્રિલ', 'મે', 'જૂન', 'ઓગસ્ટ',
            'નવેમ્બર', 'ડિસેમ્બર', 'તારીખ',
        ]
        self.v_passive = ['ગણાય છે', 'કરાય છે', 'કહેવાય છે', 'થાય છે', 'ઓળખાય છે']
        self.w_passive = ['કરવામાં આવે છે', 'આપવામાં આવે છે', 'બનાવવામાં આવે છે',
                          'માનવામાં આવે છે', 'કરવામાં આવ્યા હતા', 'કરવામાં આવ્યું હતું']
        self.definition_markers = ['એટલે', 'કહેવાય', 'ગણાય', 'રૂપે ઓળખાય', 'એટલે કે']
        self.traditional_translit = ['ૉ', 'ૅ', 'ઑ', 'ઍ']
        self.modern_translit = ['ો', 'ે', 'ૈ']

        self.english_letters = re.compile(r'[a-zA-Z]')
        self.gujarati_letters = re.compile(r'[\u0A80-\u0AFF]')

    FEATURE_NAMES = [
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
        'cnt_તથા', 'cnt_વળી', 'cnt_કહેવાય_છે', 'cnt_એટલે', 'cnt_કરાય_છે',
        'cnt_શામેલ', 'cnt_દ્વારા', 'cnt_સક્ષમ', 'cnt_ઉલ્લેખ', 'cnt_કરવામાં_આવે_છે',
        'translit_traditional_count', 'translit_modern_count',
        'translit_style_ratio',
    ]

    def extract_style_matrix(self, text: str) -> Dict[str, float]:
        if not text or not isinstance(text, str) or len(text) < 20:
            return self._empty()

        words = self.tokenizer.tokenize_words(text)
        sentences = self.tokenizer.tokenize_sentences(text)
        if len(words) < 5:
            return self._empty()

        wc = len(words)
        cc = len(text)
        sc = max(len(sentences), 1)

        f: Dict[str, float] = {}
        f['word_count'] = float(wc)
        f['log_word_count'] = float(np.log1p(wc))
        f['char_count'] = float(cc)
        f['log_char_count'] = float(np.log1p(cc))
        f['sentence_count'] = float(sc)
        f['avg_word_length'] = float(np.mean([len(w) for w in words])) if words else 0.0

        sl = [len(self.tokenizer.tokenize_words(s)) for s in sentences]
        sl = [l for l in sl if l > 0]
        f['avg_sentence_length'] = float(np.mean(sl)) if sl else 0.0
        f['std_sentence_length'] = float(np.std(sl)) if len(sl) > 1 else 0.0
        f['max_sentence_length'] = float(max(sl)) if sl else 0.0
        f['min_sentence_length'] = float(min(sl)) if sl else 0.0

        uniq = set(words)
        f['type_token_ratio'] = len(uniq) / wc if wc > 0 else 0.0
        f['hapax_ratio'] = (sum(1 for c in Counter(words).values() if c == 1)
                            / max(len(uniq), 1))

        v_c = sum(text.count(m) for m in self.v_markers)
        w_c = sum(text.count(m) for m in self.w_markers)
        f['v_markers_per_1000'] = (v_c / wc) * 1000 if wc > 0 else 0.0
        f['w_markers_per_1000'] = (w_c / wc) * 1000 if wc > 0 else 0.0
        f['marker_diff_per_1000'] = ((v_c - w_c) / wc) * 1000 if wc > 0 else 0.0
        f['marker_ratio'] = v_c / (v_c + w_c) if (v_c + w_c) > 0 else 0.5

        p_c = sum(text.count(m) for m in self.v_passive + self.w_passive)
        f['passive_per_1000'] = (p_c / wc) * 1000 if wc > 0 else 0.0

        eng_chars = len(self.english_letters.findall(text))
        guj_chars = len(self.gujarati_letters.findall(text))
        f['english_char_ratio'] = eng_chars / cc if cc > 0 else 0.0
        f['gujarati_char_ratio'] = guj_chars / cc if cc > 0 else 0.0
        f['script_ratio'] = guj_chars / (guj_chars + eng_chars + 1)

        f['colon_per_1000'] = (text.count(':') / wc) * 1000 if wc > 0 else 0.0
        f['comma_per_1000'] = (text.count(',') / wc) * 1000 if wc > 0 else 0.0
        f['paren_per_1000'] = ((text.count('(') + text.count(')')) / wc) * 1000 if wc > 0 else 0.0
        f['space_comma_per_1000'] = (len(re.findall(r'\s,', text)) / wc) * 1000 if wc > 0 else 0.0
        f['hyphen_per_1000'] = (text.count('-') / wc) * 1000 if wc > 0 else 0.0
        f['danda_per_1000'] = (text.count('।') / wc) * 1000 if wc > 0 else 0.0

        f['citation_count'] = float(len(re.findall(r'\[\d+\]', text)))
        f['wiki_heading_count'] = float(len(re.findall(r'==+.*?==+', text)))
        first_200 = text[:200]
        f['colon_in_first_200'] = 1.0 if ':' in first_200 else 0.0
        f['def_in_first_200'] = 1.0 if any(m in first_200
                                           for m in ['એટલે', 'કહેવાય', 'ગણાય']) else 0.0

        for m in ['તથા', 'વળી', 'કહેવાય છે', 'એટલે', 'કરાય છે',
                  'શામેલ', 'દ્વારા', 'સક્ષમ', 'ઉલ્લેખ', 'કરવામાં આવે છે']:
            f['cnt_' + m.replace(' ', '_')] = float(text.count(m))

        trad = sum(text.count(c) for c in self.traditional_translit)
        mod = sum(text.count(c) for c in self.modern_translit)
        f['translit_traditional_count'] = float(trad)
        f['translit_modern_count'] = float(mod)
        f['translit_style_ratio'] = trad / (trad + mod) if (trad + mod) > 0 else 0.5

        return f

    def _empty(self) -> Dict[str, float]:
        return {k: 0.0 for k in self.FEATURE_NAMES}


# ============================================================================
# SAFE PIPELINE — robust wrapper around the pickled training pipeline.
# ============================================================================
class SafePipeline:
    def __init__(self, raw_pipeline):
        self.raw = raw_pipeline
        self.extractor = GujaratiStyleMatrixExtractor()
        n = None
        try:
            if hasattr(raw_pipeline, "scaler") and hasattr(raw_pipeline.scaler, "n_features_in_"):
                n = int(raw_pipeline.scaler.n_features_in_)
        except Exception:
            n = None
        if n is None:
            try:
                n = len(raw_pipeline.scaler.mean_)
            except Exception:
                n = 41
        self.expected_style_features = n

    def _style_matrix(self, texts):
        names = GujaratiStyleMatrixExtractor.FEATURE_NAMES
        n_expect = self.expected_style_features
        rows = []
        for t in texts:
            f = self.extractor.extract_style_matrix(t)
            vals = [float(f.get(k, 0.0)) for k in names]
            if len(vals) < n_expect:
                vals = vals + [0.0] * (n_expect - len(vals))
            elif len(vals) > n_expect:
                vals = vals[:n_expect]
            rows.append(vals)
        arr = np.array(rows, dtype=float)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def transform(self, texts):
        raw = getattr(self, "raw", None)
        if raw is not None:
            try:
                return raw.transform(texts)
            except Exception:
                pass

        style_feats = self._style_matrix(texts)

        scaler = getattr(self, "raw", None)
        if scaler is not None and hasattr(scaler, "scaler"):
            try:
                scaled = scaler.scaler.transform(style_feats)
            except Exception:
                scaled = style_feats
        else:
            scaled = style_feats

        parts = [csr_matrix(scaled)]
        if scaler is not None and hasattr(scaler, "word_tfidf"):
            try:
                parts.append(scaler.word_tfidf.transform(texts))
            except Exception:
                pass
        if scaler is not None and hasattr(scaler, "char_tfidf"):
            try:
                parts.append(scaler.char_tfidf.transform(texts))
            except Exception:
                pass

        return hstack(parts).tocsr()


# ============================================================================
# Aliases so unpickling can find the classes it was saved with
# ============================================================================
Fpipe = SafePipeline
StyleExt = GujaratiStyleMatrixExtractor
Tk = GujaratiTokenizer
FeaturePipeline = SafePipeline
GujaratiStyleMatrixExtractor = GujaratiStyleMatrixExtractor


def _inject_into_main():
    injected = []
    candidates = {
        "SafePipeline": SafePipeline,
        "GujaratiStyleMatrixExtractor": GujaratiStyleMatrixExtractor,
        "StyleMatrixExtractor": GujaratiStyleMatrixExtractor,
        "GujaratiTokenizer": GujaratiTokenizer,
        "FeaturePipeline": SafePipeline,
        "Fpipe": SafePipeline,
        "StyleExt": GujaratiStyleMatrixExtractor,
        "Tk": GujaratiTokenizer,
    }
    for mod_name in ["main", "__main__"]:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        for k, v in candidates.items():
            setattr(mod, k, v)
        injected.append(mod_name)
    return injected


INJECTED_MODULES = _inject_into_main()

if "main" not in sys.modules:
    sys.modules["main"] = sys.modules[__name__]
    for k, v in {
        "SafePipeline": SafePipeline,
        "GujaratiStyleMatrixExtractor": GujaratiStyleMatrixExtractor,
        "GujaratiTokenizer": GujaratiTokenizer,
        "FeaturePipeline": SafePipeline,
        "Fpipe": SafePipeline,
        "StyleExt": GujaratiStyleMatrixExtractor,
        "Tk": GujaratiTokenizer,
    }.items():
        setattr(sys.modules["main"], k, v)
    INJECTED_MODULES.append("main (newly created)")


# ============================================================================
# NUMBERED CALCULATION TRACE
# ============================================================================
def build_calculation_trace(text: str) -> List[Dict[str, Any]]:
    ext = GujaratiStyleMatrixExtractor()
    tk = GujaratiTokenizer()
    feats = ext.extract_style_matrix(text)

    words = tk.tokenize_words(text)
    sentences = tk.tokenize_sentences(text)
    wc = max(len(words), 1)
    cc = max(len(text), 1)

    sent_lengths = [len(tk.tokenize_words(s)) for s in sentences]
    sent_lengths = [l for l in sent_lengths if l > 0]

    uniq = set(words)
    word_freq = Counter(words)
    hapax = [w for w, c in word_freq.items() if c == 1]

    v_hits = {m: text.count(m) for m in ext.v_markers}
    w_hits = {m: text.count(m) for m in ext.w_markers}
    v_total = sum(v_hits.values())
    w_total = sum(w_hits.values())

    p_hits = {m: text.count(m) for m in ext.v_passive + ext.w_passive}
    p_total = sum(p_hits.values())

    eng_chars = len(ext.english_letters.findall(text))
    guj_chars = len(ext.gujarati_letters.findall(text))

    trad = sum(text.count(c) for c in ext.traditional_translit)
    mod = sum(text.count(c) for c in ext.modern_translit)

    first_200 = text[:200]
    citation_n = len(re.findall(r'\[\d+\]', text))
    heading_n = len(re.findall(r'==+.*?==+', text))

    trace: List[Dict[str, Any]] = []
    n = [0]

    def add(name, formula, explanation, value):
        n[0] += 1
        trace.append({
            "n": n[0], "name": name, "value": value,
            "formula": formula, "explanation": explanation,
        })

    add("word_count", "len(tokenize_words(text))",
        f"Total word-tokens found: {len(words)}. "
        f"Tokenizer regex: [\\u0A80-\\u0AFF]+ | [a-zA-Z]+ | [0-9]+",
        feats['word_count'])

    add("log_word_count", "log1p(word_count)",
        f"log(1 + {len(words)}) = {feats['log_word_count']:.4f}  (smooths huge word counts)",
        feats['log_word_count'])

    add("char_count", "len(text)",
        f"Raw character count including spaces and punctuation: {len(text)}",
        feats['char_count'])

    add("log_char_count", "log1p(char_count)",
        f"log(1 + {len(text)}) = {feats['log_char_count']:.4f}",
        feats['log_char_count'])

    add("sentence_count", "non-empty sentences split on . ! ? |",
        f"Found {len(sentences)} sentences each with >2 chars.",
        feats['sentence_count'])

    add("avg_word_length", "mean(len(w) for w in words)",
        f"Sum of all word lengths / {len(words)} tokens = "
        f"{sum(len(w) for w in words)}/{len(words)} = {feats['avg_word_length']:.4f}",
        feats['avg_word_length'])

    if sent_lengths:
        add("avg_sentence_length", "mean(sent_lengths)",
            f"Sum of sentence word counts / number of sentences = "
            f"{sum(sent_lengths)}/{len(sent_lengths)} = {feats['avg_sentence_length']:.2f}",
            feats['avg_sentence_length'])
        add("std_sentence_length", "std(sent_lengths)",
            f"Standard deviation of sentence lengths = {feats['std_sentence_length']:.2f}",
            feats['std_sentence_length'])
        add("max_sentence_length", "max(sent_lengths)",
            f"Longest sentence: {int(feats['max_sentence_length'])} words",
            feats['max_sentence_length'])
        add("min_sentence_length", "min(sent_lengths)",
            f"Shortest sentence: {int(feats['min_sentence_length'])} words",
            feats['min_sentence_length'])
    else:
        for k in ['avg_sentence_length', 'std_sentence_length',
                  'max_sentence_length', 'min_sentence_length']:
            add(k, "(no sentences)", "Not enough text to compute.", 0.0)

    add("type_token_ratio", "len(unique_words) / word_count",
        f"Unique words = {len(uniq)}; ratio = {len(uniq)}/{len(words)} = "
        f"{feats['type_token_ratio']:.4f}",
        feats['type_token_ratio'])

    add("hapax_ratio", "count(words with freq == 1) / unique_words",
        f"Hapax words = {len(hapax)}; ratio = {len(hapax)}/{len(uniq)} = "
        f"{feats['hapax_ratio']:.4f}",
        feats['hapax_ratio'])

    add("v_markers_per_1000", "(v_marker_count / word_count) × 1000",
        f"Visvakosh markers hit: " +
        (", ".join(f"{m}×{c}" for m, c in v_hits.items() if c > 0) or "none") +
        f"; ({v_total}/{wc})×1000 = {feats['v_markers_per_1000']:.2f}",
        feats['v_markers_per_1000'])

    add("w_markers_per_1000", "(w_marker_count / word_count) × 1000",
        f"Wikipedia markers hit: " +
        (", ".join(f"{m}×{c}" for m, c in w_hits.items() if c > 0) or "none") +
        f"; ({w_total}/{wc})×1000 = {feats['w_markers_per_1000']:.2f}",
        feats['w_markers_per_1000'])

    add("marker_diff_per_1000", "(v_count − w_count) / word_count × 1000",
        f"({v_total} − {w_total})/{wc} × 1000 = {feats['marker_diff_per_1000']:.2f}",
        feats['marker_diff_per_1000'])

    add("marker_ratio", "v_count / (v_count + w_count)",
        f"{v_total}/({v_total}+{w_total}) = {feats['marker_ratio']:.4f}",
        feats['marker_ratio'])

    add("passive_per_1000", "(passive_count / word_count) × 1000",
        f"Passive hits: " +
        (", ".join(f"{m}×{c}" for m, c in p_hits.items() if c > 0) or "none") +
        f"; ({p_total}/{wc})×1000 = {feats['passive_per_1000']:.2f}",
        feats['passive_per_1000'])

    add("english_char_ratio", "english_char_count / char_count",
        f"English letters = {eng_chars}; {eng_chars}/{cc} = "
        f"{feats['english_char_ratio']:.4f}",
        feats['english_char_ratio'])

    add("gujarati_char_ratio", "gujarati_char_count / char_count",
        f"Gujarati letters = {guj_chars}; {guj_chars}/{cc} = "
        f"{feats['gujarati_char_ratio']:.4f}",
        feats['gujarati_char_ratio'])

    add("script_ratio", "guj_chars / (guj_chars + eng_chars + 1)",
        f"{guj_chars}/({guj_chars}+{eng_chars}+1) = {feats['script_ratio']:.4f} "
        f"(0=all English, 1=all Gujarati)",
        feats['script_ratio'])

    def punct(name, char, label):
        raw = text.count(char)
        add(name, f"(count of '{label}' / word_count) × 1000",
            f"'{label}' appears {raw}× ; ({raw}/{wc})×1000 = "
            f"{feats[name]:.2f}",
            feats[name])

    punct('colon_per_1000', ':', 'colon')
    punct('comma_per_1000', ',', 'comma')

    raw_paren = text.count('(') + text.count(')')
    add('paren_per_1000', "((count '(' + count ')') / word_count) × 1000",
        f"Parentheses appear {raw_paren}× ; ({raw_paren}/{wc})×1000 = "
        f"{feats['paren_per_1000']:.2f}",
        feats['paren_per_1000'])

    raw_sc = len(re.findall(r'\s,', text))
    add('space_comma_per_1000', "(count of ' ,' / word_count) × 1000",
        f"Space-before-comma '{raw_sc}' occurrences; ({raw_sc}/{wc})×1000 = "
        f"{feats['space_comma_per_1000']:.2f}",
        feats['space_comma_per_1000'])

    punct('hyphen_per_1000', '-', 'hyphen')
    punct('danda_per_1000', '।', 'danda (।)')

    add("citation_count", "len(re.findall(r'\\[\\d+\\]', text))",
        f"Matches of [number] like [1],[2] → {citation_n}",
        feats['citation_count'])

    add("wiki_heading_count", "len(re.findall(r'==+.*?==+', text))",
        f"Wiki-style == headings found → {heading_n}",
        feats['wiki_heading_count'])

    add("colon_in_first_200", "1 if ':' in text[:200] else 0",
        f"First 200 chars {'contain' if ':' in first_200 else 'do NOT contain'} a colon.",
        feats['colon_in_first_200'])

    add("def_in_first_200", "1 if any(એટલે | કહેવાય | ગણાય in text[:200]) else 0",
        f"Definition marker in first 200 chars: "
        f"{'yes' if feats['def_in_first_200'] else 'no'}",
        feats['def_in_first_200'])

    for m in ['તથા', 'વળી', 'કહેવાય છે', 'એટલે', 'કરાય છે',
              'શામેલ', 'દ્વારા', 'સક્ષમ', 'ઉલ્લેખ', 'કરવામાં આવે છે']:
        key = 'cnt_' + m.replace(' ', '_')
        raw = text.count(m)
        add(key, f"text.count('{m}')",
            f"Occurrences of '{m}' = {raw}",
            feats.get(key, 0.0))

    add("translit_traditional_count",
        "Σ text.count(c) for c in ['ૉ','ૅ','ઑ','ઍ']",
        f"Traditional characters found = {trad}",
        feats['translit_traditional_count'])

    add("translit_modern_count",
        "Σ text.count(c) for c in ['ો','ે','ૈ']",
        f"Modern characters found = {mod}",
        feats['translit_modern_count'])

    add("translit_style_ratio",
        "traditional / (traditional + modern)",
        f"{trad}/({trad}+{mod}) = {feats['translit_style_ratio']:.4f}",
        feats['translit_style_ratio'])

    return trace


# ============================================================================
# Rule-based classifier fallback
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
                'marker_metrics': {'visvakosh_markers_per_1000': 0, 'wikipedia_markers_per_1000': 0,
                                   'interpretation': 'N/A'},
                'passive_metrics': {'visvakosh_passive_per_1000': 0, 'wikipedia_passive_per_1000': 0,
                                    'interpretation': 'N/A'},
                'transliteration_metrics': {'traditional_count': 0, 'modern_count': 0,
                                            'interpretation': 'N/A'},
                'punctuation_metrics': {'colon_per_1000': 0, 'semicolon_per_1000': 0,
                                        'parentheses_per_1000': 0},
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
    .trace-row { display: grid; grid-template-columns: 48px 260px 1fr; gap: 8px;
                 padding: 6px 4px; border-bottom: 1px solid #eee; font-size: 0.88rem; }
    .trace-num { color: #1f4e79; font-weight: bold; }
    .trace-name { font-family: monospace; color: #333; }
    .trace-value { background: #f1f5f9; padding: 1px 6px; border-radius: 4px;
                   font-family: monospace; margin-right: 6px; color: #0b3d91; }
    .cat-banner { padding: 14px 20px; border-radius: 10px; margin: 10px 0;
                  font-size: 1.15rem; font-weight: bold; border-left: 8px solid; }
    .cat-banner-v { background: #d4edda; color: #155724; border-left-color: #28a745; }
    .cat-banner-w { background: #cce5ff; color: #004085; border-left-color: #0066cc; }
    .cat-banner-unk { background: #fff3cd; color: #856404; border-left-color: #ffc107; }
    .cat-word-chip { display: inline-block; padding: 2px 8px; border-radius: 10px;
                     margin: 2px; font-size: 0.8rem; font-family: monospace; }
    .chip-v { background: #28a745; color: white; }
    .chip-w { background: #0066cc; color: white; }
    .stTextArea textarea { font-family: 'Noto Sans Gujarati', 'Shruti', sans-serif; font-size: 15px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📚 Gujarati Source Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Rule-based Visvakosh vs Wikipedia classification '
            'with numbered calculation traces + category detection</div>', unsafe_allow_html=True)

if not STYLE_IMPORT_OK:
    st.warning(f"⚠️ `style_matrix_classifier.py` not found — rule-based classifier disabled. "
               f"({STYLE_IMPORT_ERR})")


# ============================================================================
# MODEL DISCOVERY + LOAD
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
    for path in discover_pkl_files():
        fname = os.path.relpath(path, SCRIPT_DIR)
        size = os.path.getsize(path)
        base = os.path.basename(path)

        if base in SKIP_FILES:
            continue
        if size < 200:
            continue
        try:
            data = joblib.load(path)
            if not isinstance(data, dict):
                continue
            name = data.get("model_name", base.replace(".pkl", ""))
            if "model" not in data or "feature_pipeline" not in data:
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
                           f"{name} (style feats: {safe_pipe.expected_style_features})"))
        except Exception:
            continue
    return models, status


if "ml_models" not in st.session_state:
    _m, _s = load_all_ml_models()
    st.session_state["ml_models"] = _m
    st.session_state["ml_status"] = _s

ALL_ML_MODELS = st.session_state["ml_models"]
LOAD_STATUS = st.session_state["ml_status"]


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("⚙️ About")
    st.info("**Rule-Based Classifier** + **Category Detection** + "
            "**ML Models Ensemble** (with numbered traces)")
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

    if st.button("🔄 Reload Models", use_container_width=True, key="reload_btn"):
        st.session_state.pop("ml_models", None)
        st.session_state.pop("ml_status", None)
        st.rerun()

    ok_count = sum(1 for _, _, ok, _ in LOAD_STATUS if ok)
    st.success(f"✅ {ok_count} models loaded")

    with st.expander(f"✅ Loaded ({ok_count})", expanded=(ok_count > 0)):
        for fname, size, ok, msg in LOAD_STATUS:
            if ok:
                st.write(f"✓ `{fname}` ({size:,} B) → **{msg}**")


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

        feats = GujaratiStyleMatrixExtractor().extract_style_matrix(text)
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
                reasons.append(f"Citations found ({feats['citation_count']:.0f})")
                signals.append(f"citations={feats['citation_count']:.0f}")
            if not reasons:
                reasons.append("Statistical profile matches Wikipedia training")
        else:
            if feats.get("v_markers_per_1000", 0) > feats.get("w_markers_per_1000", 0):
                reasons.append(f"Visvakosh markers dominate "
                               f"({feats['v_markers_per_1000']:.1f}/1000 vs "
                               f"{feats['w_markers_per_1000']:.1f}/1000)")
                signals.append(f"v_markers={feats['v_markers_per_1000']:.1f}")
            if feats.get("colon_in_first_200", 0) == 1:
                reasons.append("Definition-first pattern (colon in first 200 chars)")
                signals.append("def_colon")
            if not reasons:
                reasons.append("Statistical profile matches Visvakosh training")

        conf = out["confidence"]
        cw = "high" if conf > 0.85 else "moderate" if conf > 0.65 else "low"
        reasons.append(f"Confidence: {conf:.1%} ({cw})")
        if out["proba_v"] is not None:
            reasons.append(f"P(V)={out['proba_v']:.1%}, P(W)={out['proba_w']:.1%}")
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
# MAIN UI
# ============================================================================
st.header("📝 Enter Gujarati Text")
text_input = st.text_area(
    "Paste Gujarati paragraph here:",
    value=st.session_state.get('sample_text', ''),
    height=250,
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
        trace = build_calculation_trace(text_input)
        # Run category detection using the rule-based prediction as a hint
        cat_result = detect_category(text_input, source=result.get('prediction'))

    st.success("✅ Analysis complete")
    st.markdown("---")
    st.header("🎯 Prediction")

    pred = result['prediction']
    conf = result['confidence']
    if pred == "Visvakosh": css, emoji = "visvakosh-pred", "📖"
    elif pred == "Wikipedia": css, emoji = "wikipedia-pred", "🌐"
    else: css, emoji = "unknown-pred", "❓"

    st.markdown(
        f'<div class="prediction-box {css}">{emoji} Likely Source: '
        f'<strong>{pred}</strong><br>'
        f'<span style="font-size:1rem;">Confidence: {conf:.1%}</span></div>',
        unsafe_allow_html=True
    )

    # ---------------- CATEGORY DETECTION BANNER ----------------
    st.markdown("### 🏷️ Detected Category")
    cat_name = cat_result["category"]
    cat_source = cat_result["best_source"]

    if cat_source == "Visvakosh":
        banner_css = "cat-banner cat-banner-v"
        banner_emoji = "📖"
    elif cat_source == "Wikipedia":
        banner_css = "cat-banner cat-banner-w"
        banner_emoji = "🌐"
    else:
        banner_css = "cat-banner cat-banner-unk"
        banner_emoji = "❓"

    best_info = cat_result.get("best_info", {})
    vh = best_info.get("v_hits", 0)
    wh = best_info.get("w_hits", 0)

    st.markdown(
        f'<div class="{banner_css}">'
        f'{banner_emoji} <strong>{cat_source}</strong> — Category: '
        f'<strong>{cat_name}</strong><br>'
        f'<span style="font-size:0.9rem;font-weight:normal;">'
        f'Matched {vh} Visvakosh keyword(s) and {wh} Wikipedia keyword(s) '
        f'in this category.</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Show matched words as chips
    if best_info and (best_info.get("v_words") or best_info.get("w_words")):
        st.markdown("**🔑 Category keywords found in text:**")
        chips = []
        for w in best_info.get("v_words", []):
            cnt = text_input.count(w)
            chips.append(f'<span class="cat-word-chip chip-v">{w} ×{cnt}</span>')
        for w in best_info.get("w_words", []):
            cnt = text_input.count(w)
            chips.append(f'<span class="cat-word-chip chip-w">{w} ×{cnt}</span>')
        st.markdown(" ".join(chips), unsafe_allow_html=True)

    # Show all categories ranked
    with st.expander("📊 All category scores (ranked)"):
        scores = cat_result.get("scores", {})
        ranked = sorted(scores.items(), key=lambda kv: -kv[1]["score"])
        rows = []
        for cat, s in ranked:
            if s["score"] == 0:
                continue
            rows.append({
                "Category": cat,
                "Visvakosh hits": s["v_hits"],
                "Wikipedia hits": s["w_hits"],
                "Total": s["total"],
                "Score": s["score"],
                "V words": ", ".join(s["v_words"][:6]) + ("…" if len(s["v_words"]) > 6 else ""),
                "W words": ", ".join(s["w_words"][:6]) + ("…" if len(s["w_words"]) > 6 else ""),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No category keywords matched in this text.")

    v = result['votes']
    c1, c2, c3 = st.columns(3)
    c1.metric("📖 V Votes", v['visvakosh_total'])
    c2.metric("🌐 W Votes", v['wikipedia_total'])
    c3.metric("V Ratio", f"{v['visvakosh_ratio']:.1%}")
    st.progress(v['visvakosh_ratio'])

    # ---------------- NUMBERED CALCULATION TRACE ----------------
    st.markdown("---")
    st.header("🧮 Numbered Style Feature Calculation Trace")
    st.caption("Every style-matrix feature used by the ML models, with formula and "
               "the exact numbers that produced it.")

    with st.expander(f"🔬 Show all {len(trace)} numbered feature calculations",
                     expanded=False):
        for t in trace:
            val = t["value"]
            val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
            st.markdown(
                f'<div class="trace-row">'
                f'<div class="trace-num">#{t["n"]}</div>'
                f'<div class="trace-name">{t["name"]}</div>'
                f'<div><span class="trace-value">{val_str}</span>'
                f'<em>{t["formula"]}</em><br>{t["explanation"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    with st.expander("📋 Raw feature values (as a table)"):
        df = pd.DataFrame([
            {"#": t["n"], "Feature": t["name"], "Value": t["value"],
             "Formula": t["formula"], "Explanation": t["explanation"]}
            for t in trace
        ])
        st.dataframe(df, use_container_width=True, height=400)

    # ---------------- RULE-BY-RULE ----------------
    st.markdown("---")
    st.header("📋 Rule-by-Rule Breakdown")
    for r in result['rule_results']:
        vv, wv = r['visvakosh_votes'], r['wikipedia_votes']
        if vv == 0 and wv == 0:
            continue
        if vv > 0:
            st.markdown(f"**+{vv} Visvakosh**  {r['reason']}")
        if wv > 0:
            st.markdown(f"**+{wv} Wikipedia**  {r['reason']}")

    st.markdown("---")
    st.header("✅ Satisfied Style Properties")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📖 Visvakosh Indicators")
        if result['visvakosh_satisfied']:
            for r in result['visvakosh_satisfied']:
                st.markdown(f'<span class="satisfied-tag">{r["rule"]}</span>',
                            unsafe_allow_html=True)
        else:
            st.info("None met")
    with c2:
        st.subheader("🌐 Wikipedia Indicators")
        if result['wikipedia_satisfied']:
            for r in result['wikipedia_satisfied']:
                st.markdown(f'<span class="wikipedia-tag">{r["rule"]}</span>',
                            unsafe_allow_html=True)
        else:
            st.info("None met")

    # ---------------- ML MODELS ----------------
    st.markdown("---")
    st.header("🤖 ML Models — Individual Predictions & Reasoning")

    if not ALL_ML_MODELS:
        st.error("⚠️ 0 ML models loaded.")
    else:
        with st.spinner(f"Running {len(ALL_ML_MODELS)} ML models..."):
            ml_results = ml_predict_all(text_input)

        ml_results = [r for r in ml_results if not r["error"]]

        n_v = n_w = 0
        cv_sum = cw_sum = 0.0
        for r in ml_results:
            if r["prediction"] == "Visvakosh":
                n_v += 1; cv_sum += r["confidence"] or 0
            else:
                n_w += 1; cw_sum += r["confidence"] or 0

        total = n_v + n_w
        avg_v = cv_sum / n_v if n_v else 0
        avg_w = cw_sum / n_w if n_w else 0

        if n_v > n_w: ens_verdict, ens_css, ens_emoji = "Visvakosh", "visvakosh-pred", "📖"
        elif n_w > n_v: ens_verdict, ens_css, ens_emoji = "Wikipedia", "wikipedia-pred", "🌐"
        else: ens_verdict, ens_css, ens_emoji = "Tie", "unknown-pred", "⚖️"

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

        for idx, r in enumerate(sorted_results, 1):
            cc = "model-card-v" if r["prediction"] == "Visvakosh" else "model-card-w"
            icon = "📖" if r["prediction"] == "Visvakosh" else "🌐"
            conf = r["confidence"] or 0.5

            badges = []
            if r["val_v_ok"] and r["val_w_ok"]:
                badges.append("✅ both validations passed")
            badges.append(f"CV F1 = {r['cv_f1']:.4f}")
            badges.append(f"style feats = {r.get('expected_features','?')}")

            sig_html = "".join(
                f'<span class="model-signal">{s}</span>'
                for s in r.get("signals", [])
            )

            proba_html = ""
            if r["proba_v"] is not None and r["proba_w"] is not None:
                pv = r["proba_v"] * 100
                pw = r["proba_w"] * 100
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
                f'<div class="model-card {cc}">'
                f'<div class="model-name">#{idx} {icon} {r["model"]} → '
                f'<span style="color:{color};">{r["prediction"]}</span> '
                f'<span style="font-size:0.85rem;color:#666;">(confidence: {conf:.1%})</span></div>'
                f'<div style="font-size:0.85rem;color:#666;margin-top:4px;">{" • ".join(badges)}</div>'
                f'<div style="margin-top:6px;">{sig_html}</div>'
                f'<div class="model-reason">💡 <b>Why?</b> {r["reason"]}</div>'
                f'{proba_html}</div>',
                unsafe_allow_html=True
            )

            with st.expander(
                f"🧮 Show numbered calculation trace used by #{idx} {r['model']}"
            ):
                st.caption(
                    f"This model consumes the same {len(trace)} numbered style "
                    f"features listed below (then concatenated with word-TF-IDF "
                    f"and char-TF-IDF before being fed to the classifier). "
                    f"Scaler expects **{r.get('expected_features','?')}** style features."
                )
                for t in trace:
                    val = t["value"]
                    val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
                    st.markdown(
                        f'<div class="trace-row">'
                        f'<div class="trace-num">#{t["n"]}</div>'
                        f'<div class="trace-name">{t["name"]}</div>'
                        f'<div><span class="trace-value">{val_str}</span>'
                        f'<em>{t["formula"]}</em><br>{t["explanation"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        # ---------------- SUMMARY TABLE ----------------
        st.markdown("### 📊 Summary Table (numbered)")
        rows = []
        for i, r in enumerate(sorted_results, 1):
            rows.append({
                "#": i,
                "Model": r["model"],
                "Prediction": ("📖 " if r["prediction"] == "Visvakosh"
                               else "🌐 ") + r["prediction"],
                "Confidence": f"{r['confidence']:.1%}" if r["confidence"] else "—",
                "V %": f"{r['proba_v']:.1%}" if r["proba_v"] is not None else "—",
                "W %": f"{r['proba_w']:.1%}" if r["proba_w"] is not None else "—",
                "CV F1": f"{r['cv_f1']:.4f}",
                "V-val": "✓" if r["val_v_ok"] else "·",
                "W-val": "✓" if r["val_w_ok"] else "·"
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     height=min(600, 40 + 35 * len(rows)))

        # ---------------- DOWNLOADS ----------------
        ml_report = {
            "ensemble_verdict": ens_verdict,
            "votes": {"visvakosh": n_v, "wikipedia": n_w, "total": total},
            "category_detection": {
                "category": cat_result["category"],
                "best_source": cat_result["best_source"],
                "scores": {k: {"v_hits": v["v_hits"], "w_hits": v["w_hits"],
                               "score": v["score"], "total": v["total"]}
                           for k, v in cat_result.get("scores", {}).items()},
            },
            "feature_trace": trace,
            "models": [
                {
                    "rank": i,
                    "model": r["model"],
                    "prediction": r["prediction"],
                    "confidence": r["confidence"],
                    "proba_visvakosh": r["proba_v"],
                    "proba_wikipedia": r["proba_w"],
                    "cv_f1": r["cv_f1"],
                    "reason": r["reason"],
                    "signals": r["signals"],
                    "error": r["error"],
                }
                for i, r in enumerate(sorted_results, 1)
            ]
        }
        st.download_button(
            "⬇️ Download ML Predictions + Category + Feature Trace (JSON)",
            data=json.dumps(ml_report, indent=2, ensure_ascii=False, default=str),
            file_name=f"ml_predictions_{ens_verdict.lower()}.json",
            mime="application/json",
            use_container_width=True,
            key="download_ml_report"
        )
