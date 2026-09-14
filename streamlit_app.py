# ============================================================================
# streamlit_app.py — with NUMBERED CALCULATION TRACES for every ML result
# Uses the SAME feature extractor as the training script (GujaratiStyleMatrixExtractor)
# so pickled models load correctly WITHOUT retraining.
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
# TOKENIZER  (matches training script exactly)
# ============================================================================
class GujaratiTokenizer:
    GUJARATI_PATTERN = re.compile(r'[\u0A80-\u0AFF]+')
    ENGLISH_PATTERN = re.compile(r'[a-zA-Z]+')
    DIGIT_PATTERN = re.compile(r'[0-9]+')

    @classmethod
    def tokenize_words(cls, text: str) -> List[str]:
        if not text:
            return []
        gujarati = cls.GUJARATI_PATTERN.findall(text)
        english = cls.ENGLISH_PATTERN.findall(text)
        numbers = cls.DIGIT_PATTERN.findall(text)
        return [t for t in (gujarati + english + numbers) if len(t) > 0]

    @classmethod
    def tokenize_sentences(cls, text: str) -> List[str]:
        if not text:
            return []
        text = text.replace('।', '.')
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 2]

    @classmethod
    def get_ngrams(cls, tokens: List[str], n: int) -> List[Tuple]:
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


# ============================================================================
# STYLE MATRIX EXTRACTOR  (EXACT same as training script — 80+ features)
# ============================================================================
class GujaratiStyleMatrixExtractor:
    def __init__(self):
        self.tokenizer = GujaratiTokenizer()

        self.visvakosh_markers = [
            'તથા', 'વળી', 'આથી', 'ગણાય', 'પ્રચલિત', 'આવાં', 'કેટલાંક',
            'અલબત્ત', 'તદુપરાંત', 'દા.ત.', 'જુઓ', 'એટલે કે', 'કહેવાય છે',
            'દા. ત.', 'તેમજ', 'ઉપરાંત', 'વિશેષ', 'અત્રે', 'તેવી જ રીતે'
        ]
        self.wikipedia_markers = [
            'શામેલ', 'ઘણીવાર', 'કોઈપણ', 'વ્યાખ્યાયિત', 'ઉદાહરણ તરીકે',
            'મોડેલ', 'સોફ્ટવેર', 'ઓફ', 'મુખ્ય લેખ', 'આ પણ જુઓ', 'જો કે',
            'દ્વારા', 'સંદર્ભ', 'બાહ્ય કડીઓ', 'સ્રોત', 'ટીકા', 'વિવાદ'
        ]
        self.visvakosh_passive = ['ગણાય છે', 'કરાય છે', 'કહેવાય છે', 'થાય છે', 'ઓળખાય છે']
        self.wikipedia_passive = [
            'કરવામાં આવે છે', 'આપવામાં આવે છે', 'બનાવવામાં આવે છે',
            'માનવામાં આવે છે', 'કરવામાં આવ્યા હતા', 'કરવામાં આવ્યું હતું'
        ]
        self.definition_markers = ['એટલે', 'કહેવાય', 'ગણાય', 'રૂપે ઓળખાય', 'એટલે કે']
        self.traditional_translit = ['ૉ', 'ૅ', 'ઑ']
        self.modern_translit = ['ો', 'ે', 'ઓ']

        self.english_letters = re.compile(r'[a-zA-Z]')
        self.gujarati_letters = re.compile(r'[\u0A80-\u0AFF]')

    def extract_style_matrix(self, text: str) -> Dict[str, float]:
        features = {}
        if not text or len(text) < 10:
            return self._get_empty_matrix()

        words = self.tokenizer.tokenize_words(text)
        sentences = self.tokenizer.tokenize_sentences(text)
        if not words:
            return self._get_empty_matrix()

        word_count = len(words)
        char_count = len(text)
        sentence_count = max(len(sentences), 1)

        # BLOCK 1
        features['word_count'] = word_count
        features['char_count'] = char_count
        features['sentence_count'] = sentence_count
        features['log_word_count'] = np.log1p(word_count)
        features['log_char_count'] = np.log1p(char_count)
        features['avg_word_length'] = float(np.mean([len(w) for w in words]))

        sent_lengths = [len(self.tokenizer.tokenize_words(s)) for s in sentences]
        sent_lengths = [l for l in sent_lengths if l > 0]
        if sent_lengths:
            features['avg_sentence_length'] = float(np.mean(sent_lengths))
            features['std_sentence_length'] = float(np.std(sent_lengths)) if len(sent_lengths) > 1 else 0
            features['max_sentence_length'] = float(max(sent_lengths))
            features['min_sentence_length'] = float(min(sent_lengths))
            features['median_sentence_length'] = float(np.median(sent_lengths))
        else:
            for k in ['avg_sentence_length', 'std_sentence_length', 'max_sentence_length',
                      'min_sentence_length', 'median_sentence_length']:
                features[k] = 0

        # BLOCK 2
        unique_words = set(words)
        features['unique_word_count'] = len(unique_words)
        features['type_token_ratio'] = len(unique_words) / word_count

        word_freq = Counter(words)
        hapax = [w for w, c in word_freq.items() if c == 1]
        features['hapax_count'] = len(hapax)
        features['hapax_ratio'] = len(hapax) / len(unique_words) if unique_words else 0

        dis_legomena = [w for w, c in word_freq.items() if c == 2]
        features['dis_legomena_count'] = len(dis_legomena)
        features['dis_legomena_ratio'] = len(dis_legomena) / len(unique_words) if unique_words else 0

        freq_of_freq = Counter(word_freq.values())
        m1 = len(words)
        m2 = sum(f * (i ** 2) for i, f in freq_of_freq.items())
        features['yule_k'] = 10000 * (m2 - m1) / (m1 ** 2) if m1 > 0 else 0

        features['mattr_50'] = self._calculate_mattr(words, window=50)
        features['mattr_100'] = self._calculate_mattr(words, window=100)

        # BLOCK 3
        v_passive_count = sum(text.count(m) for m in self.visvakosh_passive)
        features['visvakosh_passive_count'] = v_passive_count
        features['visvakosh_passive_per_1000'] = (v_passive_count / word_count) * 1000

        w_passive_count = sum(text.count(m) for m in self.wikipedia_passive)
        features['wikipedia_passive_count'] = w_passive_count
        features['wikipedia_passive_per_1000'] = (w_passive_count / word_count) * 1000

        total_passive = v_passive_count + w_passive_count
        features['total_passive_count'] = total_passive
        features['total_passive_per_1000'] = (total_passive / word_count) * 1000

        passive_sentences = sum(1 for s in sentences
                                if any(m in s for m in self.visvakosh_passive + self.wikipedia_passive))
        features['passive_sentence_ratio'] = passive_sentences / sentence_count

        # BLOCK 4
        english_chars = len(self.english_letters.findall(text))
        features['english_char_count'] = english_chars
        features['english_char_ratio'] = english_chars / char_count if char_count > 0 else 0

        english_glosses = re.findall(r'\([A-Za-z][A-Za-z\s\.\-]+\)', text)
        features['english_gloss_count'] = len(english_glosses)
        features['english_glosses_per_1000'] = (len(english_glosses) / word_count) * 1000

        latin_tokens = re.findall(r'[A-Za-z]+', text)
        features['latin_token_count'] = len(latin_tokens)
        features['latin_token_ratio'] = len(latin_tokens) / word_count

        # BLOCK 5
        trad_count = sum(text.count(m) for m in self.traditional_translit)
        features['traditional_translit_count'] = trad_count
        features['traditional_translit_ratio'] = trad_count / char_count if char_count > 0 else 0

        modern_count = sum(text.count(m) for m in self.modern_translit)
        features['modern_translit_count'] = modern_count
        features['modern_translit_ratio'] = modern_count / char_count if char_count > 0 else 0

        total_translit = trad_count + modern_count
        features['translit_style_ratio'] = trad_count / total_translit if total_translit > 0 else 0.5

        # BLOCK 6
        features['colon_count'] = text.count(':')
        features['colon_per_1000'] = (features['colon_count'] / word_count) * 1000
        features['semicolon_count'] = text.count(';')
        features['parentheses_count'] = text.count('(') + text.count(')')
        features['parentheses_per_1000'] = (features['parentheses_count'] / word_count) * 1000
        features['comma_count'] = text.count(',')
        features['comma_per_1000'] = (features['comma_count'] / word_count) * 1000
        features['hyphen_count'] = text.count('-')
        features['danda_count'] = text.count('।')
        features['quote_count'] = text.count('"') + text.count('"') + text.count('"')
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['bracket_count'] = text.count('[') + text.count(']')

        # BLOCK 7
        v_marker_count = sum(text.count(m) for m in self.visvakosh_markers)
        features['visvakosh_marker_count'] = v_marker_count
        features['visvakosh_markers_per_1000'] = (v_marker_count / word_count) * 1000

        w_marker_count = sum(text.count(m) for m in self.wikipedia_markers)
        features['wikipedia_marker_count'] = w_marker_count
        features['wikipedia_markers_per_1000'] = (w_marker_count / word_count) * 1000

        total_markers = v_marker_count + w_marker_count
        features['marker_style_ratio'] = v_marker_count / total_markers if total_markers > 0 else 0.5

        for i, marker in enumerate(self.visvakosh_markers[:12]):
            features[f'v_marker_{i}'] = text.count(marker)
        for i, marker in enumerate(self.wikipedia_markers[:12]):
            features[f'w_marker_{i}'] = text.count(marker)

        # BLOCK 8
        first_200 = text[:200]
        first_100 = text[:100]
        features['colon_in_first_200'] = 1 if ':' in first_200 else 0
        features['colon_in_first_100'] = 1 if ':' in first_100 else 0

        def_marker_count = sum(text.count(m) for m in self.definition_markers)
        features['definition_marker_count'] = def_marker_count
        features['definition_markers_per_1000'] = (def_marker_count / word_count) * 1000

        if sentences:
            first_sent = sentences[0]
            features['first_sentence_has_colon'] = 1 if ':' in first_sent else 0
            features['first_sentence_has_definition'] = 1 if any(m in first_sent for m in self.definition_markers) else 0
            features['first_sentence_length'] = len(self.tokenizer.tokenize_words(first_sent))
        else:
            features['first_sentence_has_colon'] = 0
            features['first_sentence_has_definition'] = 0
            features['first_sentence_length'] = 0

        # BLOCK 9
        paragraphs = [p.strip() for p in re.split(r'\n+', text) if p.strip()]
        features['paragraph_count'] = max(len(paragraphs), 1)

        if paragraphs:
            para_lengths = [len(self.tokenizer.tokenize_words(p)) for p in paragraphs]
            para_lengths = [l for l in para_lengths if l > 0]
            if para_lengths:
                features['avg_paragraph_length'] = float(np.mean(para_lengths))
                features['std_paragraph_length'] = float(np.std(para_lengths)) if len(para_lengths) > 1 else 0
            else:
                features['avg_paragraph_length'] = 0
                features['std_paragraph_length'] = 0
        else:
            features['avg_paragraph_length'] = 0
            features['std_paragraph_length'] = 0

        # BLOCK 10
        if len(text) >= 2:
            char_bigrams = set(text[i:i+2] for i in range(len(text)-1))
            features['unique_char_bigrams'] = len(char_bigrams)
            features['char_bigram_ratio'] = len(char_bigrams) / char_count if char_count > 0 else 0
        else:
            features['unique_char_bigrams'] = 0
            features['char_bigram_ratio'] = 0

        if len(text) >= 3:
            char_trigrams = set(text[i:i+3] for i in range(len(text)-2))
            features['unique_char_trigrams'] = len(char_trigrams)
            features['char_trigram_ratio'] = len(char_trigrams) / char_count if char_count > 0 else 0
        else:
            features['unique_char_trigrams'] = 0
            features['char_trigram_ratio'] = 0

        # BLOCK 11
        if len(words) >= 2:
            word_bigrams = self.tokenizer.get_ngrams(words, 2)
            features['word_bigram_count'] = len(word_bigrams)
            features['unique_word_bigrams'] = len(set(word_bigrams))
            features['word_bigram_ratio'] = len(set(word_bigrams)) / len(word_bigrams) if word_bigrams else 0
        else:
            features['word_bigram_count'] = 0
            features['unique_word_bigrams'] = 0
            features['word_bigram_ratio'] = 0

        if len(words) >= 3:
            word_trigrams = self.tokenizer.get_ngrams(words, 3)
            features['word_trigram_count'] = len(word_trigrams)
            features['unique_word_trigrams'] = len(set(word_trigrams))
            features['word_trigram_ratio'] = len(set(word_trigrams)) / len(word_trigrams) if word_trigrams else 0
        else:
            features['word_trigram_count'] = 0
            features['unique_word_trigrams'] = 0
            features['word_trigram_ratio'] = 0

        # BLOCK 12
        gujarati_chars = len(self.gujarati_letters.findall(text))
        features['gujarati_char_count'] = gujarati_chars
        features['gujarati_char_ratio'] = gujarati_chars / char_count if char_count > 0 else 0
        features['script_ratio'] = gujarati_chars / (gujarati_chars + english_chars + 1)

        # BLOCK 13
        features['long_word_count'] = sum(1 for w in words if len(w) > 8)
        features['long_word_ratio'] = features['long_word_count'] / word_count
        features['short_word_count'] = sum(1 for w in words if len(w) <= 3)
        features['short_word_ratio'] = features['short_word_count'] / word_count

        return features

    def _calculate_mattr(self, words: List[str], window: int = 50) -> float:
        if len(words) < window:
            return len(set(words)) / len(words) if words else 0
        ttrs = []
        for i in range(len(words) - window + 1):
            window_words = words[i:i + window]
            ttrs.append(len(set(window_words)) / len(window_words))
        return float(np.mean(ttrs)) if ttrs else 0

    def _get_empty_matrix(self) -> Dict[str, float]:
        return {k: 0 for k in self._get_all_feature_names()}

    def _get_all_feature_names(self) -> List[str]:
        names = [
            'word_count', 'char_count', 'sentence_count', 'log_word_count', 'log_char_count',
            'avg_word_length', 'avg_sentence_length', 'std_sentence_length',
            'max_sentence_length', 'min_sentence_length', 'median_sentence_length',
            'unique_word_count', 'type_token_ratio', 'hapax_count', 'hapax_ratio',
            'dis_legomena_count', 'dis_legomena_ratio', 'yule_k', 'mattr_50', 'mattr_100',
            'visvakosh_passive_count', 'visvakosh_passive_per_1000',
            'wikipedia_passive_count', 'wikipedia_passive_per_1000',
            'total_passive_count', 'total_passive_per_1000', 'passive_sentence_ratio',
            'english_char_count', 'english_char_ratio', 'english_gloss_count',
            'english_glosses_per_1000', 'latin_token_count', 'latin_token_ratio',
            'traditional_translit_count', 'traditional_translit_ratio',
            'modern_translit_count', 'modern_translit_ratio', 'translit_style_ratio',
            'colon_count', 'colon_per_1000', 'semicolon_count',
            'parentheses_count', 'parentheses_per_1000', 'comma_count', 'comma_per_1000',
            'hyphen_count', 'danda_count', 'quote_count', 'exclamation_count',
            'question_count', 'bracket_count',
            'visvakosh_marker_count', 'visvakosh_markers_per_1000',
            'wikipedia_marker_count', 'wikipedia_markers_per_1000', 'marker_style_ratio',
            'colon_in_first_200', 'colon_in_first_100',
            'definition_marker_count', 'definition_markers_per_1000',
            'first_sentence_has_colon', 'first_sentence_has_definition', 'first_sentence_length',
            'paragraph_count', 'avg_paragraph_length', 'std_paragraph_length',
            'unique_char_bigrams', 'char_bigram_ratio',
            'unique_char_trigrams', 'char_trigram_ratio',
            'word_bigram_count', 'unique_word_bigrams', 'word_bigram_ratio',
            'word_trigram_count', 'unique_word_trigrams', 'word_trigram_ratio',
            'gujarati_char_count', 'gujarati_char_ratio', 'script_ratio',
            'long_word_count', 'long_word_ratio', 'short_word_count', 'short_word_ratio'
        ]
        for i in range(12):
            names.append(f'v_marker_{i}')
            names.append(f'w_marker_{i}')
        return names


# ============================================================================
# CALCULATION TRACER — explains each feature for a given text
# ============================================================================
def build_calculation_trace(text: str) -> List[Dict[str, Any]]:
    """
    Returns a numbered list of style-feature calculations:
      [{"n": 1, "name": "word_count", "value": 123, "formula": "...", "explanation": "..."},
       ...]
    """
    ext = GujaratiStyleMatrixExtractor()
    feats = ext.extract_style_matrix(text)
    tk = ext.tokenizer

    words = tk.tokenize_words(text)
    sentences = tk.tokenize_sentences(text)
    word_count = max(len(words), 1)
    char_count = max(len(text), 1)
    sentence_count = max(len(sentences), 1)

    sent_lengths = [len(tk.tokenize_words(s)) for s in sentences]
    sent_lengths = [l for l in sent_lengths if l > 0]

    v_marker_hits = {m: text.count(m) for m in ext.visvakosh_markers}
    w_marker_hits = {m: text.count(m) for m in ext.wikipedia_markers}
    v_marker_total = sum(v_marker_hits.values())
    w_marker_total = sum(w_marker_hits.values())

    v_passive_hits = {m: text.count(m) for m in ext.visvakosh_passive}
    w_passive_hits = {m: text.count(m) for m in ext.wikipedia_passive}

    trace = []
    n = 1

    def add(name, formula, explanation, value):
        nonlocal n
        trace.append({
            "n": n,
            "name": name,
            "value": value,
            "formula": formula,
            "explanation": explanation,
        })
        n += 1

    # ---- 1-6 basic length ----
    add("word_count", "len(tokenize_words(text))",
        f"Text has {len(words)} word-tokens (Gujarati + English + digit sequences).",
        feats['word_count'])

    add("char_count", "len(text)",
        f"Raw character count (spaces included).", feats['char_count'])

    add("sentence_count", "count of non-empty sentences split on . ! ? |",
        f"Found {len(sentences)} sentences (each > 2 chars).",
        feats['sentence_count'])

    add("log_word_count", "log1p(word_count)",
        f"Natural-log-smoothed word count: log(1 + {len(words)}) = {feats['log_word_count']:.4f}",
        feats['log_word_count'])

    add("log_char_count", "log1p(char_count)",
        f"log(1 + {len(text)}) = {feats['log_char_count']:.4f}",
        feats['log_char_count'])

    add("avg_word_length", "mean(len(w) for w in words)",
        f"Average word length across {len(words)} tokens.",
        feats['avg_word_length'])

    # ---- sentence metrics ----
    if sent_lengths:
        add("avg_sentence_length", "mean(sent_lengths)",
            f"Sum of sentence word-counts / number of sentences = "
            f"{sum(sent_lengths)}/{len(sent_lengths)} = {feats['avg_sentence_length']:.2f}",
            feats['avg_sentence_length'])
        add("max_sentence_length", "max(sent_lengths)",
            f"Longest sentence has {int(feats['max_sentence_length'])} words.",
            feats['max_sentence_length'])
        add("min_sentence_length", "min(sent_lengths)",
            f"Shortest sentence has {int(feats['min_sentence_length'])} words.",
            feats['min_sentence_length'])
    else:
        add("avg_sentence_length", "(no sentences)", "Not enough text.", 0)
        add("max_sentence_length", "(no sentences)", "Not enough text.", 0)
        add("min_sentence_length", "(no sentences)", "Not enough text.", 0)

    # ---- lexical ----
    uniq = set(words)
    add("type_token_ratio", "len(unique_words) / word_count",
        f"Unique words = {len(uniq)}; ratio = {len(uniq)}/{len(words)} = {feats['type_token_ratio']:.4f}",
        feats['type_token_ratio'])

    wf = Counter(words)
    hapax = [w for w, c in wf.items() if c == 1]
    add("hapax_ratio", "count(words appearing once) / unique_words",
        f"Hapax (freq=1) words = {len(hapax)}; ratio = {len(hapax)}/{len(uniq)} = {feats['hapax_ratio']:.4f}",
        feats['hapax_ratio'])

    # ---- markers ----
    add("visvakosh_marker_count", "Σ text.count(m) over Visvakosh markers",
        "Markers: " + ", ".join(f"{m}={c}" for m, c in v_marker_hits.items() if c > 0) or "none found",
        feats['visvakosh_marker_count'])

    add("visvakosh_markers_per_1000", "(visvakosh_marker_count / word_count) × 1000",
        f"({v_marker_total}/{word_count})×1000 = {feats['visvakosh_markers_per_1000']:.2f}",
        feats['visvakosh_markers_per_1000'])

    add("wikipedia_marker_count", "Σ text.count(m) over Wikipedia markers",
        "Markers: " + ", ".join(f"{m}={c}" for m, c in w_marker_hits.items() if c > 0) or "none found",
        feats['wikipedia_marker_count'])

    add("wikipedia_markers_per_1000", "(wikipedia_marker_count / word_count) × 1000",
        f"({w_marker_total}/{word_count})×1000 = {feats['wikipedia_markers_per_1000']:.2f}",
        feats['wikipedia_markers_per_1000'])

    add("marker_style_ratio", "v_marker_count / (v_marker_count + w_marker_count)",
        f"{v_marker_total}/{v_marker_total + w_marker_total} = {feats['marker_style_ratio']:.4f}",
        feats['marker_style_ratio'])

    # ---- passive ----
    add("visvakosh_passive_count", "Σ text.count(m) over Visvakosh passive forms",
        "Hits: " + ", ".join(f"{m}={c}" for m, c in v_passive_hits.items() if c > 0) or "none found",
        feats['visvakosh_passive_count'])

    add("wikipedia_passive_count", "Σ text.count(m) over Wikipedia passive forms",
        "Hits: " + ", ".join(f"{m}={c}" for m, c in w_passive_hits.items() if c > 0) or "none found",
        feats['wikipedia_passive_count'])

    # ---- english ----
    eng_chars = len(ext.english_letters.findall(text))
    add("english_char_ratio", "english_char_count / char_count",
        f"{eng_chars}/{char_count} = {feats['english_char_ratio']:.4f}",
        feats['english_char_ratio'])

    # ---- script ----
    guj_chars = len(ext.gujarati_letters.findall(text))
    add("gujarati_char_ratio", "gujarati_char_count / char_count",
        f"{guj_chars}/{char_count} = {feats['gujarati_char_ratio']:.4f}",
        feats['gujarati_char_ratio'])

    add("script_ratio", "gujarati_chars / (gujarati_chars + english_chars + 1)",
        f"{guj_chars}/({guj_chars}+{eng_chars}+1) = {feats['script_ratio']:.4f}",
        feats['script_ratio'])

    # ---- punctuation ----
    for k, label in [
        ('colon_per_1000', "colon"),
        ('comma_per_1000', "comma"),
    ]:
        raw = text.count(label[0])
        add(k, f"(count of {label} / word_count) × 1000",
            f"({raw}/{word_count})×1000 = {feats[k]:.2f}",
            feats[k])

    # hyphen (extractor only stores hyphen_count, so compute per-1000 here)
    hyphen_raw = text.count('-')
    hyphen_per_1000 = (hyphen_raw / word_count) * 1000
    add("hyphen_per_1000", "(count of hyphen / word_count) × 1000",
        f"({hyphen_raw}/{word_count})×1000 = {hyphen_per_1000:.2f}",
        hyphen_per_1000)

    # ---- structure ----
    citation_count = len(re.findall(r'\[\d+\]', text))
    add("citation_count", "len(re.findall(r'\\[\\d+\\]', text))",
        f"Citations like [1], [2]... found: {citation_count}",
        citation_count)

    add("colon_in_first_200", "1 if ':' in text[:200] else 0",
        f"First 200 chars {'contain' if feats['colon_in_first_200'] else 'do NOT contain'} a colon.",
        feats['colon_in_first_200'])

    # def_in_first_200 is NOT in the extractor — compute it locally
    first_200 = text[:200]
    def_in_first_200 = 1 if any(m in first_200 for m in ext.definition_markers) else 0
    add("def_in_first_200", "1 if any(એટલે|કહેવાય|ગણાય in text[:200]) else 0",
        f"Definition marker in first 200 chars: {'yes' if def_in_first_200 else 'no'}.",
        def_in_first_200)

    return trace

# ============================================================================
# Aliases for pickle compatibility
# ============================================================================
Fpipe = None  # set after SafePipeline defined
StyleExt = GujaratiStyleMatrixExtractor
Tk = GujaratiTokenizer
FeaturePipeline = None


# ============================================================================
# NO-OP SafePipeline — models load directly using the training extractor
# ============================================================================
class SafePipeline:
    """
    Wraps the pickled pipeline. Since our app's extractor now matches the
    training extractor exactly, we can call raw.transform() directly.
    """
    def __init__(self, raw_pipeline):
        self.raw = raw_pipeline
        try:
            n = raw_pipeline.scaler.n_features_in_
        except Exception:
            n = "?"
        self.expected_style_features = n

    def transform(self, texts):
        return self.raw.transform(texts)


Fpipe = SafePipeline
FeaturePipeline = SafePipeline


# ============================================================================
# Inject into main so pickle can find the classes
# ============================================================================
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
    for k, v in {
        "SafePipeline": SafePipeline,
        "GujaratiStyleMatrixExtractor": GujaratiStyleMatrixExtractor,
        "GujaratiTokenizer": GujaratiTokenizer,
        "FeaturePipeline": SafePipeline,
        "Fpipe": SafePipeline,
        "StyleExt": GujaratiStyleMatrixExtractor,
        "Tk": GujaratiTokenizer,
    }.items():
        if v is not None:
            setattr(sys.modules["main"], k, v)
    INJECTED_MODULES.append("main (newly created)")


# ============================================================================
# Rule-based classifier
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
    .trace-row { display: grid; grid-template-columns: 40px 220px 1fr; gap: 8px;
                 padding: 4px 0; border-bottom: 1px solid #eee; font-size: 0.88rem; }
    .trace-num { color: #1f4e79; font-weight: bold; }
    .trace-name { font-family: monospace; color: #333; }
    .trace-value { background: #f1f5f9; padding: 1px 6px; border-radius: 4px;
                   font-family: monospace; margin-right: 6px; }
    .stTextArea textarea { font-family: 'Noto Sans Gujarati', 'Shruti', sans-serif; font-size: 15px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📚 Gujarati Source Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Rule-based Visvakosh vs Wikipedia classification '
            'with numbered calculation traces for every model</div>', unsafe_allow_html=True)

if not STYLE_IMPORT_OK:
    st.warning(f"⚠️ `style_matrix_classifier.py` not found — rule-based classifier disabled.")


# ============================================================================
# LOAD MODELS
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
    st.info("**Rule-Based Classifier** + **ML Models Ensemble** (with full calculation traces)")
    st.markdown("---")
    st.header("📝 Sample Texts")
    sample_v = """કોમ્પ્યૂટર : વિવિધ કાર્યક્રમમાં આપેલી સૂચના અનુસાર માહિતીસંગ્રહ અને માહિતીપ્રક્રમણ માટેનું વીજાણુસાધન."""
    sample_w = """કમ્પ્યુટર એ એક ઇલેક્ટ્રોનિક ઉપકરણ છે જે માહિતીને સંગ્રહિત કરી શકે છે. આ ઉપકરણનો ઉપયોગ વિવિધ ક્ષેત્રોમાં કરવામાં આવે છે. [1][2]"""
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


# ============================================================================
# ML PREDICTION + REASONING + NUMBERED TRACE
# ============================================================================
def ml_predict_one(text, name, bundle, trace):
    model, pipeline = bundle["model"], bundle["pipeline"]
    out = {"model": name, "prediction": None, "confidence": None,
           "proba_v": None, "proba_w": None,
           "reason": "", "signals": [], "error": None,
           "cv_f1": bundle.get("cv_f1", 0.0),
           "val_v_ok": bundle.get("val_v_ok", False),
           "val_w_ok": bundle.get("val_w_ok", False),
           "source": bundle.get("source", "?"),
           "expected_features": bundle.get("expected_features", "?"),
           "trace": trace}

    try:
        X = pipeline.transform([text])
        if name in DENSE_ONLY:
            X = X.toarray()

        pred = model.predict(X)[0]
        out["prediction"] = "Wikipedia" if pred == 1 else "Visvakosh"

        # confidence
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

        # Build reason summary
        feats = {t["name"]: t["value"] for t in trace}
        reasons, signals = [], []

        if pred == 1:
            if feats.get("wikipedia_markers_per_1000", 0) > feats.get("visvakosh_markers_per_1000", 0):
                reasons.append(f"Wikipedia markers dominate "
                               f"({feats['wikipedia_markers_per_1000']:.1f}/1000 vs "
                               f"{feats['visvakosh_markers_per_1000']:.1f}/1000)")
                signals.append(f"w_markers={feats['wikipedia_markers_per_1000']:.1f}")
            if feats.get("english_char_ratio", 0) > 0.02:
                reasons.append(f"English glosses ({feats['english_char_ratio']:.1%})")
                signals.append(f"eng={feats['english_char_ratio']:.1%}")
            if feats.get("citation_count", 0) > 0:
                reasons.append(f"Citations found ({feats['citation_count']})")
                signals.append(f"citations={feats['citation_count']}")
            if not reasons:
                reasons.append("Statistical profile matches Wikipedia training")
        else:
            if feats.get("visvakosh_markers_per_1000", 0) > feats.get("wikipedia_markers_per_1000", 0):
                reasons.append(f"Visvakosh markers dominate "
                               f"({feats['visvakosh_markers_per_1000']:.1f}/1000 vs "
                               f"{feats['wikipedia_markers_per_1000']:.1f}/1000)")
                signals.append(f"v_markers={feats['visvakosh_markers_per_1000']:.1f}")
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


def ml_predict_all(text, trace):
    return [ml_predict_one(text, n, b, trace) for n, b in ALL_ML_MODELS.items()]


# ============================================================================
# MAIN INPUT
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

    st.success("✅ Analysis complete")
    st.markdown("---")
    st.header("🎯 Prediction")
    pred = result['prediction']; conf = result['confidence']
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

    # -------------- SHOW NUMBERED CALCULATION TRACE --------------
    st.markdown("---")
    st.header("🧮 Numbered Style Feature Calculation Trace")
    st.caption("Every style-matrix feature computed for your input, with formula and value.")

    with st.expander(f"🔬 Show all {len(trace)} numbered feature calculations", expanded=False):
        for t in trace:
            st.markdown(
                f'<div class="trace-row">'
                f'<div class="trace-num">#{t["n"]}</div>'
                f'<div class="trace-name">{t["name"]}</div>'
                f'<div><span class="trace-value">{t["value"] if not isinstance(t["value"], float) else round(t["value"], 4)}</span>'
                f'<em>{t["formula"]}</em><br>{t["explanation"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    with st.expander("📋 Raw feature values (as a table)"):
        df = pd.DataFrame([
            {"#": t["n"], "Feature": t["name"], "Value": t["value"], "Formula": t["formula"]}
            for t in trace
        ])
        st.dataframe(df, use_container_width=True, height=400)

    # ---------------- RULE-BY-RULE ----------------
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

    # ---------------- ML MODELS ----------------
    st.markdown("---")
    st.header("🤖 ML Models — Individual Predictions & Reasoning")

    if not ALL_ML_MODELS:
        st.error("⚠️ 0 ML models loaded.")
    else:
        with st.spinner(f"Running {len(ALL_ML_MODELS)} ML models..."):
            ml_results = ml_predict_all(text_input, trace)

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

        for idx, r in enumerate(sorted_results, 1):
            if r["error"]:
                st.markdown(
                    f'<div class="model-card model-card-err">'
                    f'<div class="model-name">#{idx} ❌ {r["model"]}</div>'
                    f'<div class="model-reason">{r["reason"]}</div></div>',
                    unsafe_allow_html=True
                )
                continue

            cc = "model-card-v" if r["prediction"] == "Visvakosh" else "model-card-w"
            icon = "📖" if r["prediction"] == "Visvakosh" else "🌐"
            conf = r["confidence"] or 0.5

            badges = []
            if r["val_v_ok"] and r["val_w_ok"]: badges.append("✅ both validations passed")
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
                f'<div class="model-card {cc}">'
                f'<div class="model-name">#{idx} {icon} {r["model"]} → '
                f'<span style="color:{color};">{r["prediction"]}</span> '
                f'<span style="font-size:0.85rem;color:#666;">(confidence: {conf:.1%})</span></div>'
                f'<div style="font-size:0.85rem;color:#666;margin-top:4px;">{" • ".join(badges)}</div>'
                f'<div style="margin-top:6px;">{sig_html}</div>'
                f'<div class="model-reason">💡 <b>Why?</b> {r["reason"]}</div>{proba_html}</div>',
                unsafe_allow_html=True
            )

            # Numbered feature trace per model
            with st.expander(f"🧮 Show numbered calculation trace used by #{idx} {r['model']}"):
                st.caption(
                    f"This model consumes the same numbered style features listed below "
                    f"(then concatenated with word-TF-IDF and char-TF-IDF). "
                    f"Scaler expects **{r.get('expected_features','?')}** style features."
                )
                for t in r["trace"]:
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

        # -------------- SUMMARY TABLE --------------
        st.markdown("### 📊 Summary Table (numbered)")
        rows = []
        for i, r in enumerate(sorted_results, 1):
            if r["error"]:
                rows.append({
                    "#": i, "Model": r["model"], "Prediction": "ERROR",
                    "Confidence": "—", "V %": "—", "W %": "—",
                    "CV F1": f"{r['cv_f1']:.4f}", "V-val": "—", "W-val": "—"
                })
            else:
                rows.append({
                    "#": i,
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

        # -------------- DOWNLOADS --------------
        ml_report = {
            "ensemble_verdict": ens_verdict,
            "votes": {"visvakosh": n_v, "wikipedia": n_w, "total": total},
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
                    "trace": r["trace"]
                }
                for i, r in enumerate(sorted_results, 1)
            ]
        }
        st.download_button(
            "⬇️ Download ML Predictions + Traces (JSON)",
            data=json.dumps(ml_report, indent=2, ensure_ascii=False, default=str),
            file_name=f"ml_predictions_{ens_verdict.lower()}.json",
            mime="application/json",
            use_container_width=True,
            key="download_ml_report"
        )
