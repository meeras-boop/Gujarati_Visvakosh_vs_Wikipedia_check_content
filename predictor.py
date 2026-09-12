# ============================================================================
# predictor.py
# Loads trained Visvakosh/Wikipedia classifier and provides detailed analysis
# FIXED VERSION - Handles pickle module mismatch
# ============================================================================

import pandas as pd
import numpy as np
import re
import os
import sys
import json
import joblib
from collections import Counter
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CUSTOM GUJARATI TOKENIZER (Must match training)
# ============================================================================

class GujaratiTokenizer:
    GUJARATI_PATTERN = re.compile(r'[\u0A80-\u0AFF]+')
    ENGLISH_PATTERN = re.compile(r'[a-zA-Z]+')
    DIGIT_PATTERN = re.compile(r'[0-9]+')

    @classmethod
    def tokenize_words(cls, text: str) -> List[str]:
        if not text:
            return []
        return [t for t in (cls.GUJARATI_PATTERN.findall(text) +
                            cls.ENGLISH_PATTERN.findall(text) +
                            cls.DIGIT_PATTERN.findall(text)) if len(t) > 0]

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
# STYLE MATRIX EXTRACTOR (Must match training exactly)
# ============================================================================

class GujaratiStyleMatrixExtractor:
    """
    Extract comprehensive style matrix features.
    MUST be identical to training extractor to ensure feature alignment.
    """

    def __init__(self):
        self.tokenizer = GujaratiTokenizer()

        self.visvakosh_markers = [
            'તથા', 'વળી', 'આથી', 'ગણાય', 'પ્રચલિત', 'આવાં', 'કેટલાંક',
            'અલબત્ત', 'તદુપરાંત', 'દા.ત.', 'જુઓ', 'એટલે કે', 'કહેવાય છે',
            'દા. ત.', 'વળી', 'તેમજ', 'ઉપરાંત'
        ]

        self.wikipedia_markers = [
            'શામેલ', 'ઘણીવાર', 'કોઈપણ', 'વ્યાખ્યાયિત', 'ઉદાહરણ તરીકે',
            'મોડેલ', 'સોફ્ટવેર', 'ઓફ', 'મુખ્ય લેખ', 'આ પણ જુઓ', 'જો કે',
            'દ્વારા', 'સંદર્ભ', 'બાહ્ય કડીઓ'
        ]

        self.visvakosh_passive = [
            'ગણાય છે', 'કરાય છે', 'કહેવાય છે', 'થાય છે', 'ઓળખાય છે'
        ]

        self.wikipedia_passive = [
            'કરવામાં આવે છે', 'આપવામાં આવે છે', 'બનાવવામાં આવે છે',
            'માનવામાં આવે છે', 'કરવામાં આવ્યા હતા', 'કરવામાં આવ્યું હતું',
            'હતું', 'હતા', 'હતી', 'છે'
        ]

        self.definition_markers = [
            'એટલે', 'કહેવાય', 'ગણાય', 'રૂપે ઓળખાય', 'ઓળખાય છે', 'એટલે કે'
        ]

        self.traditional_translit = ['ૉ', 'ૅ', 'ઑ']
        self.modern_translit = ['ો', 'ે', 'ઓ']

        self.english_letters = re.compile(r'[a-zA-Z]')

    def extract_style_matrix(self, text: str) -> Dict[str, float]:
        """Extract complete style matrix from text."""
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
        features['avg_word_length'] = np.mean([len(w) for w in words])

        sent_lengths = [len(self.tokenizer.tokenize_words(s)) for s in sentences]
        sent_lengths = [l for l in sent_lengths if l > 0]

        if sent_lengths:
            features['avg_sentence_length'] = np.mean(sent_lengths)
            features['std_sentence_length'] = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
            features['max_sentence_length'] = max(sent_lengths)
            features['min_sentence_length'] = min(sent_lengths)
            features['sentence_length_range'] = features['max_sentence_length'] - features['min_sentence_length']
        else:
            features['avg_sentence_length'] = 0
            features['std_sentence_length'] = 0
            features['max_sentence_length'] = 0
            features['min_sentence_length'] = 0
            features['sentence_length_range'] = 0

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

        passive_sentences = sum(
            1 for s in sentences
            if any(m in s for m in self.visvakosh_passive + self.wikipedia_passive)
        )
        features['passive_sentence_count'] = passive_sentences
        features['passive_sentence_ratio'] = passive_sentences / sentence_count
        features['active_sentence_ratio'] = 1 - features['passive_sentence_ratio']

        # BLOCK 4
        english_chars = len(self.english_letters.findall(text))
        features['english_char_count'] = english_chars
        features['english_char_ratio'] = english_chars / char_count

        english_glosses = re.findall(r'\([A-Za-z][A-Za-z\s\.\-]+\)', text)
        features['english_gloss_count'] = len(english_glosses)
        features['english_glosses_per_1000'] = (len(english_glosses) / word_count) * 1000

        latin_tokens = re.findall(r'[A-Za-z]+', text)
        features['latin_token_count'] = len(latin_tokens)
        features['latin_token_ratio'] = len(latin_tokens) / word_count

        trad_count = sum(text.count(m) for m in self.traditional_translit)
        features['traditional_translit_count'] = trad_count
        features['traditional_translit_ratio'] = trad_count / char_count

        modern_count = sum(text.count(m) for m in self.modern_translit)
        features['modern_translit_count'] = modern_count
        features['modern_translit_ratio'] = modern_count / char_count

        total_translit = trad_count + modern_count
        features['translit_style_ratio'] = trad_count / total_translit if total_translit > 0 else 0

        # BLOCK 5
        features['colon_count'] = text.count(':')
        features['colon_per_1000'] = (features['colon_count'] / word_count) * 1000

        features['semicolon_count'] = text.count(';')
        features['semicolon_per_1000'] = (features['semicolon_count'] / word_count) * 1000

        features['parentheses_count'] = text.count('(') + text.count(')')
        features['parentheses_per_1000'] = (features['parentheses_count'] / word_count) * 1000

        features['comma_count'] = text.count(',')
        features['comma_per_1000'] = (features['comma_count'] / word_count) * 1000

        features['hyphen_count'] = text.count('-')
        features['danda_count'] = text.count('।')
        features['quote_count'] = text.count('"') + text.count('"') + text.count('"')

        # BLOCK 6
        v_marker_count = sum(text.count(m) for m in self.visvakosh_markers)
        features['visvakosh_marker_count'] = v_marker_count
        features['visvakosh_markers_per_1000'] = (v_marker_count / word_count) * 1000

        w_marker_count = sum(text.count(m) for m in self.wikipedia_markers)
        features['wikipedia_marker_count'] = w_marker_count
        features['wikipedia_markers_per_1000'] = (w_marker_count / word_count) * 1000

        total_markers = v_marker_count + w_marker_count
        features['marker_style_ratio'] = v_marker_count / total_markers if total_markers > 0 else 0.5

        for i, marker in enumerate(self.visvakosh_markers[:10]):
            features[f'v_marker_{i}_{marker[:4]}'] = text.count(marker)

        for i, marker in enumerate(self.wikipedia_markers[:10]):
            features[f'w_marker_{i}_{marker[:4]}'] = text.count(marker)

        # BLOCK 7
        features['colon_in_first_200'] = 1 if ':' in text[:200] else 0
        features['colon_in_first_100'] = 1 if ':' in text[:100] else 0

        def_marker_count = sum(text.count(m) for m in self.definition_markers)
        features['definition_marker_count'] = def_marker_count
        features['definition_markers_per_1000'] = (def_marker_count / word_count) * 1000

        if sentences:
            first_sent = sentences[0]
            features['first_sentence_has_colon'] = 1 if ':' in first_sent else 0
            features['first_sentence_has_definition_marker'] = 1 if any(
                m in first_sent for m in self.definition_markers
            ) else 0
            features['first_sentence_length'] = len(self.tokenizer.tokenize_words(first_sent))
        else:
            features['first_sentence_has_colon'] = 0
            features['first_sentence_has_definition_marker'] = 0
            features['first_sentence_length'] = 0

        # BLOCK 8
        paragraphs = [p.strip() for p in re.split(r'\n+', text) if p.strip()]
        features['paragraph_count'] = max(len(paragraphs), 1)

        if paragraphs:
            para_lengths = [len(self.tokenizer.tokenize_words(p)) for p in paragraphs]
            para_lengths = [l for l in para_lengths if l > 0]
            if para_lengths:
                features['avg_paragraph_length'] = np.mean(para_lengths)
                features['std_paragraph_length'] = np.std(para_lengths) if len(para_lengths) > 1 else 0
            else:
                features['avg_paragraph_length'] = 0
                features['std_paragraph_length'] = 0
        else:
            features['avg_paragraph_length'] = 0
            features['std_paragraph_length'] = 0

        # BLOCK 9
        if len(text) >= 2:
            char_bigrams = set([text[i:i+2] for i in range(len(text)-1)])
            features['unique_char_bigrams'] = len(char_bigrams)
            features['char_bigram_ratio'] = len(char_bigrams) / char_count
        else:
            features['unique_char_bigrams'] = 0
            features['char_bigram_ratio'] = 0

        if len(text) >= 3:
            char_trigrams = set([text[i:i+3] for i in range(len(text)-2)])
            features['unique_char_trigrams'] = len(char_trigrams)
            features['char_trigram_ratio'] = len(char_trigrams) / char_count
        else:
            features['unique_char_trigrams'] = 0
            features['char_trigram_ratio'] = 0

        # BLOCK 10
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

        return features

    def _calculate_mattr(self, words: List[str], window: int = 50) -> float:
        if len(words) < window:
            return len(set(words)) / len(words) if words else 0
        ttrs = []
        for i in range(len(words) - window + 1):
            window_words = words[i:i + window]
            ttr = len(set(window_words)) / len(window_words)
            ttrs.append(ttr)
        return np.mean(ttrs) if ttrs else 0

    def _get_empty_matrix(self) -> Dict[str, float]:
        template = self.extract_style_matrix("આ એક પરીક્ષણ લખાણ છે. તે ખૂબ ટૂંકું છે.")
        return {k: 0 for k in template.keys()}


# ============================================================================
# MODEL CLASS
# ============================================================================

class VisvakoshWikipediaClassifier:
    """Classifier class - must match training class for joblib.load to work."""

    def __init__(self):
        self.feature_extractor = GujaratiStyleMatrixExtractor()
        self.scaler = None
        self.classifier = None
        self.word_tfidf = None
        self.char_tfidf = None
        self.feature_names = None
        self.is_fitted = False
        self.training_profile = {}

    def extract_style_features(self, texts: List[str]) -> np.ndarray:
        all_features = []
        for text in texts:
            features = self.feature_extractor.extract_style_matrix(text)
            all_features.append(features)

        feature_df = pd.DataFrame(all_features)
        feature_df = feature_df.fillna(0)
        feature_df = feature_df.replace([np.inf, -np.inf], 0)

        if self.feature_names is not None:
            for col in self.feature_names:
                if col not in feature_df.columns:
                    feature_df[col] = 0
            feature_df = feature_df[self.feature_names]

        return feature_df.values

    def predict(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Classifier not fitted.")
        from scipy.sparse import hstack, csr_matrix

        style_features = self.extract_style_features(texts)
        word_features = self.word_tfidf.transform(texts)
        char_features = self.char_tfidf.transform(texts)

        style_scaled = self.scaler.transform(style_features)
        style_sparse = csr_matrix(style_scaled)

        X = hstack([style_sparse, word_features, char_features])
        return self.classifier.predict(X)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Classifier not fitted.")
        from scipy.sparse import hstack, csr_matrix

        style_features = self.extract_style_features(texts)
        word_features = self.word_tfidf.transform(texts)
        char_features = self.char_tfidf.transform(texts)

        style_scaled = self.scaler.transform(style_features)
        style_sparse = csr_matrix(style_scaled)

        X = hstack([style_sparse, word_features, char_features])
        return self.classifier.predict_proba(X)

    @classmethod
    def load(cls, filepath: str):
        """
        Load model with pickle module mismatch fix.
        Registers classes under both 'predictor' and '__main__' so pickle can find them.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        # ────────────────────────────────────────────────────────────────
        # FIX: Register classes under __main__ so joblib/pickle can find them
        # ────────────────────────────────────────────────────────────────
        import __main__
        __main__.GujaratiTokenizer = GujaratiTokenizer
        __main__.GujaratiStyleMatrixExtractor = GujaratiStyleMatrixExtractor
        __main__.VisvakoshWikipediaClassifier = VisvakoshWikipediaClassifier

        # Also ensure this module is registered
        current_module = sys.modules[__name__]
        sys.modules['predictor'] = current_module
        sys.modules['main'] = current_module  # In case pickle used 'main'

        model_data = joblib.load(filepath)

        classifier = cls()
        classifier.classifier = model_data['classifier']
        classifier.scaler = model_data['scaler']
        classifier.word_tfidf = model_data['word_tfidf']
        classifier.char_tfidf = model_data['char_tfidf']
        classifier.feature_names = model_data['feature_names']
        classifier.feature_extractor = model_data['feature_extractor']
        classifier.training_profile = model_data.get('training_profile', {})
        classifier.is_fitted = model_data.get('is_fitted', True)

        return classifier


# ============================================================================
# DETAILED ANALYZER (unchanged)
# ============================================================================

class DetailedAnalyzer:
    def __init__(self, classifier: VisvakoshWikipediaClassifier):
        self.classifier = classifier
        self.extractor = classifier.feature_extractor

    def analyze(self, text: str) -> Dict[str, Any]:
        features = self.extractor.extract_style_matrix(text)
        prediction = self.classifier.predict([text])[0]
        probability = self.classifier.predict_proba([text])[0]
        style_scores = self._compute_style_scores(features)
        quant_analysis = self._quantitative_analysis(features)
        qual_analysis = self._qualitative_analysis(features, text)
        satisfied = self._compute_satisfied_properties(features, prediction, probability)
        predicted_source = 'Wikipedia' if prediction == 1 else 'Visvakosh'

        return {
            'prediction': predicted_source,
            'confidence': float(max(probability)),
            'probabilities': {
                'Visvakosh': float(probability[0]),
                'Wikipedia': float(probability[1])
            },
            'raw_features': features,
            'style_scores': style_scores,
            'quantitative_analysis': quant_analysis,
            'qualitative_analysis': qual_analysis,
            'satisfied_properties': satisfied,
            'summary': self._generate_summary(predicted_source, style_scores, satisfied)
        }

    def _compute_style_scores(self, features: Dict[str, float]) -> Dict[str, float]:
        scores = {}
        scores['visvakosh_marker_score'] = features.get('visvakosh_markers_per_1000', 0)
        scores['definition_first_score'] = features.get('colon_in_first_200', 0) * 100
        scores['traditional_translit_score'] = features.get('traditional_translit_ratio', 0) * 1000
        scores['concise_passive_score'] = features.get('visvakosh_passive_per_1000', 0)

        scores['wikipedia_marker_score'] = features.get('wikipedia_markers_per_1000', 0)
        scores['modern_translit_score'] = features.get('modern_translit_ratio', 0) * 1000
        scores['extended_passive_score'] = features.get('wikipedia_passive_per_1000', 0)

        v_score = (scores['visvakosh_marker_score'] +
                   scores['definition_first_score'] +
                   scores['concise_passive_score'])
        w_score = (scores['wikipedia_marker_score'] +
                   scores['extended_passive_score'])

        total = v_score + w_score
        scores['visvakosh_style_ratio'] = v_score / total if total > 0 else 0.5
        return {k: round(v, 4) for k, v in scores.items()}

    def _quantitative_analysis(self, features: Dict[str, float]) -> Dict[str, Any]:
        return {
            'length_metrics': {
                'word_count': int(features.get('word_count', 0)),
                'character_count': int(features.get('char_count', 0)),
                'sentence_count': int(features.get('sentence_count', 0)),
                'paragraph_count': int(features.get('paragraph_count', 0)),
                'avg_word_length': round(features.get('avg_word_length', 0), 2),
            },
            'sentence_metrics': {
                'avg_sentence_length': round(features.get('avg_sentence_length', 0), 2),
                'std_sentence_length': round(features.get('std_sentence_length', 0), 2),
                'max_sentence_length': int(features.get('max_sentence_length', 0)),
                'min_sentence_length': int(features.get('min_sentence_length', 0)),
                'interpretation': self._interpret_sentence_length(features.get('avg_sentence_length', 0))
            },
            'vocabulary_metrics': {
                'unique_words': int(features.get('unique_word_count', 0)),
                'type_token_ratio': round(features.get('type_token_ratio', 0), 4),
                'hapax_count': int(features.get('hapax_count', 0)),
                'hapax_ratio': round(features.get('hapax_ratio', 0), 4),
                'yule_k': round(features.get('yule_k', 0), 2),
                'mattr_50': round(features.get('mattr_50', 0), 4),
                'mattr_100': round(features.get('mattr_100', 0), 4),
                'interpretation': self._interpret_ttr(features.get('type_token_ratio', 0))
            },
            'voice_metrics': {
                'total_passive_count': int(features.get('total_passive_count', 0)),
                'total_passive_per_1000': round(features.get('total_passive_per_1000', 0), 2),
                'visvakosh_passive_per_1000': round(features.get('visvakosh_passive_per_1000', 0), 2),
                'wikipedia_passive_per_1000': round(features.get('wikipedia_passive_per_1000', 0), 2),
                'passive_sentence_ratio': round(features.get('passive_sentence_ratio', 0), 4),
                'interpretation': self._interpret_passive(features.get('passive_sentence_ratio', 0))
            },
            'transliteration_metrics': {
                'traditional_translit_count': int(features.get('traditional_translit_count', 0)),
                'traditional_translit_ratio': round(features.get('traditional_translit_ratio', 0), 6),
                'modern_translit_count': int(features.get('modern_translit_count', 0)),
                'modern_translit_ratio': round(features.get('modern_translit_ratio', 0), 6),
                'translit_style_ratio': round(features.get('translit_style_ratio', 0), 4),
                'interpretation': self._interpret_transliteration(features.get('translit_style_ratio', 0))
            },
            'punctuation_metrics': {
                'colon_per_1000': round(features.get('colon_per_1000', 0), 2),
                'semicolon_per_1000': round(features.get('semicolon_per_1000', 0), 2),
                'parentheses_per_1000': round(features.get('parentheses_per_1000', 0), 2),
                'comma_per_1000': round(features.get('comma_per_1000', 0), 2),
                'interpretation': self._interpret_punctuation(
                    features.get('colon_per_1000', 0),
                    features.get('parentheses_per_1000', 0))
            },
            'english_usage_metrics': {
                'english_char_ratio': round(features.get('english_char_ratio', 0), 4),
                'english_gloss_count': int(features.get('english_gloss_count', 0)),
                'english_glosses_per_1000': round(features.get('english_glosses_per_1000', 0), 2),
                'latin_token_ratio': round(features.get('latin_token_ratio', 0), 4),
                'interpretation': self._interpret_english_usage(features.get('english_glosses_per_1000', 0))
            }
        }

    def _qualitative_analysis(self, features: Dict[str, float], text: str) -> Dict[str, Any]:
        return {
            'tone': {
                'definition_first': bool(features.get('colon_in_first_200', 0)),
                'definition_style': self._classify_definition_style(features),
                'encyclopedic_style': self._classify_encyclopedic(features),
                'narrative_vs_expository': self._classify_narrative(features),
            },
            'register': {
                'formality': self._classify_formality(features),
                'technical_density': self._classify_technical_density(features),
                'terminology_glossing': 'High (Gujarati → English)'
                    if features.get('english_glosses_per_1000', 0) > 15
                    else 'Moderate' if features.get('english_glosses_per_1000', 0) > 5
                    else 'Low',
            },
            'structural_style': {
                'paragraph_density': self._classify_paragraph_density(features),
                'sectioning': self._detect_sectioning(text),
                'list_usage': self._detect_lists(text),
                'citation_presence': self._detect_citations(text),
            },
            'vocabulary_style': {
                'function_words': self._classify_function_words(features),
                'technical_terms': self._classify_technical_terms(features),
                'rare_word_usage': self._classify_rare_words(features),
            },
            'grammatical_style': {
                'passive_voice_style': self._classify_passive_style(features),
                'sentence_complexity': self._classify_sentence_complexity(features),
                'tense_usage': self._classify_tense(text),
            }
        }

    def _compute_satisfied_properties(self, features, prediction, probability) -> Dict[str, Any]:
        predicted_source = 'Wikipedia' if prediction == 1 else 'Visvakosh'
        confidence = float(max(probability))

        visvakosh_thresholds = {
            'definition_first': features.get('colon_in_first_200', 0) == 1,
            'high_visvakosh_markers': features.get('visvakosh_markers_per_1000', 0) > 5,
            'traditional_transliteration': features.get('traditional_translit_ratio', 0) > 0.001,
            'shorter_sentences': features.get('avg_sentence_length', 0) < 20,
            'high_ttr': features.get('type_token_ratio', 0) > 0.5,
            'concise_passive': features.get('visvakosh_passive_per_1000', 0) > 2,
            'high_english_glosses': features.get('english_glosses_per_1000', 0) > 10,
            'high_colon_usage': features.get('colon_per_1000', 0) > 8,
            'high_parentheses': features.get('parentheses_per_1000', 0) > 30,
        }

        wikipedia_thresholds = {
            'wikipedia_markers': features.get('wikipedia_markers_per_1000', 0) > 2,
            'modern_transliteration': features.get('modern_translit_ratio', 0) > 0.02,
            'longer_sentences': features.get('avg_sentence_length', 0) > 18,
            'extended_passive': features.get('wikipedia_passive_per_1000', 0) > 10,
            'lower_ttr': features.get('type_token_ratio', 0) < 0.55,
            'sectioning': self._detect_sectioning_count(text=None) > 0,
            'low_english_glosses': features.get('english_glosses_per_1000', 0) < 10,
            'extended_tense': features.get('wikipedia_passive_per_1000', 0) > 15,
        }

        if predicted_source == 'Visvakosh':
            satisfied = {k: v for k, v in visvakosh_thresholds.items() if v}
            unsatisfied = {k: v for k, v in visvakosh_thresholds.items() if not v}
        else:
            satisfied = {k: v for k, v in wikipedia_thresholds.items() if v}
            unsatisfied = {k: v for k, v in wikipedia_thresholds.items() if not v}

        total = len(satisfied) + len(unsatisfied)
        satisfaction_ratio = len(satisfied) / total if total > 0 else 0

        return {
            'predicted_source': predicted_source,
            'confidence': confidence,
            'satisfied_properties': satisfied,
            'unsatisfied_properties': unsatisfied,
            'satisfaction_ratio': round(satisfaction_ratio, 4),
            'total_checked': total,
            'total_satisfied': len(satisfied)
        }

    def _interpret_sentence_length(self, avg_len: float) -> str:
        if avg_len < 15:
            return "Short sentences → Visvakosh pattern"
        elif avg_len < 20:
            return "Medium sentences → Borderline"
        return "Long sentences → Wikipedia pattern"

    def _interpret_ttr(self, ttr: float) -> str:
        if ttr > 0.55:
            return "High lexical diversity → Visvakosh pattern"
        elif ttr > 0.45:
            return "Moderate lexical diversity → Borderline"
        return "Low lexical diversity → Wikipedia pattern"

    def _interpret_passive(self, ratio: float) -> str:
        if ratio < 0.15:
            return "Low passive usage → Visvakosh pattern"
        elif ratio < 0.30:
            return "Moderate passive usage → Borderline"
        return "High passive usage → Wikipedia pattern"

    def _interpret_transliteration(self, ratio: float) -> str:
        if ratio > 0.6:
            return "Traditional transliteration dominant → Visvakosh style"
        elif ratio > 0.3:
            return "Mixed transliteration → Borderline"
        return "Modern transliteration dominant → Wikipedia style"

    def _interpret_punctuation(self, colon: float, paren: float) -> str:
        if colon > 8 and paren > 30:
            return "High colon + parentheses → Visvakosh pattern"
        elif colon < 5 and paren < 20:
            return "Low colon + parentheses → Wikipedia pattern"
        return "Moderate punctuation → Borderline"

    def _interpret_english_usage(self, gloss_per_1000: float) -> str:
        if gloss_per_1000 > 15:
            return "High English glossing → Visvakosh pattern"
        elif gloss_per_1000 > 5:
            return "Moderate English glossing → Borderline"
        return "Low English glossing → Wikipedia pattern"

    def _classify_definition_style(self, features) -> str:
        if features.get('colon_in_first_200', 0) == 1:
            if features.get('first_sentence_has_colon', 0) == 1:
                return "Term : Definition (strong Visvakosh signature)"
            return "Colon appears early (Visvakosh-like)"
        return "No early definition (Wikipedia-like)"

    def _classify_encyclopedic(self, features) -> str:
        marker_ratio = features.get('marker_style_ratio', 0.5)
        if marker_ratio > 0.7:
            return "Traditional encyclopedic (Visvakosh)"
        elif marker_ratio < 0.3:
            return "Modern encyclopedic (Wikipedia)"
        return "Neutral encyclopedic"

    def _classify_narrative(self, features) -> str:
        avg_sent = features.get('avg_sentence_length', 15)
        std_sent = features.get('std_sentence_length', 5)
        if std_sent / max(avg_sent, 1) > 0.6:
            return "Varied narrative style"
        return "Consistent expository style"

    def _classify_formality(self, features) -> str:
        markers = features.get('visvakosh_markers_per_1000', 0) + features.get('wikipedia_markers_per_1000', 0)
        if markers > 10:
            return "High formal register"
        return "Moderate formal register"

    def _classify_technical_density(self, features) -> str:
        gloss = features.get('english_glosses_per_1000', 0)
        if gloss > 20:
            return "High technical density with glosses"
        elif gloss > 8:
            return "Moderate technical density"
        return "Low technical density"

    def _classify_paragraph_density(self, features) -> str:
        avg_para = features.get('avg_paragraph_length', 0)
        if avg_para > 200:
            return "Dense continuous paragraphs (Visvakosh)"
        elif avg_para > 80:
            return "Medium-length paragraphs"
        return "Short segmented paragraphs (Wikipedia)"

    def _classify_function_words(self, features) -> str:
        v = features.get('visvakosh_markers_per_1000', 0)
        w = features.get('wikipedia_markers_per_1000', 0)
        if v > w * 2:
            return "Visvakosh function words dominant (તથા, વળી, આથી)"
        elif w > v * 2:
            return "Wikipedia function words dominant (શામેલ, ઘણીવાર)"
        return "Mixed function word usage"

    def _classify_technical_terms(self, features) -> str:
        latin = features.get('latin_token_ratio', 0)
        if latin > 0.05:
            return "Heavy Latin/English term usage"
        elif latin > 0.02:
            return "Moderate English term integration"
        return "Minimal English terms"

    def _classify_rare_words(self, features) -> str:
        hapax = features.get('hapax_ratio', 0)
        if hapax > 0.75:
            return "High hapax ratio (concise vocabulary) → Visvakosh"
        elif hapax > 0.65:
            return "Moderate hapax ratio"
        return "Low hapax ratio (repeated terms) → Wikipedia"

    def _classify_passive_style(self, features) -> str:
        v = features.get('visvakosh_passive_per_1000', 0)
        w = features.get('wikipedia_passive_per_1000', 0)
        if v > w:
            return "Concise passive (ગણાય છે, કરાય છે) → Visvakosh"
        elif w > v:
            return "Extended passive (કરવામાં આવે છે) → Wikipedia"
        return "Balanced passive usage"

    def _classify_sentence_complexity(self, features) -> str:
        avg = features.get('avg_sentence_length', 15)
        std = features.get('std_sentence_length', 5)
        if avg < 16 and std < 8:
            return "Simple, consistent sentence structure → Visvakosh"
        elif avg > 20:
            return "Complex, long sentences → Wikipedia"
        return "Moderate complexity"

    def _classify_tense(self, text: str) -> str:
        past = len(re.findall(r'હત[ુંાી]|થય[ુંો]|કર્યું|ગયું', text))
        present = len(re.findall(r'છે|થાય|કરે|ગણાય', text))
        if past > present * 1.5:
            return "Past tense dominant"
        elif present > past * 1.5:
            return "Present tense dominant"
        return "Mixed tense usage"

    def _detect_sectioning(self, text: str) -> str:
        heading_pattern = re.compile(r'^==+\s*.+\s*==+$', re.MULTILINE)
        if heading_pattern.search(text):
            return "Wiki-style section headings present → Wikipedia"
        return "No section headings → Visvakosh style"

    def _detect_sectioning_count(self, text: str) -> int:
        if text is None:
            return 0
        return len(re.findall(r'^==+.+==+$', text, re.MULTILINE))

    def _detect_lists(self, text: str) -> str:
        if re.search(r'^\s*[\*\-\#]\s', text, re.MULTILINE):
            return "Bullet/numbered lists present → Wikipedia"
        return "No list formatting"

    def _detect_citations(self, text: str) -> str:
        if re.search(r'\[\d+\]', text):
            return "Citation markers present → Wikipedia"
        return "No citation markers"

    def _generate_summary(self, prediction: str, style_scores: Dict, satisfied: Dict) -> str:
        conf = satisfied['confidence']
        sat_count = satisfied['total_satisfied']
        total = satisfied['total_checked']

        lines = [
            f"📊 Prediction: **{prediction}** (confidence: {conf:.1%})",
            f"✅ Satisfied {sat_count}/{total} style properties for {prediction}",
            "",
            "Key indicators:",
            f"  • Visvakosh marker score: {style_scores['visvakosh_marker_score']:.2f}",
            f"  • Wikipedia marker score: {style_scores['wikipedia_marker_score']:.2f}",
            f"  • Definition-first score: {style_scores['definition_first_score']:.2f}",
            f"  • Style ratio (V/(V+W)): {style_scores['visvakosh_style_ratio']:.3f}",
        ]
        return "\n".join(lines)


# ============================================================================
# PUBLIC API
# ============================================================================

def load_and_analyze(text: str, model_path: str = 'visvakosh_classifier.pkl') -> Dict[str, Any]:
    """Main entry point: Load model and analyze text."""
    classifier = VisvakoshWikipediaClassifier.load(model_path)
    analyzer = DetailedAnalyzer(classifier)
    return analyzer.analyze(text)


def load_model(model_path: str = 'visvakosh_classifier.pkl'):
    """Load the classifier object for reuse."""
    return VisvakoshWikipediaClassifier.load(model_path)


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    test_text = """
    કોમ્પ્યૂટર : વિવિધ કાર્યક્રમમાં આપેલી સૂચના અનુસાર માહિતીસંગ્રહ 
    અને માહિતીપ્રક્રમણ માટેનું વીજાણુસાધન. તે સંજ્ઞાઓનું ઝડપથી અને 
    ચોકસાઈપૂર્વક રૂપાંતર કરી શકતું મશીન છે. આથી તેને ગણાય છે.
    """
    result = load_and_analyze(test_text, 'visvakosh_classifier.pkl')
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(result['summary'])
