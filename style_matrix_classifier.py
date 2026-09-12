# ============================================================================
# style_matrix_classifier.py
# RULE-BASED Visvakosh vs Wikipedia Classifier
# Uses style matrix thresholds from research (NO ML, NO CHEATING)
# ============================================================================

import re
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Any


# ============================================================================
# TOKENIZER
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


# ============================================================================
# STYLE MATRIX EXTRACTOR
# ============================================================================

class StyleMatrixExtractor:
    """Extracts raw style matrix features. Same as before."""

    def __init__(self):
        self.tokenizer = GujaratiTokenizer()

        self.visvakosh_markers = [
            'તથા', 'વળી', 'આથી', 'ગણાય', 'પ્રચલિત', 'આવાં', 'કેટલાંક',
            'અલબત્ત', 'તદુપરાંત', 'જુઓ', 'એટલે કે', 'કહેવાય છે',
            'તેમજ', 'ઉપરાંત', 'ક્રમશ:'
        ]

        self.wikipedia_markers = [
            'શામેલ', 'ઘણીવાર', 'કોઈપણ', 'વ્યાખ્યાયિત', 'ઉદાહરણ તરીકે',
            'મોડેલ', 'સોફ્ટવેર', 'ઓફ', 'મુખ્ય લેખ', 'આ પણ જુઓ', 'જો કે',
            'દ્વારા', 'સંદર્ભ', 'બાહ્ય કડીઓ'
        ]

        self.visvakosh_passive = [
            'ગણાય છે', 'કરાય છે', 'કહેવાય છે', 'ઓળખાય છે'
        ]

        self.wikipedia_passive = [
            'કરવામાં આવે છે', 'આપવામાં આવે છે', 'બનાવવામાં આવે છે',
            'માનવામાં આવે છે', 'કરવામાં આવ્યા હતા', 'કરવામાં આવ્યું હતું'
        ]

        self.definition_markers = [
            'એટલે', 'કહેવાય', 'ગણાય', 'રૂપે ઓળખાય', 'ઓળખાય છે', 'એટલે કે'
        ]

        self.traditional_translit = ['ૉ', 'ૅ', 'ઑ']
        self.modern_translit = ['ો', 'ે', 'ઓ']
        self.english_letters = re.compile(r'[a-zA-Z]')

    def extract(self, text: str) -> Dict[str, float]:
        if not text or len(text) < 20:
            return self._empty()

        words = self.tokenizer.tokenize_words(text)
        sentences = self.tokenizer.tokenize_sentences(text)

        if not words:
            return self._empty()

        wc = len(words)
        cc = len(text)
        sc = max(len(sentences), 1)

        # --- Length ---
        f = {}
        f['word_count'] = wc
        f['char_count'] = cc
        f['sentence_count'] = sc

        # --- Sentence ---
        slens = [len(self.tokenizer.tokenize_words(s)) for s in sentences]
        slens = [l for l in slens if l > 0]
        f['avg_sentence_length'] = float(np.mean(slens)) if slens else 0
        f['std_sentence_length'] = float(np.std(slens)) if len(slens) > 1 else 0

        # --- Vocabulary ---
        uw = set(words)
        f['type_token_ratio'] = len(uw) / wc
        wf = Counter(words)
        hapax = sum(1 for w, c in wf.items() if c == 1)
        f['hapax_ratio'] = hapax / len(uw) if uw else 0

        # --- Passive (two styles) ---
        v_pass = sum(text.count(m) for m in self.visvakosh_passive)
        w_pass = sum(text.count(m) for m in self.wikipedia_passive)
        f['visvakosh_passive_per_1000'] = (v_pass / wc) * 1000
        f['wikipedia_passive_per_1000'] = (w_pass / wc) * 1000
        f['total_passive_per_1000'] = ((v_pass + w_pass) / wc) * 1000

        # --- Transliteration ---
        trad = sum(text.count(m) for m in self.traditional_translit)
        modern = sum(text.count(m) for m in self.modern_translit)
        f['traditional_translit_count'] = trad
        f['modern_translit_count'] = modern
        f['traditional_translit_ratio'] = trad / cc
        f['modern_translit_ratio'] = modern / cc

        # --- Punctuation ---
        f['colon_count'] = text.count(':')
        f['colon_per_1000'] = (f['colon_count'] / wc) * 1000
        f['semicolon_per_1000'] = (text.count(';') / wc) * 1000
        f['parentheses_per_1000'] = ((text.count('(') + text.count(')')) / wc) * 1000

        # --- Markers ---
        v_mark = sum(text.count(m) for m in self.visvakosh_markers)
        w_mark = sum(text.count(m) for m in self.wikipedia_markers)
        f['visvakosh_marker_count'] = v_mark
        f['wikipedia_marker_count'] = w_mark
        f['visvakosh_markers_per_1000'] = (v_mark / wc) * 1000
        f['wikipedia_markers_per_1000'] = (w_mark / wc) * 1000

        # --- Definition style ---
        f['colon_in_first_200'] = 1 if ':' in text[:200] else 0
        f['colon_in_first_100'] = 1 if ':' in text[:100] else 0
        f['first_sentence_has_colon'] = 0
        f['first_sentence_has_def_marker'] = 0
        if sentences:
            fs = sentences[0]
            f['first_sentence_has_colon'] = 1 if ':' in fs else 0
            f['first_sentence_has_def_marker'] = 1 if any(
                m in fs for m in self.definition_markers) else 0

        # --- English usage ---
        eng_chars = len(self.english_letters.findall(text))
        f['english_char_ratio'] = eng_chars / cc
        gl = re.findall(r'\([A-Za-z][A-Za-z\s\.\-]+\)', text)
        f['english_gloss_count'] = len(gl)
        f['english_glosses_per_1000'] = (len(gl) / wc) * 1000

        # --- Structural ---
        # Wiki headings
        wiki_headings = re.findall(r'^=+\s*.+\s*=+$', text, re.MULTILINE)
        f['wiki_heading_count'] = len(wiki_headings)
        # Citation markers
        citations = re.findall(r'\[\d+\]', text)
        f['citation_count'] = len(citations)

        # --- Modern vs traditional style ratio ---
        total_t = trad + modern
        f['translit_style_ratio'] = trad / total_t if total_t > 0 else 0

        return f

    def _empty(self) -> Dict[str, float]:
        return {
            'word_count': 0, 'char_count': 0, 'sentence_count': 0,
            'avg_sentence_length': 0, 'std_sentence_length': 0,
            'type_token_ratio': 0, 'hapax_ratio': 0,
            'visvakosh_passive_per_1000': 0, 'wikipedia_passive_per_1000': 0,
            'total_passive_per_1000': 0,
            'traditional_translit_count': 0, 'modern_translit_count': 0,
            'traditional_translit_ratio': 0, 'modern_translit_ratio': 0,
            'colon_count': 0, 'colon_per_1000': 0,
            'semicolon_per_1000': 0, 'parentheses_per_1000': 0,
            'visvakosh_marker_count': 0, 'wikipedia_marker_count': 0,
            'visvakosh_markers_per_1000': 0, 'wikipedia_markers_per_1000': 0,
            'colon_in_first_200': 0, 'colon_in_first_100': 0,
            'first_sentence_has_colon': 0, 'first_sentence_has_def_marker': 0,
            'english_char_ratio': 0, 'english_gloss_count': 0,
            'english_glosses_per_1000': 0,
            'wiki_heading_count': 0, 'citation_count': 0,
            'translit_style_ratio': 0
        }


# ============================================================================
# RULE-BASED CLASSIFIER (Based on YOUR research findings)
# ============================================================================

class RuleBasedStyleClassifier:
    """
    Classifies text using ONLY style matrix thresholds derived from research.
    NO machine learning. NO synthetic data. Pure rules.

    Each rule is a "voting" rule. Text with more Visvakosh votes = Visvakosh.
    """

    def __init__(self):
        self.extractor = StyleMatrixExtractor()

        # =====================================================================
        # RULES (from your research findings)
        # Each rule returns (visvakosh_vote, wikipedia_vote, reason)
        # =====================================================================
        self.rules = [
            # RULE 1: Definition-first opening
            # Visvakosh: 70.5% definition-first | Wikipedia: 9.1%
            self._rule_definition_first,

            # RULE 2: Visvakosh marker words (તથા, વળી, આથી, ગણાય)
            # Visvakosh: 5.28/1000 vs Wikipedia: 0.01/1000
            self._rule_visvakosh_markers,

            # RULE 3: Wikipedia marker words (શામેલ, ઘણીવાર, મુખ્ય લેખ)
            self._rule_wikipedia_markers,

            # RULE 4: Traditional transliteration (ૉ, ૅ, ઑ)
            # Visvakosh uses these much more than Wikipedia
            self._rule_transliteration,

            # RULE 5: English glosses in parentheses
            # Visvakosh: 19.86/1000 vs Wikipedia: 4.87/1000
            self._rule_english_glosses,

            # RULE 6: Passive voice style
            # Visvakosh: concise (ગણાય છે) | Wikipedia: extended (કરવામાં આવે છે)
            self._rule_passive_style,

            # RULE 7: Colon usage
            # Visvakosh: 14.18/1000 vs Wikipedia: 5.25/1000
            self._rule_colon_usage,

            # RULE 8: Parentheses usage
            # Visvakosh: 27.48/1000 vs Wikipedia: 13.25/1000
            self._rule_parentheses,

            # RULE 9: Citation markers [1], [2] → Wikipedia
            self._rule_citations,

            # RULE 10: Wiki-style headings (== Section ==) → Wikipedia
            self._rule_headings,

            # RULE 11: Sentence length
            # Visvakosh: 14.95 avg | Wikipedia: 19.26 avg
            self._rule_sentence_length,

            # RULE 12: Type-Token Ratio (lexical diversity)
            # Visvakosh: higher TTR (0.45-0.65) | Wikipedia: lower (0.39-0.52)
            self._rule_ttr,

            # RULE 13: Hapax ratio
            # Visvakosh: 73.32% | Wikipedia: 66.02%
            self._rule_hapax,

            # RULE 14: Definition markers (એટલે, કહેવાય)
            self._rule_definition_markers,
        ]

    # =========================================================================
    # INDIVIDUAL RULES
    # =========================================================================

    def _rule_definition_first(self, f):
        """Definition-first opening pattern."""
        if f['colon_in_first_100'] == 1 or f['first_sentence_has_colon'] == 1:
            return (2, 0, "✅ Definition-first opening (term : definition) — Visvakosh signature")
        if f['colon_in_first_200'] == 1:
            return (1, 0, "✅ Colon appears early (100-200 chars) — mildly Visvakosh")
        return (0, 1, "❌ No definition-first opening — more Wikipedia-like")

    def _rule_visvakosh_markers(self, f):
        """Visvakosh characteristic function words."""
        val = f['visvakosh_markers_per_1000']
        if val > 8:
            return (2, 0, f"✅ Strong Visvakosh markers ({val:.1f}/1000): તથા, વળી, આથી, ગણાય")
        elif val > 3:
            return (1, 0, f"✅ Moderate Visvakosh markers ({val:.1f}/1000)")
        return (0, 1, f"❌ Few Visvakosh markers ({val:.1f}/1000)")

    def _rule_wikipedia_markers(self, f):
        """Wikipedia characteristic function words."""
        val = f['wikipedia_markers_per_1000']
        if val > 2:
            return (0, 2, f"✅ Wikipedia markers present ({val:.1f}/1000): શામેલ, ઘણીવાર")
        return (0, 0, f"ℹ Wikipedia markers absent ({val:.1f}/1000)")

    def _rule_transliteration(self, f):
        """Traditional (ૉ, ૅ, ઑ) vs modern (ો, ે, ઓ) transliteration."""
        trad = f['traditional_translit_count']
        modern = f['modern_translit_count']
        if trad > 3 and trad > modern * 0.15:
            return (2, 0, f"✅ Traditional transliteration dominant ({trad} traditional marks)")
        elif trad > 0:
            return (1, 0, f"✅ Some traditional transliteration ({trad} marks)")
        return (0, 1, f"❌ No traditional transliteration — Wikipedia uses modern style")

    def _rule_english_glosses(self, f):
        """English terms in parentheses (Gujarati → English gloss)."""
        val = f['english_glosses_per_1000']
        if val > 12:
            return (2, 0, f"✅ Heavy English glossing ({val:.1f}/1000) — Visvakosh terminology style")
        elif val > 5:
            return (1, 0, f"✅ Moderate English glossing ({val:.1f}/1000)")
        return (0, 1, f"❌ Low English glossing ({val:.1f}/1000) — Wikipedia style")

    def _rule_passive_style(self, f):
        """Concise vs extended passive voice."""
        v = f['visvakosh_passive_per_1000']
        w = f['wikipedia_passive_per_1000']
        if v > w and v > 1:
            return (2, 0, f"✅ Concise passive style (ગણાય છે, કરાય છે) — Visvakosh")
        elif w > v and w > 2:
            return (0, 2, f"✅ Extended passive style (કરવામાં આવે છે) — Wikipedia")
        return (0, 0, "ℹ No strong passive voice signal")

    def _rule_colon_usage(self, f):
        """Colon frequency."""
        val = f['colon_per_1000']
        if val > 12:
            return (2, 0, f"✅ High colon usage ({val:.1f}/1000) — Visvakosh label style")
        elif val > 6:
            return (1, 0, f"✅ Moderate colon usage ({val:.1f}/1000)")
        return (0, 1, f"❌ Low colon usage ({val:.1f}/1000)")

    def _rule_parentheses(self, f):
        """Parentheses frequency."""
        val = f['parentheses_per_1000']
        if val > 40:
            return (2, 0, f"✅ Heavy parentheses ({val:.1f}/1000) — Visvakosh glossing")
        elif val > 20:
            return (1, 0, f"✅ Moderate parentheses ({val:.1f}/1000)")
        return (0, 1, f"❌ Low parentheses ({val:.1f}/1000)")

    def _rule_citations(self, f):
        """Wikipedia citation markers [1], [2]."""
        if f['citation_count'] > 0:
            return (0, 3, f"✅ STRONG: {f['citation_count']} citation markers [n] — definite Wikipedia")
        return (0, 0, "ℹ No citation markers")

    def _rule_headings(self, f):
        """Wiki-style section headings."""
        if f['wiki_heading_count'] > 0:
            return (0, 3, f"✅ STRONG: {f['wiki_heading_count']} wiki-style headings (== ... ==) — Wikipedia")
        return (0, 0, "ℹ No wiki-style headings")

    def _rule_sentence_length(self, f):
        """Average sentence length."""
        val = f['avg_sentence_length']
        if val < 15:
            return (2, 0, f"✅ Short sentences (avg {val:.1f} words) — Visvakosh style")
        elif val < 19:
            return (1, 0, f"✅ Medium sentences (avg {val:.1f} words)")
        return (0, 2, f"✅ Long sentences (avg {val:.1f} words) — Wikipedia style")

    def _rule_ttr(self, f):
        """Type-Token Ratio (lexical diversity)."""
        val = f['type_token_ratio']
        if val > 0.55:
            return (2, 0, f"✅ High lexical diversity (TTR {val:.3f}) — Visvakosh")
        elif val > 0.48:
            return (1, 0, f"✅ Moderate lexical diversity (TTR {val:.3f})")
        return (0, 1, f"❌ Low lexical diversity (TTR {val:.3f})")

    def _rule_hapax(self, f):
        """Hapax ratio (percentage of words appearing once)."""
        val = f['hapax_ratio']
        if val > 0.72:
            return (2, 0, f"✅ High hapax ratio ({val:.3f}) — Visvakosh concise vocabulary")
        elif val > 0.65:
            return (1, 0, f"✅ Moderate hapax ratio ({val:.3f})")
        return (0, 1, f"❌ Low hapax ratio ({val:.3f})")

    def _rule_definition_markers(self, f):
        """Definition markers like એટલે, કહેવાય."""
        # Count from raw text (recovered from vocab features)
        if f['first_sentence_has_def_marker'] == 1:
            return (1, 0, "✅ Definition marker in first sentence")
        return (0, 0, "ℹ No definition marker in first sentence")

    # =========================================================================
    # MAIN CLASSIFY METHOD
    # =========================================================================

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify text using style matrix rules.
        Returns complete analysis with voting breakdown.
        """
        features = self.extractor.extract(text)

        # Collect votes
        v_total = 0
        w_total = 0
        rule_results = []

        for rule in self.rules:
            v, w, reason = rule(features)
            v_total += v
            w_total += w
            rule_results.append({
                'rule': rule.__name__.replace('_rule_', ''),
                'visvakosh_votes': v,
                'wikipedia_votes': w,
                'reason': reason
            })

        # Decide prediction
        total = v_total + w_total
        if total == 0:
            prediction = "Unknown"
            confidence = 0.5
        else:
            v_ratio = v_total / total
            prediction = "Visvakosh" if v_ratio > 0.5 else "Wikipedia"
            confidence = max(v_ratio, 1 - v_ratio)

        # Separate satisfied/unsatisfied Visvakosh properties
        visvakosh_satisfied = [r for r in rule_results if r['visvakosh_votes'] > 0]
        wikipedia_satisfied = [r for r in rule_results if r['wikipedia_votes'] > 0]

        # Quantitative summary
        quant = self._quant_summary(features)
        qual = self._qual_summary(features)

        return {
            'prediction': prediction,
            'confidence': round(confidence, 4),
            'votes': {
                'visvakosh_total': v_total,
                'wikipedia_total': w_total,
                'total_rules': len(self.rules),
                'visvakosh_ratio': round(v_total / total, 4) if total > 0 else 0.5
            },
            'rule_results': rule_results,
            'visvakosh_satisfied': visvakosh_satisfied,
            'wikipedia_satisfied': wikipedia_satisfied,
            'raw_features': features,
            'quantitative_analysis': quant,
            'qualitative_analysis': qual,
            'summary': self._generate_summary(
                prediction, confidence, v_total, w_total, features
            )
        }

    def _quant_summary(self, f):
        return {
            'length_metrics': {
                'word_count': int(f['word_count']),
                'character_count': int(f['char_count']),
                'sentence_count': int(f['sentence_count']),
            },
            'sentence_metrics': {
                'avg_sentence_length': round(f['avg_sentence_length'], 2),
                'std_sentence_length': round(f['std_sentence_length'], 2),
                'interpretation': (
                    "Short → Visvakosh" if f['avg_sentence_length'] < 15
                    else "Long → Wikipedia" if f['avg_sentence_length'] > 19
                    else "Medium → Borderline"
                )
            },
            'vocabulary_metrics': {
                'type_token_ratio': round(f['type_token_ratio'], 4),
                'hapax_ratio': round(f['hapax_ratio'], 4),
                'interpretation': (
                    "High TTR → Visvakosh" if f['type_token_ratio'] > 0.55
                    else "Low TTR → Wikipedia"
                )
            },
            'marker_metrics': {
                'visvakosh_markers_per_1000': round(f['visvakosh_markers_per_1000'], 2),
                'wikipedia_markers_per_1000': round(f['wikipedia_markers_per_1000'], 2),
                'interpretation': (
                    "Strong Visvakosh markers" if f['visvakosh_markers_per_1000'] > 8
                    else "Strong Wikipedia markers" if f['wikipedia_markers_per_1000'] > 2
                    else "Weak markers"
                )
            },
            'passive_metrics': {
                'visvakosh_passive_per_1000': round(f['visvakosh_passive_per_1000'], 2),
                'wikipedia_passive_per_1000': round(f['wikipedia_passive_per_1000'], 2),
                'interpretation': (
                    "Concise passive → Visvakosh"
                    if f['visvakosh_passive_per_1000'] > f['wikipedia_passive_per_1000']
                    else "Extended passive → Wikipedia"
                )
            },
            'transliteration_metrics': {
                'traditional_count': int(f['traditional_translit_count']),
                'modern_count': int(f['modern_translit_count']),
                'interpretation': (
                    "Traditional → Visvakosh" if f['traditional_translit_count'] > 0
                    else "Modern only → Wikipedia"
                )
            },
            'punctuation_metrics': {
                'colon_per_1000': round(f['colon_per_1000'], 2),
                'semicolon_per_1000': round(f['semicolon_per_1000'], 2),
                'parentheses_per_1000': round(f['parentheses_per_1000'], 2),
            },
            'gloss_metrics': {
                'english_gloss_count': int(f['english_gloss_count']),
                'english_glosses_per_1000': round(f['english_glosses_per_1000'], 2),
            },
            'structural_metrics': {
                'citation_count': int(f['citation_count']),
                'wiki_heading_count': int(f['wiki_heading_count']),
            }
        }

    def _qual_summary(self, f):
        return {
            'tone': {
                'definition_first': bool(f['colon_in_first_200']),
                'definition_style': (
                    "Term : Definition pattern — Visvakosh signature"
                    if f['first_sentence_has_colon'] else "No definition-first opening"
                ),
            },
            'structure': {
                'has_citations': f['citation_count'] > 0,
                'has_wiki_headings': f['wiki_heading_count'] > 0,
                'citation_note': (
                    "Wikipedia-style citations present" if f['citation_count'] > 0
                    else "No Wikipedia citations"
                ),
            },
            'vocabulary_style': {
                'marker_style': (
                    "Visvakosh function words dominant"
                    if f['visvakosh_markers_per_1000'] > f['wikipedia_markers_per_1000']
                    else "Wikipedia function words dominant"
                ),
                'transliteration_style': (
                    "Traditional transliteration (ૉ, ૅ) — Visvakosh"
                    if f['traditional_translit_count'] > 0
                    else "Modern transliteration — Wikipedia style"
                ),
            },
            'glossing_style': {
                'gloss_density': (
                    "Heavy glossing — Visvakosh terminology policy"
                    if f['english_glosses_per_1000'] > 12
                    else "Light glossing — Wikipedia"
                ),
            }
        }

    def _generate_summary(self, prediction, confidence, v_votes, w_votes, f):
        lines = [
            f"**Prediction:** {prediction}  ",
            f"**Confidence:** {confidence:.1%}  ",
            f"**Vote breakdown:** Visvakosh = {v_votes}, Wikipedia = {w_votes}  ",
            "",
            "**Key style matrix results:**",
        ]
        lines.append(f"  • Definition-first opening: {'Yes' if f['colon_in_first_200'] else 'No'}")
        lines.append(f"  • Visvakosh markers/1000: {f['visvakosh_markers_per_1000']:.2f}")
        lines.append(f"  • Wikipedia markers/1000: {f['wikipedia_markers_per_1000']:.2f}")
        lines.append(f"  • Traditional translit count: {f['traditional_translit_count']}")
        lines.append(f"  • Citation markers [n]: {f['citation_count']}")
        lines.append(f"  • Wiki headings (== ==): {f['wiki_heading_count']}")
        lines.append(f"  • Average sentence length: {f['avg_sentence_length']:.1f}")
        return "\n".join(lines)


# ============================================================================
# PUBLIC API
# ============================================================================

_classifier_instance = None

def get_classifier():
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = RuleBasedStyleClassifier()
    return _classifier_instance


def analyze_text(text: str) -> Dict[str, Any]:
    """Analyze Gujarati text using rule-based style matrix."""
    return get_classifier().classify(text)


# ============================================================================
# CLI TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RULE-BASED STYLE MATRIX CLASSIFIER")
    print("=" * 70)

    # Visvakosh-style text
    visvakosh_sample = """કોમ્પ્યૂટર : વિવિધ કાર્યક્રમમાં આપેલી સૂચના અનુસાર 
    માહિતીસંગ્રહ અને માહિતીપ્રક્રમણ માટેનું વીજાણુસાધન. તે સંજ્ઞાઓનું ઝડપથી 
    અને ચોકસાઈપૂર્વક રૂપાંતર કરી શકતું મશીન છે. આથી તેને ગણાય છે. 
    કોમ્પ્યૂટરમાં દ્વિઅંકી સંજ્ઞા (binary code) 0 અને 1 વપરાય છે. 
    તેની શક્તિ 10–5 eVથી 0.01 eV જેટલી હોય છે. વળી, ઍનાલિટિક એન્જિન 
    (analytical engine) નામે ગણનયંત્ર ચાર્લ્સ બેબેજે બનાવ્યું. તથા 
    તે 1837માં બનાવવામાં આવ્યું હતું."""

    # Wikipedia-style text
    wikipedia_sample = """કમ્પ્યુટર એ એક ઇલેક્ટ્રોનિક ઉપકરણ છે જે માહિતીને 
    સંગ્રહિત કરી શકે છે અને પ્રક્રિયા કરી શકે છે. આ ઉપકરણનો ઉપયોગ વિવિધ 
    ક્ષેત્રોમાં કરવામાં આવે છે. ઉદાહરણ તરીકે, શિક્ષણ, આરોગ્ય સંભાળ, વ્યાપાર 
    વગેરેમાં કમ્પ્યુટરનો ઉપયોગ કરવામાં આવે છે. કમ્પ્યુટરની શોધ ઘણા વૈજ્ઞાનિકો 
    દ્વારા કરવામાં આવી હતી. જો કે, ચાર્લ્સ બેબેજને કમ્પ્યુટરના પિતા ગણવામાં 
    આવે છે. મુખ્ય લેખ: કમ્પ્યુટરનો ઇતિહાસ [1][2]"""

    for label, text in [("Visvakosh sample", visvakosh_sample),
                        ("Wikipedia sample", wikipedia_sample)]:
        print(f"\n{'─' * 70}")
        print(f"Testing: {label}")
        print(f"{'─' * 70}")
        result = analyze_text(text)
        print(f"\n{result['summary']}")

        print(f"\nVotes: V={result['votes']['visvakosh_total']}, "
              f"W={result['votes']['wikipedia_total']}, "
              f"Ratio={result['votes']['visvakosh_ratio']}")

        print(f"\nRule-by-rule breakdown:")
        for r in result['rule_results']:
            if r['visvakosh_votes'] > 0 or r['wikipedia_votes'] > 0:
                print(f"  [{r['visvakosh_votes']}v/{r['wikipedia_votes']}w] {r['reason']}")
