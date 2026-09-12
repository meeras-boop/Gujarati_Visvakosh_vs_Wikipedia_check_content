# ============================================================================
# streamlit_app.py
# Streamlit UI for Visvakosh vs Wikipedia source prediction
# ============================================================================

import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import json

# Import the predictor module
from predictor import (
    load_model,
    DetailedAnalyzer,
    VisvakoshWikipediaClassifier
)


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Visvakosh vs Wikipedia Classifier",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .visvakosh-pred {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #28a745;
    }
    .wikipedia-pred {
        background-color: #cce5ff;
        color: #004085;
        border: 2px solid #0066cc;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f4e79;
        margin: 0.5rem 0;
    }
    .satisfied-tag {
        display: inline-block;
        background-color: #28a745;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        margin: 3px;
        font-size: 0.85rem;
    }
    .unsatisfied-tag {
        display: inline-block;
        background-color: #dc3545;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        margin: 3px;
        font-size: 0.85rem;
    }
    .neutral-tag {
        display: inline-block;
        background-color: #ffc107;
        color: #333;
        padding: 3px 10px;
        border-radius: 12px;
        margin: 3px;
        font-size: 0.85rem;
    }
    .stTextArea textarea {
        font-family: 'Noto Sans Gujarati', 'Shruti', sans-serif;
        font-size: 15px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")

    # Model path input
    model_path = st.text_input(
        "Model (.pkl) path",
        value="visvakosh_classifier.pkl",
        help="Path to the trained classifier .pkl file"
    )

    # Model status
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        st.success(f"✅ Model found ({size_mb:.2f} MB)")
    else:
        st.error("❌ Model not found")
        st.info("Upload your `visvakosh_classifier.pkl` to the app folder.")

    st.markdown("---")

    # Sample text buttons
    st.header("📝 Sample Texts")

    sample_visvakosh = """કોમ્પ્યૂટર : વિવિધ કાર્યક્રમમાં આપેલી સૂચના અનુસાર માહિતીસંગ્રહ અને માહિતીપ્રક્રમણ માટેનું વીજાણુસાધન. તે સંજ્ઞાઓનું ઝડપથી અને ચોકસાઈપૂર્વક રૂપાંતર કરી શકતું મશીન છે. આથી તેને ગણાય છે. આ ઉપરાંત વળી, તે ઘણું ઉપયોગી છે. તથા તેનો ઉપયોગ વિવિધ ક્ષેત્રોમાં થાય છે."""

    sample_wikipedia = """કમ્પ્યુટર એ એક ઇલેક્ટ્રોનિક ઉપકરણ છે જે માહિતીને સંગ્રહિત કરી શકે છે અને પ્રક્રિયા કરી શકે છે. આ ઉપકરણનો ઉપયોગ વિવિધ ક્ષેત્રોમાં કરવામાં આવે છે. ઉદાહરણ તરીકે, શિક્ષણ, આરોગ્ય સંભાળ, વ્યાપાર વગેરેમાં. કમ્પ્યુટરની શોધ ઘણા વૈજ્ઞાનિકો દ્વારા કરવામાં આવી હતી. જો કે, ચાર્લ્સ બેબેજને કમ્પ્યુટરના પિતા ગણવામાં આવે છે."""

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📖 Visvakosh Sample", use_container_width=True):
            st.session_state['sample_text'] = sample_visvakosh
    with col2:
        if st.button("🌐 Wikipedia Sample", use_container_width=True):
            st.session_state['sample_text'] = sample_wikipedia

    if st.button("🗑️ Clear Text", use_container_width=True):
        st.session_state['sample_text'] = ""

    st.markdown("---")
    st.caption("Model: Random Forest + TF-IDF + Style Matrix (95 features)")


# ============================================================================
# MAIN PAGE
# ============================================================================

st.markdown('<div class="main-title">📚 Gujarati Source Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict: Gujarati Visvakosh vs Wikipedia based on Style Matrix</div>', unsafe_allow_html=True)


# ============================================================================
# TEXT INPUT
# ============================================================================

st.header("📝 Enter Gujarati Text to Analyze")

text_input = st.text_area(
    "Paste your full Gujarati paragraph here:",
    value=st.session_state.get('sample_text', ''),
    height=300,
    placeholder="અહીં તમારું ગુજરાતી લખાણ પેસ્ટ કરો...",
    key="main_text_input"
)

# Character/word count
if text_input:
    word_count = len(text_input.split())
    char_count = len(text_input)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Characters", f"{char_count:,}")
    with col2:
        st.metric("Words (approx)", f"{word_count:,}")
    with col3:
        st.metric("Ready", "✅" if len(text_input) > 50 else "⚠️ Too short")


# ============================================================================
# ANALYZE BUTTON
# ============================================================================

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_btn = st.button(
        "🔍 ANALYZE TEXT",
        type="primary",
        use_container_width=True,
        disabled=(not text_input or len(text_input) < 30)
    )


# ============================================================================
# ANALYSIS & RESULTS
# ============================================================================

if analyze_btn:
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found. Please check the path in the sidebar.")
    else:
        with st.spinner("🔄 Loading model and analyzing text..."):
            try:
                # Load model
                classifier = load_model(model_path)
                analyzer = DetailedAnalyzer(classifier)

                # Analyze
                result = analyzer.analyze(text_input)

                st.success("✅ Analysis complete!")
                st.markdown("---")

                # ---------------------------------------------------------------
                # MAIN PREDICTION BOX
                # ---------------------------------------------------------------
                st.header("🎯 Prediction Result")

                pred = result['prediction']
                conf = result['confidence']
                proba = result['probabilities']

                css_class = "visvakosh-pred" if pred == "Visvakosh" else "wikipedia-pred"
                emoji = "📖" if pred == "Visvakosh" else "🌐"

                st.markdown(
                    f'<div class="prediction-box {css_class}">'
                    f'{emoji} This text is likely: <strong>{pred}</strong><br>'
                    f'<span style="font-size: 1rem;">Confidence: {conf:.2%}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # Probability bars
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "📖 Visvakosh Probability",
                        f"{proba['Visvakosh']:.2%}"
                    )
                    st.progress(proba['Visvakosh'])
                with col2:
                    st.metric(
                        "🌐 Wikipedia Probability",
                        f"{proba['Wikipedia']:.2%}"
                    )
                    st.progress(proba['Wikipedia'])

                # Summary
                with st.expander("📋 Summary", expanded=True):
                    st.markdown(result['summary'])

                st.markdown("---")

                # ---------------------------------------------------------------
                # SATISFIED PROPERTIES
                # ---------------------------------------------------------------
                st.header("✅ Style Matrix Properties Satisfied")

                sat = result['satisfied_properties']

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Predicted Source", sat['predicted_source'])
                with col2:
                    st.metric("Total Checked", sat['total_checked'])
                with col3:
                    st.metric("Satisfied", f"{sat['total_satisfied']} / {sat['total_checked']}")

                st.markdown(f"**Satisfaction Ratio:** {sat['satisfaction_ratio']:.2%}")

                # Satisfied tags
                st.markdown("**✅ Satisfied Properties:**")
                satisfied_html = "".join([
                    f'<span class="satisfied-tag">{k.replace("_", " ").title()}</span>'
                    for k in sat['satisfied_properties'].keys()
                ])
                st.markdown(satisfied_html, unsafe_allow_html=True)

                # Unsatisfied tags
                if sat['unsatisfied_properties']:
                    st.markdown("**❌ Unsatisfied Properties:**")
                    unsatisfied_html = "".join([
                        f'<span class="unsatisfied-tag">{k.replace("_", " ").title()}</span>'
                        for k in sat['unsatisfied_properties'].keys()
                    ])
                    st.markdown(unsatisfied_html, unsafe_allow_html=True)

                st.markdown("---")

                # ---------------------------------------------------------------
                # STYLE SCORES
                # ---------------------------------------------------------------
                st.header("🎨 Style Indicator Scores")

                scores = result['style_scores']

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📖 Visvakosh Indicators")
                    st.metric("Visvakosh Marker Score", f"{scores['visvakosh_marker_score']:.2f}")
                    st.metric("Definition-First Score", f"{scores['definition_first_score']:.2f}")
                    st.metric("Traditional Transliteration", f"{scores['traditional_translit_score']:.4f}")
                    st.metric("Concise Passive Style", f"{scores['concise_passive_score']:.2f}")

                with col2:
                    st.subheader("🌐 Wikipedia Indicators")
                    st.metric("Wikipedia Marker Score", f"{scores['wikipedia_marker_score']:.2f}")
                    st.metric("Modern Transliteration", f"{scores['modern_translit_score']:.2f}")
                    st.metric("Extended Passive Style", f"{scores['extended_passive_score']:.2f}")

                # Overall ratio gauge
                st.markdown("#### 📊 Overall Style Ratio (Visvakosh vs Wikipedia)")
                ratio = scores['visvakosh_style_ratio']
                st.progress(ratio)
                st.caption(
                    f"Visvakosh-style ratio: {ratio:.2%} "
                    f"(0% = pure Wikipedia, 100% = pure Visvakosh)"
                )

                st.markdown("---")

                # ---------------------------------------------------------------
                # QUANTITATIVE ANALYSIS
                # ---------------------------------------------------------------
                st.header("📈 Quantitative Analysis")

                quant = result['quantitative_analysis']

                # Length metrics
                with st.expander("📏 Length Metrics", expanded=True):
                    lm = quant['length_metrics']
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Words", f"{lm['word_count']:,}")
                    c2.metric("Characters", f"{lm['character_count']:,}")
                    c3.metric("Sentences", lm['sentence_count'])
                    c4.metric("Paragraphs", lm['paragraph_count'])
                    c5.metric("Avg Word Len", f"{lm['avg_word_length']:.2f}")

                # Sentence metrics
                with st.expander("📝 Sentence Metrics", expanded=True):
                    sm = quant['sentence_metrics']
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Avg Sentence Length", f"{sm['avg_sentence_length']:.2f}")
                    c2.metric("Std Sentence Length", f"{sm['std_sentence_length']:.2f}")
                    c3.metric("Max Length", sm['max_sentence_length'])
                    c4.metric("Min Length", sm['min_sentence_length'])
                    st.info(f"💡 {sm['interpretation']}")

                # Vocabulary metrics
                with st.expander("📚 Vocabulary Metrics", expanded=True):
                    vm = quant['vocabulary_metrics']
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Unique Words", f"{vm['unique_words']:,}")
                    c2.metric("Type-Token Ratio", f"{vm['type_token_ratio']:.4f}")
                    c3.metric("Hapax Count", f"{vm['hapax_count']:,}")
                    c4.metric("Hapax Ratio", f"{vm['hapax_ratio']:.4f}")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Yule's K", f"{vm['yule_k']:.2f}")
                    c2.metric("MATTR-50", f"{vm['mattr_50']:.4f}")
                    c3.metric("MATTR-100", f"{vm['mattr_100']:.4f}")
                    st.info(f"💡 {vm['interpretation']}")

                # Voice metrics
                with st.expander("🔊 Voice Metrics", expanded=True):
                    vc = quant['voice_metrics']
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Passive", vc['total_passive_count'])
                    c2.metric("Passive/1000", f"{vc['total_passive_per_1000']:.2f}")
                    c3.metric("Passive Sentence Ratio", f"{vc['passive_sentence_ratio']:.4f}")
                    c1, c2 = st.columns(2)
                    c1.metric("Visvakosh-style Passive/1000", f"{vc['visvakosh_passive_per_1000']:.2f}")
                    c2.metric("Wikipedia-style Passive/1000", f"{vc['wikipedia_passive_per_1000']:.2f}")
                    st.info(f"💡 {vc['interpretation']}")

                # Transliteration metrics
                with st.expander("🔤 Transliteration Metrics", expanded=True):
                    tm = quant['transliteration_metrics']
                    c1, c2 = st.columns(2)
                    c1.metric("Traditional Count (ૉ, ૅ, ઑ)", tm['traditional_translit_count'])
                    c2.metric("Modern Count (ો, ે, ઓ)", tm['modern_translit_count'])
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Traditional Ratio", f"{tm['traditional_translit_ratio']:.6f}")
                    c2.metric("Modern Ratio", f"{tm['modern_translit_ratio']:.6f}")
                    c3.metric("Style Ratio (Trad/(Trad+Mod))", f"{tm['translit_style_ratio']:.4f}")
                    st.info(f"💡 {tm['interpretation']}")

                # Punctuation metrics
                with st.expander("❕ Punctuation Metrics", expanded=True):
                    pm = quant['punctuation_metrics']
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Colons/1000", f"{pm['colon_per_1000']:.2f}")
                    c2.metric("Semicolons/1000", f"{pm['semicolon_per_1000']:.2f}")
                    c3.metric("Parentheses/1000", f"{pm['parentheses_per_1000']:.2f}")
                    c4.metric("Commas/1000", f"{pm['comma_per_1000']:.2f}")
                    st.info(f"💡 {pm['interpretation']}")

                # English usage metrics
                with st.expander("🔠 English Usage Metrics", expanded=True):
                    em = quant['english_usage_metrics']
                    c1, c2, c3 = st.columns(3)
                    c1.metric("English Char Ratio", f"{em['english_char_ratio']:.4f}")
                    c2.metric("English Glosses", em['english_gloss_count'])
                    c3.metric("Glosses/1000", f"{em['english_glosses_per_1000']:.2f}")
                    st.metric("Latin Token Ratio", f"{em['latin_token_ratio']:.4f}")
                    st.info(f"💡 {em['interpretation']}")

                st.markdown("---")

                # ---------------------------------------------------------------
                # QUALITATIVE ANALYSIS
                # ---------------------------------------------------------------
                st.header("🎭 Qualitative Analysis")

                qual = result['qualitative_analysis']

                # Tone
                with st.expander("🎵 Tone & Style", expanded=True):
                    tone = qual['tone']
                    st.markdown(f"**Definition-First:** {'Yes ✅' if tone['definition_first'] else 'No ❌'}")
                    st.markdown(f"**Definition Style:** {tone['definition_style']}")
                    st.markdown(f"**Encyclopedic Style:** {tone['encyclopedic_style']}")
                    st.markdown(f"**Narrative Style:** {tone['narrative_vs_expository']}")

                # Register
                with st.expander("📋 Register", expanded=True):
                    reg = qual['register']
                    st.markdown(f"**Formality:** {reg['formality']}")
                    st.markdown(f"**Technical Density:** {reg['technical_density']}")
                    st.markdown(f"**Terminology Glossing:** {reg['terminology_glossing']}")

                # Structural
                with st.expander("🏗️ Structural Style", expanded=True):
                    struct = qual['structural_style']
                    st.markdown(f"**Paragraph Density:** {struct['paragraph_density']}")
                    st.markdown(f"**Sectioning:** {struct['sectioning']}")
                    st.markdown(f"**List Usage:** {struct['list_usage']}")
                    st.markdown(f"**Citations:** {struct['citation_presence']}")

                # Vocabulary
                with st.expander("📖 Vocabulary Style", expanded=True):
                    vocab = qual['vocabulary_style']
                    st.markdown(f"**Function Words:** {vocab['function_words']}")
                    st.markdown(f"**Technical Terms:** {vocab['technical_terms']}")
                    st.markdown(f"**Rare Word Usage:** {vocab['rare_word_usage']}")

                # Grammatical
                with st.expander("✍️ Grammatical Style", expanded=True):
                    gram = qual['grammatical_style']
                    st.markdown(f"**Passive Voice Style:** {gram['passive_voice_style']}")
                    st.markdown(f"**Sentence Complexity:** {gram['sentence_complexity']}")
                    st.markdown(f"**Tense Usage:** {gram['tense_usage']}")

                st.markdown("---")

                # ---------------------------------------------------------------
                # RAW FEATURES TABLE
                # ---------------------------------------------------------------
                with st.expander("🔬 Raw Feature Values (All 95 Features)"):
                    features_df = pd.DataFrame([
                        {"Feature": k, "Value": v}
                        for k, v in result['raw_features'].items()
                    ])
                    st.dataframe(features_df, use_container_width=True, height=400)

                # ---------------------------------------------------------------
                # DOWNLOAD REPORT
                # ---------------------------------------------------------------
                st.markdown("---")
                st.header("📥 Download Analysis Report")

                report = {
                    "prediction": result['prediction'],
                    "confidence": result['confidence'],
                    "probabilities": result['probabilities'],
                    "style_scores": result['style_scores'],
                    "satisfied_properties": result['satisfied_properties'],
                    "quantitative_analysis": result['quantitative_analysis'],
                    "qualitative_analysis": result['qualitative_analysis'],
                    "raw_features": result['raw_features'],
                }

                report_json = json.dumps(report, indent=2, ensure_ascii=False, default=str)

                st.download_button(
                    label="⬇️ Download JSON Report",
                    data=report_json,
                    file_name=f"analysis_{result['prediction'].lower()}.json",
                    mime="application/json",
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                with st.expander("🔍 Traceback"):
                    import traceback
                    st.code(traceback.format_exc())


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption(
    "🔬 **Model Details:** Random Forest | 95 Style Matrix Features + TF-IDF (word + char) "
    "| Trained on Visvakosh corpus with synthetic Wikipedia augmentation"
)
