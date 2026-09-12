# ============================================================================
# streamlit_app.py  — Rule-Based Version (No ML, No .pkl needed)
# ============================================================================

import streamlit as st
import json
import pandas as pd

from style_matrix_classifier import analyze_text


st.set_page_config(
    page_title="Visvakosh vs Wikipedia Classifier",
    page_icon="📚",
    layout="wide"
)

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
    .stTextArea textarea { font-family: 'Noto Sans Gujarati', 'Shruti', sans-serif;
                           font-size: 15px; }
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="main-title">📚 Gujarati Source Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Rule-based Visvakosh vs Wikipedia classification '
    'using style matrix (qualitative + quantitative)</div>',
    unsafe_allow_html=True
)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("⚙️ About")
    st.info(
        "**Rule-Based Classifier**\n\n"
        "No ML model. No training data needed.\n\n"
        "Uses 14 style matrix rules based on published research findings:\n"
        "- Definition-first opening\n"
        "- Function word markers (તથા, વળી vs શામેલ, ઘણીવાર)\n"
        "- Transliteration style (ૉ, ૅ vs ો, ે)\n"
        "- Passive voice style\n"
        "- Citation markers, wiki headings\n"
        "- Punctuation patterns\n"
        "- Lexical diversity metrics"
    )

    st.markdown("---")
    st.header("📝 Sample Texts")

    sample_v = """કોમ્પ્યૂટર : વિવિધ કાર્યક્રમમાં આપેલી સૂચના અનુસાર માહિતીસંગ્રહ અને માહિતીપ્રક્રમણ માટેનું વીજાણુસાધન. તે સંજ્ઞાઓનું ઝડપથી અને ચોકસાઈપૂર્વક રૂપાંતર કરી શકતું મશીન છે. આથી તેને ગણાય છે. કોમ્પ્યૂટરમાં દ્વિઅંકી સંજ્ઞા (binary code) 0 અને 1 વપરાય છે. વળી, ઍનાલિટિક એન્જિન (analytical engine) નામે ગણનયંત્ર ચાર્લ્સ બેબેજે બનાવ્યું હતું. તથા તે 1837માં બનાવવામાં આવ્યું હતું."""

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

    # ---- PREDICTION BOX ----
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

    # Votes
    v = result['votes']
    c1, c2, c3 = st.columns(3)
    c1.metric("📖 Visvakosh Votes", v['visvakosh_total'])
    c2.metric("🌐 Wikipedia Votes", v['wikipedia_total'])
    c3.metric("Visvakosh Ratio", f"{v['visvakosh_ratio']:.1%}")

    st.progress(v['visvakosh_ratio'])

    # ---- RULE-BY-RULE BREAKDOWN ----
    st.markdown("---")
    st.header("📋 Rule-by-Rule Breakdown")
    st.caption("Each rule contributes votes to Visvakosh or Wikipedia based on research thresholds")

    for r in result['rule_results']:
        v_votes = r['visvakosh_votes']
        w_votes = r['wikipedia_votes']
        if v_votes == 0 and w_votes == 0:
            continue
        if v_votes > 0:
            st.markdown(f"**+{v_votes} Visvakosh**  {r['reason']}")
        if w_votes > 0:
            st.markdown(f"**+{w_votes} Wikipedia**  {r['reason']}")

    # ---- SATISFIED PROPERTIES ----
    st.markdown("---")
    st.header("✅ Satisfied Style Properties")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📖 Visvakosh Indicators Met")
        if result['visvakosh_satisfied']:
            for r in result['visvakosh_satisfied']:
                st.markdown(
                    f'<span class="satisfied-tag">{r["rule"]}</span>',
                    unsafe_allow_html=True
                )
        else:
            st.info("None met")

    with c2:
        st.subheader("🌐 Wikipedia Indicators Met")
        if result['wikipedia_satisfied']:
            for r in result['wikipedia_satisfied']:
                st.markdown(
                    f'<span class="wikipedia-tag">{r["rule"]}</span>',
                    unsafe_allow_html=True
                )
        else:
            st.info("None met")

    # ---- QUANTITATIVE ----
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
        c1.metric("Visvakosh Markers/1000", f"{mm['visvakosh_markers_per_1000']:.2f}")
        c2.metric("Wikipedia Markers/1000", f"{mm['wikipedia_markers_per_1000']:.2f}")
        st.info(f"💡 {mm['interpretation']}")

    with st.expander("🔊 Passive Voice Metrics", expanded=True):
        pm = quant['passive_metrics']
        c1, c2 = st.columns(2)
        c1.metric("Visvakosh Passive/1000", f"{pm['visvakosh_passive_per_1000']:.2f}")
        c2.metric("Wikipedia Passive/1000", f"{pm['wikipedia_passive_per_1000']:.2f}")
        st.info(f"💡 {pm['interpretation']}")

    with st.expander("🔤 Transliteration Metrics", expanded=True):
        tm = quant['transliteration_metrics']
        c1, c2 = st.columns(2)
        c1.metric("Traditional (ૉ, ૅ, ઑ)", tm['traditional_count'])
        c2.metric("Modern (ો, ે, ઓ)", tm['modern_count'])
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
        c1.metric("Citation Markers [n]", stm['citation_count'])
        c2.metric("Wiki Headings (== ==)", stm['wiki_heading_count'])

    # ---- QUALITATIVE ----
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

    # ---- RAW FEATURES ----
    with st.expander("🔬 Raw Feature Values"):
        df = pd.DataFrame([
            {"Feature": k, "Value": v}
            for k, v in result['raw_features'].items()
        ])
        st.dataframe(df, use_container_width=True, height=400)

    # ---- DOWNLOAD ----
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
