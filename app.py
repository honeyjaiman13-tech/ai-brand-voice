import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import CountVectorizer
import google.generativeai as genai

# Page Configuration
st.set_page_config(page_title="AI Brand Voice Generator", layout="wide")

st.title("AI Brand Voice Generator")
st.caption("Powered by Google Gemini and Streamlit")

# Session State Setup
if "generated_content" not in st.session_state:
    st.session_state.generated_content = ""
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar for Settings
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Gemini API Key:", type="password", help="Enter Key from Google AI Studio")
    selected_model = st.selectbox("Select Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
    st.divider()
    st.caption("Built for NASSCOM x Google Cloud Program")

# UI Tabs
tab1, tab2, tab3 = st.tabs(["Generator Workspace", "Brand Analytics", "History and Refinement"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Brand Details")
        brand_name = st.text_input("Brand Name*", placeholder="e.g., EcoFresh")
        industry = st.text_input("Industry / Niche*", placeholder="e.g., Sustainable Skincare")
        target_audience = st.text_input("Target Audience", placeholder="e.g., Gen-Z eco-conscious consumers")
        
        st.subheader("2. Sample Reference Text")
        sample_text = st.text_area(
            "Paste brand sample copy (taglines, emails, posts)*",
            placeholder="Clean ingredients for a clearer tomorrow. Radiance, simplified. Pure hydration.",
            height=120
        )
        
        st.subheader("3. Content Format")
        content_type = st.selectbox(
            "Content Type*",
            ["Social Media Post", "Marketing Email", "Ad Headline & Tagline", "Product Description", "Blog Introduction"]
        )
        generate_btn = st.button("Generate Brand Content", type="primary", use_container_width=True)

# NLP Processing for Voice Extraction
top_keywords = []
df_freq = pd.DataFrame()

if sample_text.strip():
    cleaned = re.sub(r'[^\w\s]', '', sample_text.lower())
    words = cleaned.split()
    try:
        vec = CountVectorizer(stop_words='english')
        X = vec.fit_transform([cleaned])
        df_freq = pd.DataFrame(X.toarray().T, index=vec.get_feature_names_out(), columns=['Freq']).sort_values('Freq', ascending=False).head(5)
        top_keywords = df_freq.index.tolist()
    except Exception:
        top_keywords = words[:5]

with tab2:
    st.subheader("Extracted Brand Style Metrics")
    if sample_text.strip():
        st.write(f"**Top Extracted Keywords:** {', '.join(top_keywords)}")
        if not df_freq.empty:
            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.barh(df_freq.index, df_freq['Freq'], color='#1A73E8')
            ax.invert_yaxis()
            ax.set_title("Vocabulary Frequency Distribution")
            st.pyplot(fig)
    else:
        st.info("Paste sample text in Tab 1 to see NLP metrics.")

with col2:
    st.subheader("4. Generated Output")
    if generate_btn:
        if not api_key:
            st.error("Please enter your Gemini API Key in the sidebar.")
        elif not brand_name or not industry or not sample_text:
            st.warning("Please complete all required fields (*).")
        else:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(selected_model)
                prompt = f"""
                You are a Brand Strategist.
                Brand Name: {brand_name}
                Industry: {industry}
                Target Audience: {target_audience}
                Extracted Keywords: {', '.join(top_keywords)}
                Reference Tone/Style: "{sample_text}"
                
                Task: Generate 2 high-converting variations of a {content_type}. Strictly match the brand's tone, rhythm, and vocabulary.
                """
                with st.spinner("Generating copy via Gemini..."):
                    res = model.generate_content(prompt)
                    
                st.session_state.generated_content = res.text
                st.session_state.history.append({"type": content_type, "output": res.text})
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.generated_content:
        st.markdown(st.session_state.generated_content)
        st.text_area("Copyable Raw Output:", value=st.session_state.generated_content, height=150)

with tab3:
    st.subheader("Content Refinement")
    if st.session_state.generated_content:
        refine_instruction = st.text_input("Enter refinement instruction (e.g., 'Make it shorter and punchy'):")
        if st.button("Apply Refinement"):
            if api_key:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(selected_model)
                refine_prompt = f"Original: {st.session_state.generated_content}\n\nFeedback: {refine_instruction}\n\nRewrite adhering to original brand voice."
                with st.spinner("Refining..."):
                    res = model.generate_content(refine_prompt)
                    st.session_state.generated_content = res.text
                    st.rerun()
    st.divider()
    st.subheader("Session History")
    for item in reversed(st.session_state.history):
        st.caption(f"**Format:** {item['type']}")
        st.text(item['output'][:120] + "...")
