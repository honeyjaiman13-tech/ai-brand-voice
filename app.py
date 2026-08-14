import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import requests
import json
from sklearn.feature_extraction.text import CountVectorizer

# Page Configuration
st.set_page_config(page_title="AI Brand Voice Generator", layout="wide")

st.title("AI Brand Voice Generator")
st.caption("Powered by Google Gemini & Streamlit")

# Session State Setup
if "generated_content" not in st.session_state:
    st.session_state.generated_content = ""
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar for Settings
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Gemini API Key:", type="password", help="Enter Key from Google AI Studio")
    selected_model = st.selectbox(
        "Select Model",
        ["gemini-2.5-flash-lite", "gemini-2.5-pro", "gemini-1.5-flash"]
    )
    st.divider()
    st.caption("Built for NASSCOM x Google Cloud Program")

# Helper function to call Gemini API reliably via REST API
def call_gemini_api(prompt_text, key, model_name):
    clean_model = model_name.replace("models/", "")
    
    # Primary Google GenAI endpoint
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt_text}]
            }
        ]
    }
    
    resp = requests.post(url, headers=headers, json=payload)
    data = resp.json()
    
    if resp.status_code != 200:
        # Fallback to general endpoint if specific version 404s
        alt_url = f"https://generativelanguage.googleapis.com/v1/models/{clean_model}:generateContent?key={key}"
        alt_resp = requests.post(alt_url, headers=headers, json=payload)
        if alt_resp.status_code == 200:
            data = alt_resp.json()
        else:
            err_msg = data.get("error", {}).get("message", resp.text)
            raise Exception(f"API Error ({resp.status_code}): {err_msg}")
            
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise Exception(f"Unexpected response format: {json.dumps(data)}")

# UI Tabs
tab1, tab2, tab3 = st.tabs(["Generator Workspace", "Brand Analytics", "History & Refinement"])

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
            prompt = f"""
            You are an expert Brand Strategist.
            Brand Name: {brand_name}
            Industry: {industry}
            Target Audience: {target_audience}
            Extracted Keywords: {', '.join(top_keywords)}
            Reference Tone/Style: "{sample_text}"
            
            Task: Generate 2 high-converting variations of a {content_type}. Strictly match the brand's tone, rhythm, and vocabulary.
            """
            try:
                with st.spinner("Generating copy via Gemini API..."):
                    result_text = call_gemini_api(prompt, api_key.strip(), selected_model)
                    
                st.session_state.generated_content = result_text
                st.session_state.history.append({"type": content_type, "output": result_text})
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
                refine_prompt = f"Original Copy:\n{st.session_state.generated_content}\n\nInstruction: {refine_instruction}\n\nTask: Rewrite and adhere strictly to the original brand voice."
                try:
                    with st.spinner("Refining copy..."):
                        refined_text = call_gemini_api(refine_prompt, api_key.strip(), selected_model)
                        st.session_state.generated_content = refined_text
                        st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    st.divider()
    st.subheader("Session History")
    for item in reversed(st.session_state.history):
        st.caption(f"**Format:** {item['type']}")
        st.text(item['output'][:120] + "...")
