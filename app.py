import streamlit as st
from gradio_client import Client, handle_file
import tempfile
import os

# --- Page Config & Custom Styling ---
st.set_page_config(page_title="AI Smart Mirror | Pro", page_icon="👕", layout="wide")

# Custom CSS for a clean, modern "Glassmorphism" look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .upload-text { font-size: 1.2rem; font-weight: bold; color: #1e293b; }
    </style>
    """, unsafe_allow_html=True)

st.title("👕 Smart Mirror: Virtual Try-On Engine")
st.markdown("### High-Fidelity Neural Garment Warping")
st.caption("Based on IJARSCT Research Publication | Impact Factor: 7.53")

@st.cache_resource
def get_client():
    # Connecting to the IDM-VTON inference engine
    return Client("yisol/IDM-VTON")

client = get_client()

# --- Layout: Two Columns for Input, One for Output ---
col_in1, col_in2, col_out = st.columns([1, 1, 1.5])

with col_in1:
    st.markdown("<p class='upload-text'>1. The Person</p>", unsafe_allow_html=True)
    person_file = st.file_uploader("Drag & drop a photo of yourself (A-Pose)", type=['jpg', 'jpeg', 'png'], key="person")
    if person_file:
        st.image(person_file, caption="Target Person", use_column_width=True)

with col_in2:
    st.markdown("<p class='upload-text'>2. The Garment</p>", unsafe_allow_html=True)
    garment_file = st.file_uploader("Drag & drop a shirt/top image", type=['jpg', 'jpeg', 'png'], key="garment")
    if garment_file:
        st.image(garment_file, caption="Selected Shirt", use_column_width=True)

# --- Execution Logic ---
st.divider()

with col_out:
    st.header("3. AI Synthesis Result")
    
    if person_file and garment_file:
        if st.button("🚀 EXECUTE VIRTUAL TRY-ON"):
            with st.spinner("Analyzing body keypoints and diffusing texture..."):
                p_path, g_path = None, None
                try:
                    # Save to secure temporary storage
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as p_tmp:
                        p_tmp.write(person_file.getvalue())
                        p_path = p_tmp.name
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as g_tmp:
                        g_tmp.write(garment_file.getvalue())
                        g_path = g_tmp.name

                    # Inference call
                    result = client.predict(
                        dict={"background": handle_file(p_path), "layers": [], "composite": None},
                        garm_img=handle_file(g_path),
                        garment_des="Professional apparel",
                        is_checked=True,
                        is_checked_crop=False,
                        denoise_steps=30,
                        seed=42,
                        api_name="/tryon"
                    )

                    # Display and Download Result
                    st.image(result[0], caption="Final AI Try-On Result", use_column_width=True)
                    
                    with open(result[0], "rb") as file:
                        st.download_button(
                            label="📥 Download High-Res Result",
                            data=file,
                            file_name="smart_mirror_result.png",
                            mime="image/png"
                        )
                
                except Exception as e:
                    st.error(f"Execution Error: {e}")
                
                finally:
                    # Cleanup to prevent server bloat
                    if p_path and os.path.exists(p_path): os.remove(p_path)
                    if g_path and os.path.exists(g_path): os.remove(g_path)
    else:
        st.warning("Upload both images in the side panels to activate the AI engine.")
