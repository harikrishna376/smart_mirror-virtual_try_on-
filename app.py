import streamlit as st
from gradio_client import Client, handle_file
import tempfile
import os
from PIL import Image

st.set_page_config(page_title="AI Smart Mirror", page_icon="👕")

st.title("👕 Smart Mirror: AI Virtual Try-On")
st.write("Based on my IJARSCT Research Publication")

# Initialize Client
@st.cache_resource
def get_client():
    return Client("yisol/IDM-VTON")

client = get_client()

# Sidebar for inputs
with st.sidebar:
    st.header("Upload Images")
    person_file = st.file_uploader("Step 1: Upload Person (Standing A-Pose)", type=['jpg', 'jpeg', 'png'])
    garment_file = st.file_uploader("Step 2: Upload Garment (Shirt/Top)", type=['jpg', 'jpeg', 'png'])

if person_file and garment_file:
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(person_file, caption="Target Person", use_column_width=True)
    with col2:
        st.image(garment_file, caption="Selected Garment", use_column_width=True)

    if st.button("🚀 Run Virtual Try-On"):
        with st.spinner("AI is warping garment... Please wait ~30s"):
            try:
                # Save uploaded files to temp paths
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as p_tmp:
                    p_tmp.write(person_file.getvalue())
                    p_path = p_tmp.name
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as g_tmp:
                    g_tmp.write(garment_file.getvalue())
                    g_path = g_tmp.name

                # API Call
                result = client.predict(
                    dict={"background": handle_file(p_path), "layers": [], "composite": None},
                    garm_img=handle_file(g_path),
                    garment_des="High-quality garment",
                    is_checked=True,
                    is_checked_crop=False,
                    denoise_steps=30,
                    seed=42,
                    api_name="/tryon"
                )

                # Display Result
                st.success("✅ Synthesis Complete!")
                st.image(result[0], caption="Final AI Try-On Result", use_column_width=True)
                
                # Cleanup
                os.remove(p_path)
                os.remove(g_path)

            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Please upload both a person image and a garment image to begin.")
