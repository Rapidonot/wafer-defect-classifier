import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import plotly.express as px
from model_utils import WaferClassifier

# Page Config
st.set_page_config(page_title="WBM Defect Classifier", layout="wide")

st.title("Semiconductor Wafer Defect Classifier")
st.markdown("""
This tool uses a **ResNet-18** model to identify multiple simultaneous defect patterns in Wafer Bin Maps (WBM).
""")

# sidebar ( where you can adjust the detection threshold ) 
st.sidebar.header("Model Configuration")
weights_path = "resnet18_bce.pth" 
threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.5)


@st.cache_resource
def load_wafer_model():
    try:
        return WaferClassifier(weights_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
classifier = load_wafer_model()

# converting the images into a format the model can understand 
def rgb_to_categorical(pil_image):
    img_np = np.array(pil_image.convert('RGB'))
    h, w, _ = img_np.shape
    
    # checking if image is already in the format that supports the model 
    if img_np.max() <= 2:
        return pil_image
        
    categorical_matrix = np.zeros((h, w), dtype=np.uint8)
    
    R = img_np[:, :, 0].astype(int)
    G = img_np[:, :, 1].astype(int)
    B = img_np[:, :, 2].astype(int)
    
    
    yellow_mask = (R > 140) & (G > 140) & (B < 120)
    categorical_matrix[yellow_mask] = 2
    
   
    teal_mask = (G > 70) & (G > R + 15) & (~yellow_mask)
    categorical_matrix[teal_mask] = 1
    
    return Image.fromarray(categorical_matrix)


def colorise_wafer_map(pil_image):
    matrix = np.array(pil_image)
    if matrix.ndim == 3:
        matrix = matrix[:, :, 0]
        
   
    if matrix.max() > 2:
        return pil_image.resize((600, 600), Image.NEAREST)
        
    h, w = matrix.shape
    colored_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    color_bg = [40, 0, 70]        # (purple)
    color_normal = [30, 160, 150] # (teal)
    color_defect = [255, 240, 0]  # (yellow)
    
    colored_img[matrix == 0] = color_bg
    colored_img[matrix == 1] = color_normal
    colored_img[matrix >= 2] = color_defect
    
    img = Image.fromarray(colored_img, mode='RGB')
    return img.resize((600, 600), Image.NEAREST)

# File uploader
uploaded_file = st.file_uploader("Upload a Wafer Bin Map (Image)", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    user_image = Image.open(uploaded_file)
    
   
    ai_ready_image = rgb_to_categorical(user_image)
    
    
    display_image = colorise_wafer_map(user_image)
    
    with col1:
        st.subheader("Input Wafer Map")
        st.image(display_image, use_container_width=True)
        
       

    if classifier:
        with st.spinner('Classifying'):
            predictions = classifier.predict(ai_ready_image)
            
        df_results = pd.DataFrame(list(predictions.items()), columns=['Defect Class', 'Confidence'])
        df_results['Detected'] = df_results['Confidence'] >= threshold
        
        with col2:
            st.subheader("Classification Results")
            
            active_defects = df_results[df_results['Detected'] == True]['Defect Class'].tolist()
            if not active_defects:
                st.success("No defects detected (above threshold)")
            else:
                st.warning(f" Detected: {', '.join(active_defects)}")

            # bar chart
            fig = px.bar(
                df_results, 
                x='Confidence', 
                y='Defect Class', 
                orientation='h',
                color='Confidence',
                color_continuous_scale='RdYlGn_r',
                range_x=[0, 1]
            )
            fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
            st.plotly_chart(fig, use_container_width=True)
# a place to see the raw confidence scores 
    with st.expander("See Raw Confidence Scores"):
        st.table(df_results.sort_values(by='Confidence', ascending=False))
else:
    st.info("Please upload a wafer map image to begin.")