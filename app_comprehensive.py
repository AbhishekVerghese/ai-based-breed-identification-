# Comprehensive Livestock Breed Identification System
# Features: Image Quality Gate, Crop Assist, Lighting Warning, Prediction Controls,
# Explainability, Human-in-the-loop, Field Readiness, Reporting, Performance

import json
import numpy as np
import cv2
import io
import time
import csv
import hashlib
from pathlib import Path
from PIL import Image, ImageEnhance
import streamlit as st
from tensorflow import keras
import tensorflow as tf
from datetime import datetime
import base64
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import warnings
warnings.filterwarnings('ignore')

# -------------------- Configuration --------------------
ROOT = Path(".")
ART = ROOT / "artifacts_hier"
ROUTER_PATH = ART / "router_config.json"
CORRECTIONS_FILE = ROOT / "corrections.csv"
SESSION_FILE = ROOT / "session_history.json"

# Language translations
TRANSLATIONS = {
    'en': {
        'title': 'AI Livestock Breed Identification',
        'subtitle': 'Upload an image to identify species and breed with confidence',
        'upload_btn': 'Upload Image',
        'quality_check': 'Image Quality Check',
        'crop_assist': 'Auto Crop Assist',
        'lighting_check': 'Lighting Check',
        'species_pred': 'Species Prediction',
        'breed_pred': 'Breed Prediction',
        'confidence': 'Confidence',
        'top_suggestions': 'Top Suggestions',
        'needs_manual': 'Needs Manual Check',
        'correction': 'Not correct? Select actual breed:',
        'save_correction': 'Save Correction',
        'download_report': 'Download Report (PDF)',
        'history': 'Prediction History',
        'privacy': 'Privacy Settings',
        'offline_mode': 'Offline Mode (Lite Model)',
        'camera': 'Open Camera',
        'batch_upload': 'Batch Upload Mode',
        'latency': 'Inference Time',
        'auto_crop': 'Auto-crop',
        'use_detector': 'Use detector (placeholder)'
    },
    'hi': {
        'title': 'एआई पशु नस्ल पहचान प्रणाली',
        'subtitle': 'प्रजाति और नस्ल की पहचान के लिए छवि अपलोड करें',
        'upload_btn': 'छवि अपलोड करें',
        'quality_check': 'छवि गुणवत्ता जांच',
        'crop_assist': 'ऑटो क्रॉप सहायता',
        'lighting_check': 'रोशनी जांच',
        'species_pred': 'प्रजाति भविष्यवाणी',
        'breed_pred': 'नस्ल भविष्यवाणी',
        'confidence': 'विश्वास',
        'top_suggestions': 'शीर्ष सुझाव',
        'needs_manual': 'मैनुअल जांच की आवश्यकता है',
        'correction': 'सही नहीं है? वास्तविक नस्ल चुनें:',
        'save_correction': 'सुधार सहेजें',
        'download_report': 'रिपोर्ट डाउनलोड करें (PDF)',
        'history': 'भविष्यवाणी इतिहास',
        'privacy': 'गोपनीयता सेटिंग्स',
        'offline_mode': 'ऑफ़लाइन मोड (लाइट मॉडल)',
        'camera': 'कैमरा खोलें',
        'batch_upload': 'बैच अपलोड मोड',
        'latency': 'इन्फेरेंस समय',
        'auto_crop': 'ऑटो क्रॉप',
        'use_detector': 'डिटेक्टर उपयोग करें (प्लेसहोल्डर)'
    },
    'te': {
        'title': 'AI పశు జాతి గుర్తింపు వ్యవస్థ',
        'subtitle': 'జాతి మరియు వంశాన్ని గుర్తించేందుకు ఇమేజ్ అప్‌లోడ్ చేయండి',
        'upload_btn': 'ఇమేజ్ అప్‌లోడ్ చేయండి',
        'quality_check': 'ఇమేజ్ నాణ్యత పరీక్ష',
        'crop_assist': 'ఆటో క్రాప్ సహాయం',
        'lighting_check': 'వెలుగు పరీక్ష',
        'species_pred': 'జాతి అంచనా',
        'breed_pred': 'వంశం అంచనా',
        'confidence': 'విశ్వాసం',
        'top_suggestions': 'టాప్ సూచనలు',
        'needs_manual': 'మాన్యువల్ చెక్ అవసరం',
        'correction': 'సరిగా లేదు? నిజమైన వంశాన్ని ఎంచుకోండి:',
        'save_correction': 'సవరణను సేవ్ చేయండి',
        'download_report': 'నివేదికను డౌన్‌లోడ్ చేయండి (PDF)',
        'history': 'అంచనా చరిత్ర',
        'privacy': 'గోప్యత సెట్టింగులు',
        'offline_mode': 'ఆఫ్‌లైన్ మోడ్ (లైట్ మోడల్)',
        'camera': 'కెమెరా తెరవండి',
        'batch_upload': 'బ్యాచ్ అప్‌లోడ్ మోడ్',
        'latency': 'ఇన్ఫరెన్స్ సమయం',
        'auto_crop': 'ఆటో-క్రాప్',
        'use_detector': 'డిటెక్టరును ఉపయోగించండి (ప్లేస్‌హోల్డర్)'
    },
    'ta': {
        'title': 'AI கால்நடை இன அடையாள அமைப்பு',
        'subtitle': 'இனம் மற்றும் இனத்தை அடையாளம் காண படத்தை பதிவேற்றவும்',
        'upload_btn': 'படத்தை பதிவேற்றவும்',
        'quality_check': 'பட தர சோதனை',
        'crop_assist': 'ஆட்டோ கிராப் உதவி',
        'lighting_check': 'விளக்கம் சோதனை',
        'species_pred': 'இன கணிப்பு',
        'breed_pred': 'இன கணிப்பு',
        'confidence': 'நம்பிக்கை',
        'top_suggestions': 'மேல் பரிந்துரைகள்',
        'needs_manual': 'கைமுறை சோதனை தேவை',
        'correction': 'சரியில்லையா? உண்மையான இனத்தை தேர்ந்தெடுக்கவும்:',
        'save_correction': 'திருத்தத்தை சேமிக்கவும்',
        'download_report': 'அறிக்கையை பதிவிறக்கவும் (PDF)',
        'history': 'கணிப்பு வரலாறு',
        'privacy': 'தனியுரிமை அமைப்புகள்',
        'offline_mode': 'ஆஃப்லைன் பயன்முறை (லைட் மாடல்)',
        'camera': 'கேமராவை திறக்கவும்',
        'batch_upload': 'பேட்ச் பதிவேற்றும் பயன்முறை',
        'latency': 'அனுமான நேரம்',
        'auto_crop': 'ஆட்டோ-கிராப்',
        'use_detector': 'டிடெக்டரை பயன்படுத்து (பிளேஸ்ஹோல்டர்)'
    }
}

# Breed trait evidence database
BREED_TRAITS = {
    'Gir': {
        'dome_forehead': True,
        'lyre_horns': True,
        'dewlap': True,
        'hump': True,
        'coat_color': 'Reddish brown to dark brown',
        'region': 'Gujarat, Rajasthan'
    },
    'Sahiwal': {
        'dome_forehead': False,
        'lyre_horns': False,
        'dewlap': True,
        'hump': True,
        'coat_color': 'Reddish dun to dark brown',
        'region': 'Punjab, Haryana'
    },
    'Murrah': {
        'dome_forehead': False,
        'lyre_horns': True,
        'dewlap': False,
        'hump': False,
        'coat_color': 'Jet black',
        'region': 'Haryana, Punjab'
    },
    'Deccani': {
        'wool_type': 'Coarse carpet wool',
        'ear_length': 'Medium',
        'coat_color': 'Black, white, or mixed',
        'region': 'Maharashtra, Karnataka'
    },
    'Nellore': {
        'wool_type': 'Hair type',
        'ear_length': 'Long and drooping',
        'coat_color': 'White with red spots',
        'region': 'Andhra Pradesh, Telangana'
    }
}

# -------------------- Streamlit Setup --------------------
st.set_page_config(
    page_title="AI Livestock Breed Identification System",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .quality-badge {
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
        font-size: 12px;
        margin: 2px;
        display: inline-block;
    }
    .quality-pass { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .quality-warn { background-color: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
    .quality-fail { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .latency-chip {
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
        color: white;
    }
    .latency-good { background-color: #28a745; }
    .latency-medium { background-color: #ffc107; }
    .latency-poor { background-color: #dc3545; }
    .suggestion-chip {
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 5px 10px;
        border-radius: 15px;
        margin: 2px;
        cursor: pointer;
        border: 1px solid #bbdefb;
    }
    .suggestion-chip:hover { background-color: #bbdefb; }
    .trait-evidence {
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 12px;
        margin: 2px;
    }
    .trait-present { background-color: #e8f5e8; color: #2e7d32; border: 1px solid #c8e6c9; }
    .trait-absent { background-color: #ffebee; color: #c62828; border: 1px solid #ffcdd2; }
    .stProgress > div > div > div > div {
        background-color: #1976d2;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Helper Functions --------------------
def load_config(path: Path):
    if not path.exists():
        st.error(f"router_config.json not found at {path}")
        st.stop()
    cfg = json.loads(path.read_text())
    cfg["species_model"] = str(Path(cfg["species_model"]))
    cfg["sheep_model"] = str(Path(cfg["sheep_model"]))
    cfg["bovine_model"] = str(Path(cfg["bovine_model"]))
    cfg["img_size"] = int(cfg.get("img_size", 224))
    return cfg

def ensure_labels(field, fallback_json, fallback_dir):
    val = cfg.get(field)
    if isinstance(val, list) and val:
        return val
    if isinstance(val, str) and Path(val).exists():
        return json.loads(Path(val).read_text())
    train_dir = ROOT / "hierarchical_data" / fallback_dir / "train"
    if train_dir.exists():
        return sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    if fallback_json and Path(fallback_json).exists():
        return json.loads(Path(fallback_json).read_text())
    return []

def load_mobile_net_model():
    """Load lightweight MobileNetV3 for offline mode"""
    try:
        base_model = tf.keras.applications.MobileNetV3Small(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(2, activation='softmax')  # species only
        ])
        return model
    except Exception as e:
        st.error(f"Failed to load MobileNet model: {e}")
        return None

# Image Quality Assessment
def assess_image_quality(image: Image.Image, file_size_bytes: Optional[int] = None, max_file_size_bytes: Optional[int] = None) -> Dict:
    """Comprehensive image quality assessment"""
    results = {}
    
    # Resolution check
    width, height = image.size
    results['resolution'] = {
        'width': width,
        'height': height,
        'pass': width >= 512 and height >= 512,
        'score': min(width, height) / 512
    }
    
    # Blur detection using Laplacian variance
    img_array = np.array(image.convert('L'))
    laplacian = cv2.Laplacian(img_array, cv2.CV_64F)
    variance = laplacian.var()
    results['blur'] = {
        'variance': variance,
        'pass': variance >= 120,
        'score': variance / 120
    }
    
    # Brightness check
    brightness = np.mean(img_array) / 255.0
    results['brightness'] = {
        'value': brightness,
        'pass': 0.25 <= brightness <= 0.75,
        'score': min(brightness / 0.25, (1 - brightness) / 0.25) if brightness < 0.5 else 1.0
    }
    
    # Dynamic range
    dynamic_range = np.std(img_array) / 255.0
    results['dynamic_range'] = {
        'value': dynamic_range,
        'pass': dynamic_range >= 0.15,
        'score': dynamic_range / 0.15
    }
    
    # File size gate (optional)
    if file_size_bytes is not None and max_file_size_bytes is not None:
        results['file_size'] = {
            'bytes': file_size_bytes,
            'max_bytes': max_file_size_bytes,
            'pass': file_size_bytes <= max_file_size_bytes,
            'score': min(1.0, max_file_size_bytes / max(1, file_size_bytes))
        }
    
    # Overall quality score
    scores = [results['resolution']['score'], results['blur']['score'], 
              results['brightness']['score'], results['dynamic_range']['score']]
    results['overall_score'] = np.mean(scores)
    results['overall_pass'] = all([results['resolution']['pass'], results['blur']['pass'], 
                                  results['brightness']['pass'], results['dynamic_range']['pass']])
    
    return results

def auto_crop_image(image: Image.Image, use_face_detection: bool = False) -> Image.Image:
    """Auto crop image to focus on animal"""
    try:
        # Convert to OpenCV format
        img_array = np.array(image.convert('RGB'))
        
        if use_face_detection:
            # Simple center-weighted crop if no face detection available
            height, width = img_array.shape[:2]
            
            # Calculate center region (assuming animal is centered)
            center_x, center_y = width // 2, height // 2
            crop_size = min(width, height) * 0.8
            
            x1 = max(0, int(center_x - crop_size // 2))
            y1 = max(0, int(center_y - crop_size // 2))
            x2 = min(width, int(center_x + crop_size // 2))
            y2 = min(height, int(center_y + crop_size // 2))
            
            cropped = img_array[y1:y2, x1:x2]
            return Image.fromarray(cropped)
        else:
            # Simple square crop from center
            min_dim = min(img_array.shape[0], img_array.shape[1])
            start_x = (img_array.shape[1] - min_dim) // 2
            start_y = (img_array.shape[0] - min_dim) // 2
            
            cropped = img_array[start_y:start_y+min_dim, start_x:start_x+min_dim]
            return Image.fromarray(cropped)
            
    except Exception as e:
        st.warning(f"Auto-crop failed: {e}. Using original image.")
        return image

def prepare_image(img: Image.Image, img_size: int, normalize: bool = True):
    """Prepare image for model input"""
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((img_size, img_size))
    arr = np.asarray(img).astype("float32")
    if normalize:
        arr = arr / 255.0
    return arr

def get_gradcam_heatmap(model, img_array, last_conv_layer_name):
    """Generate Grad-CAM heatmap"""
    try:
        # Create a model that maps the input image to the activations of the last conv layer
        # as well as the output predictions
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        # Compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        
        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)
        
        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except Exception as e:
        st.error(f"Grad-CAM generation failed: {e}")
        return None

def generate_evidence_cards(predicted_breed: str, confidence: float) -> List[Dict]:
    """Generate evidence cards for breed prediction"""
    cards = []
    
    traits = BREED_TRAITS.get(predicted_breed, {})
    
    for trait, present in traits.items():
        if trait in ['dome_forehead', 'lyre_horns', 'dewlap', 'hump']:
            cards.append({
                'trait': trait.replace('_', ' ').title(),
                'present': present,
                'evidence': f"{'✓' if present else '✗'} {trait.replace('_', ' ').title()}",
                'confidence': confidence * 0.8
            })
    
    return cards

def save_correction(image_hash: str, predicted: str, actual: str, confidence: float):
    """Save user correction for active learning"""
    try:
        file_exists = CORRECTIONS_FILE.exists()
        with open(CORRECTIONS_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'image_hash', 'predicted', 'actual', 'confidence'])
            writer.writerow([datetime.now().isoformat(), image_hash, predicted, actual, confidence])
        return True
    except Exception as e:
        st.error(f"Failed to save correction: {e}")
        return False

def generate_pdf_report(image: Image.Image, species: str, breed: str, confidence: float, 
                       timestamp: str, location: str = None, notes: str = "") -> bytes:
    """Generate PDF case report"""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        
        story = []
        
        # Title
        story.append(Paragraph("Livestock Breed Identification Report", title_style))
        story.append(Spacer(1, 20))
        
        # Image (thumbnail)
        img_buffer = io.BytesIO()
        image.thumbnail((400, 400))
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        img = RLImage(img_buffer, width=4*inch, height=3*inch)
        story.append(img)
        story.append(Spacer(1, 20))
        
        # Results table
        data = [
            ['Field', 'Value'],
            ['Date/Time', timestamp],
            ['Species', species.title()],
            ['Breed', breed.replace('_', ' ').title()],
            ['Confidence', f"{confidence:.1%}"],
            ['Location', location if location else 'Not provided'],
            ['Notes', notes if notes else 'None']
        ]
        
        table = Table(data, colWidths=[2*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(table)
        story.append(Spacer(1, 30))
        
        # Disclaimer
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey
        )
        story.append(Paragraph(
            "This report is generated by an AI system and should be used as a reference only. "
            "Always consult with qualified veterinary or livestock experts for critical decisions.",
            disclaimer_style
        ))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.read()
        
    except Exception as e:
        st.error(f"PDF generation failed: {e}")
        return None

def get_image_hash(image: Image.Image) -> str:
    """Get hash of image for deduplication"""
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    return hashlib.md5(img_bytes.getvalue()).hexdigest()

# -------------------- Session State Management --------------------
def init_session_state():
    """Initialize session state variables"""
    if 'language' not in st.session_state:
        st.session_state.language = 'en'
    if 'offline_mode' not in st.session_state:
        st.session_state.offline_mode = False
    if 'privacy_mode' not in st.session_state:
        st.session_state.privacy_mode = False
    if 'session_history' not in st.session_state:
        st.session_state.session_history = []
    if 'corrections' not in st.session_state:
        st.session_state.corrections = []
    if 'active_learning_queue' not in st.session_state:
        st.session_state.active_learning_queue = []
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = []

# -------------------- Main App --------------------
def main():
    init_session_state()
    
    # Load configuration
    global cfg
    cfg = load_config(ROUTER_PATH)
    
    # Load models
    if st.session_state.offline_mode:
        species_model = load_mobile_net_model()
        sheep_model = None
        bovine_model = None
    else:
        species_model = keras.models.load_model(cfg["species_model"])
        sheep_model = keras.models.load_model(cfg["sheep_model"])
        bovine_model = keras.models.load_model(cfg["bovine_model"])
    
    # Language selector
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        language = st.selectbox('🌐 Language', ['en', 'hi', 'te', 'ta'], 
                                format_func=lambda x: {'en':'English', 'hi':'हिंदी', 'te':'తెలుగు', 'ta':'தமிழ்'}[x])
        st.session_state.language = language
    
    t = TRANSLATIONS[language]
    
    # Header
    st.markdown(f"<h1 style='text-align: center; color: #1976d2;'>{t['title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #666;'>{t['subtitle']}</p>", unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model_choice = st.selectbox(
            "Model Backbone",
            ["EffNet-B0 (Fast)", "EffNet-B2 (Accurate)", "ConvNeXt-Tiny (HQ)"],
            help="Choose between speed and accuracy"
        )
        # Display indicative latency/accuracy expectations
        if model_choice == "EffNet-B0 (Fast)":
            st.caption("⏱️ ~200–300ms • ✅ good accuracy")
        elif model_choice == "EffNet-B2 (Accurate)":
            st.caption("⏱️ ~300–500ms • ✅✅ higher accuracy")
        else:
            st.caption("⏱️ ~500–800ms • ✅✅✅ highest accuracy")
        
        # Confidence thresholds
        species_threshold = st.slider("Species Confidence Threshold", 0.5, 0.95, 0.80, 0.05)
        breed_threshold = st.slider("Breed Confidence Threshold", 0.5, 0.95, 0.70, 0.05)
        
        # Quality gates
        st.subheader("Quality Gates")
        enable_quality_check = st.checkbox("Enable Image Quality Check", value=True)
        enable_crop_assist = st.checkbox(t.get('auto_crop', 'Auto-crop'), value=True)
        use_detector_placeholder = st.checkbox(t.get('use_detector', 'Use detector (placeholder)'), value=False, help="Detector-based crop coming soon; currently center square crop")
        enable_lighting_check = st.checkbox("Enable Lighting Check", value=True)
        max_file_size_mb = st.number_input(t.get('max_file_size', 'Max file size (MB)'), min_value=1, max_value=50, value=5, step=1)
        
        # Privacy and offline settings
        st.subheader("Privacy & Offline")
        st.session_state.privacy_mode = st.checkbox("Don't Store Images", value=False)
        st.session_state.offline_mode = st.checkbox(t['offline_mode'], value=False)
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            img_size = st.selectbox("Image Size", [224, 300, 512], index=0)
            normalize = st.checkbox("Normalize Input", value=True)
            top_k = st.slider("Top-K Suggestions", 2, 5, 3)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📸 Image Upload")
        
        # Upload mode selection
        upload_mode = st.radio("Upload Mode", ["Single Image", "Batch Upload", "Camera Capture"])
        
        if upload_mode == "Single Image":
            uploaded_file = st.file_uploader(
                t['upload_btn'],
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=False
            )
            uploaded_files = [uploaded_file] if uploaded_file else []
            
        elif upload_mode == "Batch Upload":
            uploaded_files = st.file_uploader(
                "Upload multiple images (max 20)",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True
            )
            if len(uploaded_files) > 20:
                st.warning("Maximum 20 images allowed for batch processing")
                uploaded_files = uploaded_files[:20]
                
        else:  # Camera Capture
            camera_image = st.camera_input(t['camera'])
            uploaded_files = [camera_image] if camera_image else []
        
        if uploaded_files and len(uploaded_files) > 0:
            st.success(f"📁 {len(uploaded_files)} image(s) uploaded")
    
    with col2:
        if uploaded_files:
            st.header("🔍 Analysis Results")
            
            # Process each image
            for idx, uploaded_file in enumerate(uploaded_files):
                if len(uploaded_files) > 1:
                    st.subheader(f"Image {idx + 1}: {uploaded_file.name}")
                
                # Load and display image
                image = Image.open(uploaded_file)
                image_hash = get_image_hash(image)
                
                # Display original image
                col_img1, col_img2 = st.columns([1, 1])
                with col_img1:
                    st.image(image, caption="Original Image", use_column_width=True)
                
                # Image quality assessment
                if enable_quality_check:
                    with st.spinner("Checking image quality..."):
                        # Determine file size if available
                        try:
                            file_size_bytes = getattr(uploaded_file, 'size', None)
                            if file_size_bytes is None:
                                # Fallback for camera_input or other types
                                buf = uploaded_file.getbuffer()
                                file_size_bytes = len(buf)
                        except Exception:
                            file_size_bytes = None
                        quality_results = assess_image_quality(image, file_size_bytes=file_size_bytes, max_file_size_bytes=int(max_file_size_mb * 1024 * 1024))
                        
                        # Display quality badges
                        st.subheader(t['quality_check'])
                        col_badges1, col_badges2 = st.columns([1, 1])
                        
                        with col_badges1:
                            res_status = "✅ PASS" if quality_results['resolution']['pass'] else "❌ FAIL"
                            res_class = "quality-pass" if quality_results['resolution']['pass'] else "quality-fail"
                            st.markdown(f"<span class='quality-badge {res_class}'>{res_status} Resolution</span>", unsafe_allow_html=True)
                            
                            blur_status = "✅ PASS" if quality_results['blur']['pass'] else "❌ FAIL"
                            blur_class = "quality-pass" if quality_results['blur']['pass'] else "quality-fail"
                            st.markdown(f"<span class='quality-badge {blur_class}'>{blur_status} Sharpness</span>", unsafe_allow_html=True)
                        
                        with col_badges2:
                            bright_status = "✅ PASS" if quality_results['brightness']['pass'] else "❌ FAIL"
                            bright_class = "quality-pass" if quality_results['brightness']['pass'] else "quality-fail"
                            st.markdown(f"<span class='quality-badge {bright_class}'>{bright_status} Brightness</span>", unsafe_allow_html=True)
                            
                            range_status = "✅ PASS" if quality_results['dynamic_range']['pass'] else "⚠️ WARN"
                            range_class = "quality-pass" if quality_results['dynamic_range']['pass'] else "quality-warn"
                            st.markdown(f"<span class='quality-badge {range_class}'>{range_status} Dynamic Range</span>", unsafe_allow_html=True)
                            
                            # File size badge if computed
                            if 'file_size' in quality_results:
                                fs_pass = quality_results['file_size']['pass']
                                fs_status = "✅ PASS" if fs_pass else "❌ FAIL"
                                fs_class = "quality-pass" if fs_pass else "quality-fail"
                                max_mb = quality_results['file_size']['max_bytes'] / (1024*1024)
                                cur_mb = quality_results['file_size']['bytes'] / (1024*1024)
                                st.markdown(f"<span class='quality-badge {fs_class}'>{fs_status} File Size ({cur_mb:.1f}MB ≤ {max_mb:.0f}MB)</span>", unsafe_allow_html=True)
                        
                        # Quality tips
                        if not quality_results['overall_pass']:
                            tips = []
                            if not quality_results['resolution']['pass']:
                                tips.append("📸 Try taking a higher resolution photo")
                            if not quality_results['blur']['pass']:
                                tips.append("🎯 Hold camera steady or use tripod")
                            if not quality_results['brightness']['pass']:
                                tips.append("💡 Low light detected—try flash / move to shade")
                            if not quality_results['dynamic_range']['pass']:
                                tips.append("🌓 Ensure good contrast between animal and background")
                            
                            st.warning("💡 " + " | ".join(tips))
                
                # Auto crop assist
                if enable_crop_assist:
                    with st.spinner("Auto-cropping image..."):
                        cropped_image = auto_crop_image(image, use_face_detection=use_detector_placeholder)
                        
                        if cropped_image != image:
                            with col_img2:
                                st.image(cropped_image, caption="Auto-cropped", use_column_width=True)
                            processing_image = cropped_image
                        else:
                            processing_image = image
                else:
                    processing_image = image
                
                # Prepare image for model
                img_array = prepare_image(processing_image, img_size, normalize)
                img_batch = np.expand_dims(img_array, 0)
                
                # Measure inference time
                start_time = time.time()
                
                # Species prediction
                if st.session_state.offline_mode and species_model:
                    sp_probs = species_model.predict(img_batch, verbose=0)[0]
                    # Simulate breed prediction for offline mode
                    sp_label = "bovine" if sp_probs[0] > 0.5 else "sheep"
                    sp_conf = max(sp_probs)
                    breed_probs = np.random.dirichlet(np.ones(5))  # Mock breed probabilities
                else:
                    sp_probs = species_model.predict(img_batch, verbose=0)[0]
                    sp_idx = int(np.argmax(sp_probs))
                    sp_label = species_labels[sp_idx] if sp_idx < len(species_labels) else "unknown"
                    sp_conf = float(sp_probs[sp_idx])
                
                inference_time = (time.time() - start_time) * 1000  # ms
                
                # Display latency indicator
                latency_class = "latency-good" if inference_time < 300 else ("latency-medium" if inference_time < 600 else "latency-poor")
                st.markdown(f"<span class='latency-chip {latency_class}'>⏱️ {inference_time:.0f}ms</span>", unsafe_allow_html=True)
                
                # Species results
                st.subheader(t['species_pred'])
                if sp_conf >= species_threshold:
                    st.success(f"**{sp_label.title()}** - Confidence: {sp_conf:.1%}")
                    st.progress(sp_conf)
                else:
                    st.warning(f"**{sp_label.title()}** - Low confidence: {sp_conf:.1%}")
                    st.progress(sp_conf)
                    st.info("🔍 Species confidence below threshold - manual verification recommended")

                # Router guardrail - show both breed heads' top-1 when species is uncertain
                if (sp_conf < species_threshold) and (not st.session_state.offline_mode):
                    st.info("🔧 Model is unsure—compare these two likely options.")
                    sheep_probs_guard = sheep_model.predict(img_batch, verbose=0)[0]
                    bovine_probs_guard = bovine_model.predict(img_batch, verbose=0)[0]
                    sheep_idx_guard = int(np.argmax(sheep_probs_guard))
                    bovine_idx_guard = int(np.argmax(bovine_probs_guard))
                    sheep_best = (
                        sheep_labels[sheep_idx_guard] if sheep_idx_guard < len(sheep_labels) else f"sheep_class_{sheep_idx_guard}",
                        float(sheep_probs_guard[sheep_idx_guard])
                    )
                    bovine_best = (
                        bovine_labels[bovine_idx_guard] if bovine_idx_guard < len(bovine_labels) else f"bovine_class_{bovine_idx_guard}",
                        float(bovine_probs_guard[bovine_idx_guard])
                    )
                    col_guard1, col_guard2 = st.columns([1, 1])
                    with col_guard1:
                        st.metric("Sheep head top-1", sheep_best[0].replace('_',' ').title(), f"{sheep_best[1]:.1%}")
                    with col_guard2:
                        st.metric("Bovine head top-1", bovine_best[0].replace('_',' ').title(), f"{bovine_best[1]:.1%}")
                
                # Breed prediction (only if species confidence is high enough)
                if sp_conf >= species_threshold and not st.session_state.offline_mode:
                    st.subheader(t['breed_pred'])
                    
                    # Route to appropriate breed model
                    head = "sheep" if sp_label.lower() == "sheep" else "bovine"
                    labels = sheep_labels if head == "sheep" else bovine_labels
                    
                    breed_probs = (sheep_model.predict(img_batch, verbose=0)[0] if head == "sheep" 
                                  else bovine_model.predict(img_batch, verbose=0)[0])
                    
                    br_idx = int(np.argmax(breed_probs))
                    br_label = labels[br_idx] if br_idx < len(labels) else f"{head}_class_{br_idx}"
                    br_conf = float(breed_probs[br_idx])
                    
                    if br_conf >= breed_threshold:
                        st.success(f"**{br_label.replace('_', ' ').title()}** - Confidence: {br_conf:.1%}")
                        st.progress(br_conf)
                        
                        # Evidence cards
                        with st.expander("🔍 Evidence Cards"):
                            evidence_cards = generate_evidence_cards(br_label, br_conf)
                            for card in evidence_cards:
                                trait_class = "trait-present" if card['present'] else "trait-absent"
                                st.markdown(f"<span class='trait-evidence {trait_class}'>{card['evidence']}</span>", unsafe_allow_html=True)
                        
                        # Grad-CAM visualization
                        with st.expander("🔥 Grad-CAM Heatmap"):
                            st.info("Heatmap showing model attention areas")
                            # Placeholder for Grad-CAM (would need model layer names)
                            st.image(processing_image, caption="Heatmap overlay would appear here", use_column_width=True)
                        
                    else:
                        st.warning(f"**{br_label.replace('_', ' ').title()}** - Low confidence: {br_conf:.1%}")
                        st.progress(br_conf)
                        st.info("🔍 Breed confidence below threshold")
                    
                    # Top-K suggestions
                    st.subheader(t['top_suggestions'])
                    top_breeds = []
                    cols = st.columns(min(top_k, 5))
                    for i in range(min(top_k, len(labels))):
                        idx = int(np.argsort(breed_probs)[::-1][i])
                        breed_name = labels[idx] if idx < len(labels) else f"{head}_class_{idx}"
                        confidence = float(breed_probs[idx])
                        top_breeds.append((breed_name, confidence))
                        
                        with cols[i % len(cols)]:
                            st.write(f"{breed_name.replace('_', ' ').title()} ({confidence:.1%})")
                            if st.button("This one is correct", key=f"correct_{image_hash}_{i}"):
                                # Save user-marked correction to feedback loop
                                if save_correction(image_hash, br_label, breed_name, confidence):
                                    st.success("Marked and saved to corrections.csv")
                                    st.session_state.active_learning_queue.append({
                                        'image_hash': image_hash,
                                        'predicted': br_label,
                                        'actual': breed_name,
                                        'confidence': confidence,
                                        'timestamp': datetime.now().isoformat()
                                    })
                                else:
                                    st.error("Failed to save correction")
                    
                    # Human-in-the-loop correction
                    if br_conf < breed_threshold:
                        st.warning(t['needs_manual'])
                        with st.form(f"correction_form_{idx}"):
                            actual_breed = st.selectbox(
                                t['correction'],
                                options=labels,
                                format_func=lambda x: x.replace('_', ' ').title()
                            )
                            if st.form_submit_button(t['save_correction']):
                                if save_correction(image_hash, br_label, actual_breed, br_conf):
                                    st.success("Correction saved! 🎯")
                                    # Add to active learning queue
                                    st.session_state.active_learning_queue.append({
                                        'image_hash': image_hash,
                                        'predicted': br_label,
                                        'actual': actual_breed,
                                        'confidence': br_conf,
                                        'timestamp': datetime.now().isoformat()
                                    })
                                else:
                                    st.error("Failed to save correction")
                
                # Add to session history
                if not st.session_state.privacy_mode:
                    st.session_state.session_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'image_hash': image_hash,
                        'filename': uploaded_file.name,
                        'species': sp_label,
                        'species_confidence': sp_conf,
                        'breed': br_label if sp_conf >= species_threshold else None,
                        'breed_confidence': br_conf if sp_conf >= species_threshold else None,
                        'inference_time': inference_time
                    })
                
                # Results actions
                col_actions1, col_actions2, col_actions3 = st.columns([1, 1, 1])
                with col_actions1:
                    if st.button(f"📄 {t['download_report']}", key=f"report_{idx}"):
                        pdf_data = generate_pdf_report(
                            processing_image, sp_label, br_label if sp_conf >= species_threshold else "N/A",
                            br_conf if sp_conf >= species_threshold else sp_conf,
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        )
                        if pdf_data:
                            st.download_button(
                                label="📥 Download PDF",
                                data=pdf_data,
                                file_name=f"livestock_report_{image_hash[:8]}.pdf",
                                mime="application/pdf",
                                key=f"download_{idx}"
                            )
                
                with col_actions2:
                    if st.button("🔄 Analyze New Image", key=f"new_{idx}"):
                        st.rerun()
                
                with col_actions3:
                    if st.button("📊 View History", key=f"history_{idx}"):
                        st.session_state.show_history = True
                
                st.divider()
    
    # Session history and analytics
    if hasattr(st.session_state, 'show_history') and st.session_state.show_history:
        st.header(t['history'])
        
        if st.session_state.session_history:
            # Convert to DataFrame for better display
            import pandas as pd
            df_history = pd.DataFrame(st.session_state.session_history)
            
            # Display recent predictions
            st.dataframe(df_history.tail(10))
            
            # Analytics
            col_analytics1, col_analytics2 = st.columns([1, 1])
            with col_analytics1:
                # Confidence distribution
                fig_conf = px.histogram(df_history, x='species_confidence', nbins=20, 
                                      title="Species Confidence Distribution")
                st.plotly_chart(fig_conf, use_container_width=True)
            
            with col_analytics2:
                # Inference time trend
                df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
                fig_time = px.line(df_history, x='timestamp', y='inference_time',
                                   title="Inference Time Trend")
                st.plotly_chart(fig_time, use_container_width=True)
            
            # Export options
            col_export1, col_export2 = st.columns([1, 1])
            with col_export1:
                csv_data = df_history.to_csv(index=False)
                st.download_button(
                    label="📊 Export History (CSV)",
                    data=csv_data,
                    file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col_export2:
                if st.button("🗑️ Clear History"):
                    st.session_state.session_history = []
                    st.success("History cleared!")
                    st.rerun()
        else:
            st.info("No predictions in history yet")
    
    # Active learning queue status
    if st.session_state.active_learning_queue:
        with st.sidebar:
            st.header("🎯 Active Learning")
            st.success(f"📚 {len(st.session_state.active_learning_queue)} images queued for retraining")
            
            if st.button("Export Training Data"):
                training_data = pd.DataFrame(st.session_state.active_learning_queue)
                csv_data = training_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Training CSV",
                    data=csv_data,
                    file_name=f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()