# app.py — Streamlit UI for hierarchical species→breed prediction
# Layout updates:
# 1) Left: image upload & preview
# 2) Right: results
# 3) If any confidence < 50%, suggest manual check
# 4) After first result, ask user to upload another image (second uploader appears)

import json
import numpy as np
from pathlib import Path
from PIL import Image
import streamlit as st
from tensorflow import keras

# -------------------- Paths & page config --------------------
ROOT = Path(".")
ART = ROOT / "artifacts_hier"
ROUTER_PATH = ART / "router_config.json"

st.set_page_config(
    page_title="AI Livestock Breed Identification",
    page_icon=None,
    layout="centered",
)

# -------------------- Minimal styling (font + centered gradient title) --------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    .app-title {
        font-weight: 800;
        font-size: 2.2rem;
        line-height: 1.2;
        text-align: center;
        margin-top: -6px;
        margin-bottom: 8px;
        background: linear-gradient(90deg, #2563eb, #0ea5e9, #14b8a6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .app-subtitle {
        text-align: center;
        color: #374151;
        opacity: 0.9;
        margin-bottom: 16px;
        font-size: 1.02rem;
    }
    .badge {
        display:inline-block; padding:4px 8px; border-radius:999px; font-size:.85rem; font-weight:600;
    }
    .badge-green { background:#DCFCE7; color:#166534; }
    .badge-amber { background:#FEF3C7; color:#92400E; }
    .badge-red { background:#FEE2E2; color:#991B1B; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="app-title">AI Livestock Breed Identification</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Upload an image to identify the <strong>species</strong> and exact <strong>breed</strong> with confidence.</div>',
    unsafe_allow_html=True,
)

# -------------------- Species info content --------------------
SPECIES_INFO = {
    "sheep": {
        "title": "Sheep (Ovine)",
        "summary": (
            "Domesticated ruminants for <strong>wool, meat (mutton), and milk</strong>. "
            "Common Indian breeds include <strong>Deccani, Nellore, Marwari, Garole</strong>."
        ),
        "facts": [
            "Strong herding behavior; compact body size.",
            "Upper jaw has a dental pad (no incisors).",
            "Best photos: clear face profile or full side in good lighting.",
        ],
    },
    "bovine": {
        "title": "Bovine (Cattle & Buffalo)",
        "summary": (
            "Subfamily <strong>Bovinae</strong> — cattle (<em>Bos indicus/taurus</em>) "
            "and water buffalo (<em>Bubalus bubalis</em>). Common Indian breeds: "
            "Cattle — <strong>Gir, Sahiwal, Holstein Friesian, Jersey</strong>; "
            "Buffalo — <strong>Murrah, Jaffarabadi, Mehsana</strong>."
        ),
        "facts": [
            "Uses: milk, draught power, dung (biogas/manure).",
            "Zebu cattle often have a hump; buffalo are darker with different horn shapes.",
            "Best photos: side body or head profile; avoid motion blur.",
        ],
    },
}

# -------------------- Config & labels (robust) --------------------
def load_config(path: Path) -> dict:
    if not path.exists():
        st.error(f"`router_config.json` not found at {path}. Please generate it first.")
        st.stop()
    cfg = json.loads(path.read_text())
    cfg["species_model"] = str(Path(cfg["species_model"]))
    cfg["sheep_model"]   = str(Path(cfg["sheep_model"]))
    cfg["bovine_model"]  = str(Path(cfg["bovine_model"]))
    cfg["img_size"]      = int(cfg.get("img_size", 224))
    return cfg

cfg = load_config(ROUTER_PATH)

def _infer_labels_from_dir(dir_path: Path):
    train_dir = dir_path / "train"
    if train_dir.exists():
        return sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    return []

def _resolve_label_field(val, fallback_list):
    if isinstance(val, list) and val:
        return val
    if isinstance(val, str) and Path(val).exists():
        try:
            data = json.loads(Path(val).read_text())
            if isinstance(data, list) and data:
                return data
        except Exception:
            pass
    return fallback_list

def resolve_all_labels(cfg):
    species = cfg.get("species_labels", ["bovine", "sheep"])
    if isinstance(species, str) and Path(species).exists():
        try:
            species = json.loads(Path(species).read_text())
        except Exception:
            species = ["bovine", "sheep"]
    if not isinstance(species, list) or not species:
        species = ["bovine", "sheep"]

    sheep_fallback  = _infer_labels_from_dir(ROOT / "hierarchical_data" / "sheep_breeds")
    bovine_fallback = _infer_labels_from_dir(ROOT / "hierarchical_data" / "bovine_breeds")
    sheep  = _resolve_label_field(cfg.get("sheep_labels"),  sheep_fallback)
    bovine = _resolve_label_field(cfg.get("bovine_labels"), bovine_fallback)

    if not isinstance(sheep, list):  sheep = []
    if not isinstance(bovine, list): bovine = []
    return species, sheep, bovine

species_labels, sheep_labels, bovine_labels = resolve_all_labels(cfg)
st.session_state["species_labels"] = species_labels
st.session_state["sheep_labels"]   = sheep_labels
st.session_state["bovine_labels"]  = bovine_labels

# -------------------- Model loading (cached) --------------------
@st.cache_resource(show_spinner=False)
def load_models(species_path, sheep_path, bovine_path):
    try:
        species = keras.models.load_model(species_path)
        sheep   = keras.models.load_model(sheep_path)
        bovine  = keras.models.load_model(bovine_path)
    except Exception as e:
        st.error(f"Failed to load one or more models. Check paths in router_config.json.\n\n{e}")
        st.stop()
    return {"species": species, "sheep": sheep, "bovine": bovine}

models = load_models(cfg["species_model"], cfg["sheep_model"], cfg["bovine_model"])

# -------------------- Helpers --------------------
def prepare_image(img: Image.Image, img_size: int):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((img_size, img_size))
    return np.asarray(img).astype("float32") / 255.0

def topk(labels, probs, k=3):
    idxs = np.argsort(probs)[::-1][:min(k, len(labels))]
    return [(labels[i] if i < len(labels) else f"class_{i}", float(probs[i])) for i in idxs]

def reset_app():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

def render_species_info(sp_label:str):
    info = SPECIES_INFO.get(sp_label.lower())
    if info:
        with st.expander(f"About {info['title']}", expanded=True):
            st.markdown(info["summary"], unsafe_allow_html=True)
            if info.get("facts"):
                st.markdown("**Quick facts:**")
                for f in info["facts"]:
                    st.markdown(f"- {f}")

def render_low_conf_badge(conf: float):
    # Visual badge for confidence level
    if conf >= 0.80:
        st.markdown('<span class="badge badge-green">High confidence</span>', unsafe_allow_html=True)
    elif conf >= 0.50:
        st.markdown('<span class="badge badge-amber">Medium confidence</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-red">Low confidence — please verify manually</span>', unsafe_allow_html=True)

# -------------------- Sidebar (summary + reset) --------------------
with st.sidebar:
    st.subheader("Configuration")
    st.json({
        "models_dir": str(ART),
        "species_labels": species_labels,
        "sheep_labels_count": len(sheep_labels),
        "bovine_labels_count": len(bovine_labels),
    })
    st.divider()
    if st.button("New image / Reset", use_container_width=True):
        reset_app()

# -------------------- Two-column layout --------------------
left_col, right_col = st.columns([1, 1])

# ---- LEFT: Upload & preview ----
with left_col:
    uploaded = st.file_uploader(
        "Upload image (JPG/PNG/WebP). Tip: clear profile/side photo works best.",
        type=["jpg", "jpeg", "png", "webp"],
        key="uploader",
    )
    if uploaded:
        try:
            image = Image.open(uploaded)
            st.image(image, caption="Input", use_container_width=True)
        except Exception as e:
            st.error(f"Could not read image: {e}")
            image = None
    else:
        st.info("Upload a clear photo on the left to start.")

# ---- RIGHT: Results ----
with right_col:
    if uploaded:
        if image is None:
            st.stop()

        arr = prepare_image(image, cfg["img_size"])

        # Stage 1 — species
        try:
            sp_probs = models["species"].predict(np.expand_dims(arr, 0), verbose=0)[0]
        except Exception as e:
            st.error(f"Species model failed to run: {e}")
            st.stop()

        species_labels = st.session_state.get("species_labels", ["bovine", "sheep"])
        sp_idx   = int(np.argmax(sp_probs))
        sp_label = species_labels[sp_idx] if sp_idx < len(species_labels) else "unknown"
        sp_conf  = float(sp_probs[sp_idx])

        st.subheader("Stage 1 · Species")
        st.write(f"**{sp_label}**  |  confidence **{sp_conf:.2%}**")
        render_low_conf_badge(sp_conf)
        st.progress(min(1.0, sp_conf))

        # Suggest manual check if species conf < 50%
        if sp_conf < 0.50:
            st.warning("Model is not confident about the species (< 50%). Please verify manually.")

        # Species info panel
        render_species_info(sp_label)

        # Stage 2 — breed (routed head)
        head = "sheep" if sp_label.lower() == "sheep" else "bovine"
        labels = (st.session_state.get("sheep_labels", []) if head == "sheep"
                  else st.session_state.get("bovine_labels", []))

        if not labels:
            st.error(f"No {head} labels found. Ensure labels JSON or train folders exist.")
            st.stop()

        try:
            br_probs = models[head].predict(np.expand_dims(arr, 0), verbose=0)[0]
        except Exception as e:
            st.error(f"{head.title()} breed model failed to run: {e}")
            st.stop()

        br_idx   = int(np.argmax(br_probs))
        br_label = labels[br_idx] if br_idx < len(labels) else f"{head}_class_{br_idx}"
        br_conf  = float(br_probs[br_idx])

        st.subheader("Stage 2 · Breed")
        st.write(f"**{br_label}**  |  confidence **{br_conf:.2%}**")
        render_low_conf_badge(br_conf)
        st.progress(min(1.0, br_conf))

        # Suggest manual check if breed conf < 50%
        if br_conf < 0.50:
            st.error("Breed confidence is below 50%. Please cross-check manually before recording.")

        with st.expander("Top-3 candidates (breed head)"):
            for name, p in topk(labels, br_probs, k=3):
                st.write(f"- {name}: {p:.2%}")

# ---- After first result: ask for another upload (LEFT column) ----
if uploaded:
    with left_col:
        st.divider()
        st.markdown("**Would you like to analyze another image?**")
        next_upload = st.file_uploader(
            "Upload next image",
            type=["jpg", "jpeg", "png", "webp"],
            key="uploader_next",
        )
        if next_upload is not None:
            # Simple flow: replace primary upload with the new file by forcing a rerun
            # Save a tiny flag so the user sees their next image processed immediately
            st.session_state["_next_bytes"] = next_upload.getvalue()
            st.session_state["_next_name"] = next_upload.name
            st.toast("Loading next image…")
            st.rerun()

# If a next image was queued, process it immediately (simulate primary upload)
if "_next_bytes" in st.session_state:
    from io import BytesIO
    uploaded_bytes = st.session_state.pop("_next_bytes")
    uploaded_name  = st.session_state.pop("_next_name", "next.jpg")
    fake_file = BytesIO(uploaded_bytes)
    fake_image = Image.open(fake_file)
    # Set into session so user sees it in left column
    with left_col:
        st.image(fake_image, caption=f"Next Input: {uploaded_name}", use_container_width=True)
    # And run results again on right (same code as above, compact form)
    with right_col:
        arr = prepare_image(fake_image, cfg["img_size"])
        sp_probs = models["species"].predict(np.expand_dims(arr, 0), verbose=0)[0]
        sp_idx   = int(np.argmax(sp_probs))
        species_labels = st.session_state.get("species_labels", ["bovine", "sheep"])
        sp_label = species_labels[sp_idx] if sp_idx < len(species_labels) else "unknown"
        sp_conf  = float(sp_probs[sp_idx])

        st.subheader("Stage 1 · Species (Next)")
        st.write(f"**{sp_label}**  |  confidence **{sp_conf:.2%}**")
        render_low_conf_badge(sp_conf)
        st.progress(min(1.0, sp_conf))
        if sp_conf < 0.50:
            st.warning("Model is not confident about the species (< 50%). Please verify manually.")
        render_species_info(sp_label)

        head = "sheep" if sp_label.lower() == "sheep" else "bovine"
        labels = (st.session_state.get("sheep_labels", []) if head == "sheep"
                  else st.session_state.get("bovine_labels", []))
        if labels:
            br_probs = models[head].predict(np.expand_dims(arr, 0), verbose=0)[0]
            br_idx   = int(np.argmax(br_probs))
            br_label = labels[br_idx] if br_idx < len(labels) else f"{head}_class_{br_idx}"
            br_conf  = float(br_probs[br_idx])

            st.subheader("Stage 2 · Breed (Next)")
            st.write(f"**{br_label}**  |  confidence **{br_conf:.2%}**")
            render_low_conf_badge(br_conf)
            st.progress(min(1.0, br_conf))
            if br_conf < 0.50:
                st.error("Breed confidence is below 50%. Please cross-check manually before recording.")
            with st.expander("Top-3 candidates (breed head) — Next"):
                for name, p in topk(labels, br_probs, k=3):
                    st.write(f"- {name}: {p:.2%}")

# ---- Bottom reset button ----
st.divider()
if st.button("New image / Reset", type="secondary", use_container_width=True):
    reset_app()
