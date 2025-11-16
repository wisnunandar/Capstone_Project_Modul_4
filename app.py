import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import yaml
import re
from io import BytesIO
from collections import Counter
import supervision as sv
import pandas as pd
import zipfile
import os
import tempfile
import time

# ------------------------
# Config
# ------------------------
MODEL_PATH = r"C:\Caps_modul4\best.pt"       # <-- sesuaikan (utama)
DATA_YAML_PATH = r"C:\Caps_modul4\data.yaml" # <-- sesuaikan
DEFAULT_CONF_THRESHOLD = 0.3

# ------------------------
# UTIL: load labels & calories
# ------------------------
@st.cache_data
def load_yaml_labels(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    names = data.get("names", []) #get class name lists

    # ekstrak angka kalori dari nama (misal "Nasi -129 kal per 100gr-")
    label_calories = {}
    for name in names:
        m = re.search(r"(\d+)[ ]?kal", name) #ekstrak kalori 
        if m:
            label_calories[name] = int(m.group(1))
        else:
            label_calories[name] = 0
    return names, label_calories

label_names, label_calories = load_yaml_labels(DATA_YAML_PATH)
id2label = {i: name for i, name in enumerate(label_names)}
id2calorie = {i: label_calories[name] for i, name in enumerate(label_names)}

# ------------------------
# UTIL: load model 
# ------------------------
@st.cache_resource #save to cache, avoiding model reload
def load_model(path):
    return YOLO(path)

model = load_model(MODEL_PATH)

# ------------------------
# SESSION STATE init
# ------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dict per processed image

# ------------------------
# APP LAYOUT
# ------------------------
st.set_page_config(page_title="Calories Detector", layout="wide")
st.title("🍽️ Calories Detector & Calculator")

st.markdown("""
Anda Bisa Melakukan:
- Upload banyak gambar sekaligus dan unduh laporan hasil deteksi.  
- Kontrol confidence threshold .  
- Mengatur Kalori/serving.
- Mengambil gambar makanan menggunakan kamera
""")

# Sidebar controls for user
with st.sidebar:
    st.header("Pengaturan Deteksi")
    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, float(DEFAULT_CONF_THRESHOLD), 0.01)
    show_annotations = st.checkbox("Tampilkan anotasi pada gambar", value=True)
    resize_to = st.selectbox("Resize image", [320, 416, 512, 640, 736], index=3)
    st.markdown("---")
    st.header("Pengaturan Kalori / Serving")
    
    global_serving_multiplier = st.number_input("Global serving multiplier", min_value=0.01, value=1.0, step=0.1)
    st.markdown("---")
    st.write("Note : Riwayat akan tersimpan selama sesi browser ini aktif.")

# ------------------------
# INPUT: camera option + batch upload / single file
# ------------------------
st.subheader("Ambil gambar atau upload gambar")
use_camera = st.checkbox("Gunakan kamera untuk ambil gambar (kamera akan muncul di bawah)")

camera_image = None
if use_camera:
    # tampilkan kamera dan dapatkan hasil capture
    camera_image = st.camera_input("Ambil foto makanan")
    # tombol proses khusus kamera
    if camera_image is not None:
        st.info("Foto diambil. Scroll ke bawah untuk memproses atau tekan tombol 'Proses Gambar (Run Detection)'.")


upload_mode = st.radio("", ("Upload Images (multiple)", "Upload Single Image"))
if upload_mode == "Upload Images (multiple)":
    uploaded_files = st.file_uploader("Pilih gambar (jpg/png). Dapat pilih banyak.", accept_multiple_files=True, type=["jpg","jpeg","png"])
else:
    uploaded_files = st.file_uploader("Pilih satu gambar", accept_multiple_files=False, type=["jpg","jpeg","png"])
    if uploaded_files:
        uploaded_files = [uploaded_files]

process_btn = st.button("Proses Gambar (Run Detection)")

# ------------------------
# annotate/save image to bytes
# ------------------------
def annotate_image_and_bytes(image_np, detections, labels, show_ann=True): #anotasi gambar 
    ann = image_np.copy()
    if show_ann:
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        ann = box_annotator.annotate(scene=ann, detections=detections)
        ann = label_annotator.annotate(scene=ann, detections=detections, labels=labels)
    pil = Image.fromarray(ann)
    buf = BytesIO()
    pil.save(buf, format="PNG") #simpan gambar dalam memori bytesIO
    buf.seek(0)
    return buf

# ------------------------
# single image processing 
# ------------------------
def process_and_record(pil_image, image_name, results_rows_list=None, zipf_obj=None, show_preview=True):
    """
    pil_image: PIL.Image RGB
    image_name: string for naming
    results_rows_list: list to append result rows (optional)
    zipf_obj: ZipFile to write annotated image into (optional)
    returns: dict summary (counts,total_calories,annotated_bytes)
    """

    image = pil_image.convert("RGB")
    image_np = np.array(image)
    orig_h, orig_w = image_np.shape[:2]
    image_resized = cv2.resize(image_np, (resize_to, resize_to))
    image_resized = cv2.convertScaleAbs(image_resized, alpha=1.0, beta=0)
    
    
    #YOLO prediction into sv format

    results = model.predict(source=image_resized, conf=conf_threshold, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results).with_nms()

    #get class and confidence score

    class_ids = detections.class_id.tolist() if len(detections.class_id) else []
    confidences = detections.confidence.tolist() if len(detections.confidence) else []

    labels = [f"{id2label[cid]} {conf:.2f}" for cid, conf in zip(class_ids, confidences)]
    #calculating calories
    counts = Counter(class_ids)
    
    
    per_class_multiplier = {cid: global_serving_multiplier for cid in counts.keys()}

    per_class_calories = {}
    for cid, cnt in counts.items():
        base_cal = id2calorie.get(cid, 0)
        per_class_calories[cid] = int(base_cal * per_class_multiplier[cid] * cnt)
    total_calories = sum(per_class_calories.values()) #total calories

    annotated_bytes = annotate_image_and_bytes(image_resized, detections, labels, show_ann=show_annotations)
    if zipf_obj is not None:
        zipf_obj.writestr(f"{os.path.splitext(image_name)[0]}_annotated.png", annotated_bytes.getvalue())

    # prepare rows for CSV if requested
    if results_rows_list is not None:
        if counts:
            for cid, cnt in counts.items():
                row = {
                    "image_name": image_name,
                    "class_id": int(cid),
                    "label": id2label[cid],
                    "count": int(cnt),
                    "confidences": ";".join([f"{c:.3f}" for c, cl in zip(confidences, class_ids) if cl == cid]),
                    "base_cal_per_item": id2calorie.get(cid, 0),
                    "serving_multiplier": per_class_multiplier[cid],
                    "calories_for_this_class": int(id2calorie.get(cid, 0) * per_class_multiplier[cid] * cnt),
                    "total_calories_image": int(total_calories),
                }
                results_rows_list.append(row)
        else:
            results_rows_list.append({
                "image_name": image_name,
                "class_id": None,
                "label": None,
                "count": 0,
                "confidences": "",
                "base_cal_per_item": 0,
                "serving_multiplier": global_serving_multiplier,
                "calories_for_this_class": 0,
                "total_calories_image": 0
            })

    # save to session history
    st.session_state.history.append({
        "timestamp": time.time(),
        "image_name": image_name,
        "total_calories": int(total_calories),
        "counts": {id2label[k]: v for k, v in counts.items()},
        "confidences": {id2label[cid]: [float(f"{c:.3f}") for c, cl in zip(confidences, class_ids) if cl == cid] for cid in counts.keys()}
    })

    if show_preview:
        st.write(f"**{image_name}** — Total Kalori: {int(total_calories)}")
        st.image(annotated_bytes, use_column_width=True)

    return {"counts": counts, "total_calories": int(total_calories), "annotated_bytes": annotated_bytes}

# ------------------------
# PROCESSING: handle camera first OR uploaded files 
# ------------------------
# Camera capture processing 
if camera_image is not None and st.button("Proses Foto Kamera"):
    try:
        cam_bytes = camera_image.getvalue()
        cam_pil = Image.open(BytesIO(cam_bytes)).convert("RGB")
        # process single camera image and show immediately
        _ = process_and_record(cam_pil, image_name=f"camera_{int(time.time())}.png", results_rows_list=None, zipf_obj=None, show_preview=True)
        st.success("Selesai memproses foto dari kamera.")
    except Exception as e:
        st.error(f"Gagal memproses foto kamera: {e}")

# Batch / upload processing (saat tombol utama ditekan)
if process_btn:
    if not uploaded_files:
        st.warning("Belum ada file yang diupload.")
    else:
        progress = st.progress(0)
        total = len(uploaded_files)
        results_rows = []  # will collect rows for CSV export
        temp_dir = tempfile.mkdtemp()
        zip_fname = os.path.join(temp_dir, f"annotated_{int(time.time())}.zip")
        zipf = zipfile.ZipFile(zip_fname, "w", zipfile.ZIP_DEFLATED)

        for idx, up in enumerate(uploaded_files):
            try:
                pil_img = Image.open(up).convert("RGB")
            except Exception as e:
                st.error(f"Gagal membuka {up.name}: {e}")
                continue

            process_and_record(pil_img, image_name=up.name, results_rows_list=results_rows, zipf_obj=zipf, show_preview=(idx < 6))
            progress.progress((idx + 1) / total)

        zipf.close()

        # create DataFrame for CSV export
        df = pd.DataFrame(results_rows)
        if df.empty:
            st.info("Tidak ada deteksi pada semua gambar.")
        else:
            st.success(f"Selesai memproses {len(uploaded_files)} gambar. {len(df)} baris hasil dibuat.")
            st.dataframe(df.head(50))

            # CSV download
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.download_button("Unduh CSV hasil", data=csv_bytes, file_name="detection_results.csv", mime="text/csv")

            # ZIP annotated images download
            with open(zip_fname, "rb") as f:
                zip_bytes = f.read()
            st.download_button("Unduh ZIP gambar beranotasi", data=zip_bytes, file_name="annotated_images.zip", mime="application/zip")

# ------------------------
# HISTORY: tampil dan unduh
# ------------------------
st.markdown("---")
st.subheader("Riwayat Deteksi (sesi saat ini)")
if st.session_state.history:
    hist_df = pd.DataFrame([
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(h["timestamp"])),
            "image_name": h["image_name"],
            "total_calories": h["total_calories"],
            "counts": str(h["counts"])
        }
        for h in st.session_state.history
    ])
    st.dataframe(hist_df)
    st.download_button("Unduh CSV Riwayat", data=hist_df.to_csv(index=False).encode("utf-8"), file_name="detection_history.csv")
else:
    st.write("Belum ada riwayat (belum menjalankan proses).")

# ------------------------
# END
# ------------------------
st.markdown("---")
st.caption("Aplikasi ini dibuat untuk demonstrasi fitur improvised: batch processing, kontrol confidence, dan serving-size adjustments. Sesuaikan jalur model & yaml di bagian atas file.")
