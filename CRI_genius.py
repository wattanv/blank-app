# CRI_genius.py

# =================================================================
# 1. Import ไลบรารีที่จำเป็นทั้งหมด
# =================================================================
import streamlit as st
from datetime import datetime
from PIL import Image
from collections import Counter
import numpy as np
import cv2
import io
import os
from roboflow import Roboflow

# =================================================================
# 2. ตั้งค่าหน้าเว็บ, UI Styles, และโหลดโมเดล
# =================================================================
st.set_page_config(page_title="CRI Genius", layout="wide")

st.markdown(
    """
    <div style="
        background: linear-gradient(to right, #6FC3FF, #2176FF);
        border-radius: 12px;
        padding: 10px 0;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        color: white;
        margin-bottom: 20px;
    ">
        CRI Genius
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("""
<style>
.inline-label {
    display: flex; align-items: center; gap: 8px; font-weight: bold;
    font-size: 18px; color: black; margin-bottom: 5px;
}
.inline-label img { width: 24px; height: 24px; }
.stTextInput input, .stDateInput input, .stSelectbox > div > div {
    background-color: #e6f2ff !important; border-radius: 8px !important; color: black !important;
}
.stTextInput input:disabled {
    background-color: #d0e0f0 !important; font-weight: bold; text-align: center; color: black !important;
}
div.stButton > button {
    background: linear-gradient(to right, #6FC3FF, #2176FF); color: black;
    font-weight: bold; border-radius: 25px; padding: 10px 40px; font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """โหลดโมเดลจาก Roboflow และคืนค่าเป็น tuple"""
    try:
        DETECTOR_API_KEY = "FIv4Ev7vj8vn5EGPeTpY"
        CLASSIFIER_API_KEY = "FIv4Ev7vj8vn5EGPeTpY"
        
        detector_model = Roboflow(api_key=DETECTOR_API_KEY).workspace("wattanathornch").project("crystal_quality_detection").version(9).model
        classifier_model = Roboflow(api_key=CLASSIFIER_API_KEY).workspace("wattanathornch").project("crystal_quality").version(1).model
        return detector_model, classifier_model
    except Exception as e:
        st.error(f"❌ ไม่สามารถเชื่อมต่อ AI ได้: {e}")
        return None, None

detector_model, classifier_model = load_models()

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = {}

# =================================================================
# 3. ส่วน UI และ Logic การทำงานหลัก
# =================================================================
col_main_left, col_main_right = st.columns([1, 1.2])

with col_main_left:
    st.markdown("""<div class="inline-label"><img src="https://img.icons8.com/ios-filled/50/2176FF/calendar--v1.png"/><span>วันที่วิเคราะห์</span></div>""", unsafe_allow_html=True)
    st.date_input("", datetime.today(), key="date_input", label_visibility="collapsed")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div class="inline-label"><img src="https://img.icons8.com/ios-filled/50/2176FF/edit--v1.png"/><span>Stike</span></div>""", unsafe_allow_html=True)
        st.text_input("", key="stike", label_visibility="collapsed")
    with col2:
        st.markdown("""<div class="inline-label"><img src="https://img.icons8.com/ios-filled/50/2176FF/bookmark-ribbon--v1.png"/><span>class</span></div>""", unsafe_allow_html=True)
        st.selectbox("", ["A", "B", "C", "D"], key="class_selection", label_visibility="collapsed")
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""<div class="inline-label"><img src="https://img.icons8.com/ios-filled/50/2176FF/user.png"/><span>ผู้วิเคราะห์</span></div>""", unsafe_allow_html=True)
        st.text_input("", key="analyst", label_visibility="collapsed")
    with col4:
        st.markdown("""<div class="inline-label"><img src="https://img.icons8.com/ios-filled/50/2176FF/worker-male.png"/><span>ช่างเคี่ยว</span></div>""", unsafe_allow_html=True)
        st.text_input("", key="operator", label_visibility="collapsed")

    st.markdown("### 📷 เลือกรูปที่นี่")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    process_button = st.button("ประมวลผล")

if process_button:
    if uploaded_file is None:
        st.warning("⚠️ กรุณาอัปโหลดรูปภาพก่อนกดประมวลผล")
    elif not detector_model or not classifier_model:
        st.error("❌ โมเดล AI ยังไม่พร้อมใช้งาน ไม่สามารถประมวลผลได้")
    else:
        try:
            with st.spinner("🧠 กำลังประมวลผลด้วย AI... กรุณารอสักครู่"):
                image_pil = Image.open(uploaded_file).convert("RGB")
                
                # 1. Detection
                temp_path = "temp_streamlit.jpg"
                image_pil.save(temp_path)
                detections = detector_model.predict(temp_path, confidence=5, overlap=30).json().get('predictions', [])
                os.remove(temp_path)

                # 2. Classification
                full_results = []
                debug_errors = []
                for i, detection in enumerate(detections):
                    try:
                        x = float(detection.get('x', 0))
                        y = float(detection.get('y', 0))
                        w = float(detection.get('width', 0))
                        h = float(detection.get('height', 0))

                        if w == 0 or h == 0: continue

                        x1, y1, x2, y2 = int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)
                        cropped_img = image_pil.crop((x1, y1, x2, y2))
                        
                        predicted_class = None
                        try:
                            raw_result = classifier_model.model.predict(cropped_img).json()
                            if 'top' in raw_result:
                                predicted_class = raw_result.get('top')
                        except Exception as class_e:
                            pass
                        full_results.append({"class": predicted_class})

                    except (TypeError, ValueError) as loop_e:
                        error_info = f"Error in loop item #{i+1}: {loop_e} | Data: {detection}"
                        debug_errors.append(error_info)
                        continue
                
                # 3. คำนวณสรุปผล
                classified_classes = [res['class'] for res in full_results if res.get('class')]
                total_grains = len(classified_classes)
                grade_counts = Counter(classified_classes)

                N3 = grade_counts.get('class 3', 0)
                N2 = grade_counts.get('class 2', 0)
                N1 = grade_counts.get('class 1', 0)
                N0 = grade_counts.get('class 0', 0)
                
                cri_score = 0.0
                if total_grains > 0:
                     numerator = (3 * N3) + (2 * N2) + (1 * N1)
                     cri_score = (numerator / total_grains) * 100
                
                # 4. บันทึกผลลัพธ์
                st.session_state.analysis_results = {
                    "N3": N3, "N2": N2, "N1": N1, "N0": N0,
                    "total_grains": total_grains, "cri": cri_score,
                    "processed_image": image_pil
                }
                st.session_state.debug_info = {"loop_errors": debug_errors}

            st.success("✅ ประมวลผลสำเร็จ!")

        except Exception as e:
            st.error("😭 เกิดข้อผิดพลาดร้ายแรงระหว่างการประมวลผล!")
            st.exception(e)

with col_main_right:
    results = st.session_state.analysis_results

    if results:
        st.image(results["processed_image"], caption="ภาพที่วิเคราะห์", use_column_width=True)
    elif uploaded_file:
        st.image(uploaded_file, caption="ภาพที่เลือก (รอการประมวลผล)", use_column_width=True)
    else:
        placeholder_image = np.full((400, 600, 3), 240, dtype=np.uint8)
        cv2.putText(placeholder_image, "Upload an image to start", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2)
        st.image(placeholder_image, caption="ผลการวิเคราะห์จะแสดงที่นี่")

    st.markdown("### ผลการประเมินคะแนนเม็ดน้ำตาล")

    N3 = results.get("N3", 0) if results else 0
    N2 = results.get("N2", 0) if results else 0
    N1 = results.get("N1", 0) if results else 0
    N0 = results.get("N0", 0) if results else 0
    total_grains = results.get("total_grains", 0) if results else 0
    cri = results.get("cri", 0.0) if results else 0
    
    p3 = (N3 / total_grains * 100) if total_grains > 0 else 0
    p2 = (N2 / total_grains * 100) if total_grains > 0 else 0
    p1 = (N1 / total_grains * 100) if total_grains > 0 else 0
    p0 = (N0 / total_grains * 100) if total_grains > 0 else 0
    
    data_to_display = [{"class": 3, "count": N3, "percent": p3}, {"class": 2, "count": N2, "percent": p2},
                       {"class": 1, "count": N1, "percent": p1}, {"class": 0, "count": N0, "percent": p0}]

    for item in data_to_display:
        g, c, m, p, pct = st.columns([1, 1, 0.5, 1, 0.5])
        with g: st.markdown(f"คะแนน {item['class']}")
        with c: st.text_input("", str(item['count']), disabled=True, key=f"d_c_{item['class']}", label_visibility="collapsed")
        with m: st.markdown("เม็ด")
        with p: st.text_input("", f"{item['percent']:.2f}", disabled=True, key=f"d_p_{item['class']}", label_visibility="collapsed")
        with pct: st.markdown("%")

    t1, t2, t3, t4, t5 = st.columns([1, 1, 0.5, 1, 0.5])
    with t1: st.markdown("**Total**")
    with t2: st.markdown(f"**{total_grains}**")
    with t3: st.markdown("เม็ด")
    with t4: st.markdown(f"**{p3+p2+p1+p0:.2f}**")
    with t5: st.markdown("%")
    
    st.markdown("### **%CRI**")
    st.text_input("", f"{cri:.2f} %", disabled=True, key="d_cri", label_visibility="collapsed")

    if st.button("บันทึกข้อมูล"):
        if st.session_state.analysis_results:
            st.success("✅ บันทึกข้อมูลเรียบร้อยแล้ว (Logic การบันทึกยังไม่ถูกเพิ่ม)")
        else:
            st.error("❌ ไม่มีข้อมูลให้บันทึก กรุณากดประมวลผลก่อน")


# =================================================================
# ส่วนแสดงผล DEBUG 
# =================================================================
if st.session_state.debug_info:
    st.markdown("---")
    with st.expander("🔍 **ผลการตรวจสอบเบื้องหลัง (สำคัญมาก)**"):
        debug = st.session_state.debug_info
        
        loop_errors = debug.get("loop_errors", [])
        if loop_errors:
            st.error("พบข้อผิดพลาดระหว่างการประมวลผลผลึกแต่ละชิ้น:")
            for error_text in loop_errors:
                st.code(error_text, language="text")
            st.warning("ค่าต่างๆ ที่แสดงผลอาจไม่ถูกต้อง 100% เนื่องจากข้อมูลบางส่วนถูกข้ามไป")
        else:
            st.success("ไม่พบข้อผิดพลาดร้ายแรงระหว่างการประมวลผลผลึกแต่ละชิ้น")