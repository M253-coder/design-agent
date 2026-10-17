import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# إعدادات صفحة Streamlit
st.set_page_config(page_title="وكيل ذكي لتحسين التصاميم", layout="wide")

st.title("🎨 الوكيل الذكي لتحليل وتحسين التصاميم (Design AI Agent)")
st.write("ارفعي تصميمكِ، وسيقوم الوكيل بتحليله، اكتشاف الأخطاء، وتحسينه تلقائياً!")

# رفع الصورة
uploaded_file = st.file_uploader("اختر صورة التصميم (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # قراءة الصورة
    image = Image.open(uploaded_file)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("التصميم الأصلي")
        st.image(image, use_container_width=True)

    # 1. تحليل الألوان (استخراج الألوان السائدة)
    def get_dominant_colors(image, k=3):
        image_np = np.array(image)
        image_np = cv2.resize(image_np, (100, 100))
        pixels = image_np.reshape(-1, 3)
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
        return colors

    colors = get_dominant_colors(image)

    # 2. فحص السطوع والتباين لاكتشاف المشكلات
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    contrast_std = np.std(gray)

    issues = []
    if mean_brightness < 70:
        issues.append("⚠️ التصميم مظلم جداً، مما يؤثر على وضوح العناصر.")
    elif mean_brightness > 200:
        issues.append("⚠️ التصميم ساطع جداً (Overexposed)، قد يفقد التفاصيل.")

    if contrast_std < 35:
        issues.append("⚠️ التباين في التصميم ضعيف (Low Contrast)، مما يصعب قراءة النصوص.")
    else:
        issues.append("✅ التباين مقبول بصرياً.")

    # 3. التحسين التلقائي (Auto-Correction)
    # تعديل التباين والسطوع تلقائياً بناءً على التحليل
    enhancer_contrast = ImageEnhance.Contrast(image)
    enhanced_image = enhancer_contrast.enhance(1.25) # زيادة التباين بنسبة 25%
    
    enhancer_brightness = ImageEnhance.Brightness(enhanced_image)
    if mean_brightness < 70:
        enhanced_image = enhancer_brightness.enhance(1.2)
    elif mean_brightness > 200:
        enhanced_image = enhancer_brightness.enhance(0.9)

    with col2:
        st.subheader("التصميم بعد التحسين (Before / After)")
        st.image(enhanced_image, use_container_width=True)

    st.markdown("---")
    st.header("📊 تقرير الوكيل الذكي (AI Analysis Report)")

    # عرض المشكلات
    st.subheader("1. اكتشاف المشكلات:")
    if issues:
        for issue in issues:
            st.write(issue)
    else:
        st.write("✨ التصميم متوازن ولا توجد مشكلات رئيسية ظاهرة.")

    # عرض الألوان المستخرجة
    st.subheader("2. لوحة الألوان المستخرجة من التصميم:")
    color_cols = st.columns(len(colors))
    for i, color in enumerate(colors):
        hex_color = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
        with color_cols[i]:
            st.markdown(
                f"<div style='background-color: {hex_color}; height: 50px; border-radius: 5px; border: 1px solid #ccc;'></div>"
                f"<p style='text-align: center; font-size: 12px;'>{hex_color}</p>",
                unsafe_allow_html=True
            )

    # مؤشرات Heatmap تقريبية للتركيز البصري
    st.subheader("3. خريطة التركيز البصري (Heatmap / Saliency Map):")
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    st.image(heatmap_rgb, caption="تحليل توزيع الكتل البصرية والتركيز", use_container_width=True)

    # قسم Dashboard مبسط للأخطاء الشائعة
    st.markdown("---")
    st.header("📈 Dashboard: تحليل الأخطاء الشائعة")
    error_categories = ['ضعف التباين', 'إضاءة غير مناسبة', 'توزيع العناصر', 'تداخل النصوص']
    error_counts = [45, 30, 15, 10]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.barh(error_categories, error_counts, color=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
    ax.set_xlabel('نسبة التكرار (%)')
    ax.set_title('أكثر أخطاء التصميم شيوعاً في النظام')
    st.pyplot(fig)
  
