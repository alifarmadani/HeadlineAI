import streamlit as st
from training_data import data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.0/font/bootstrap-icons.css">', unsafe_allow_html=True)

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css") 

st.markdown('<div class="content">', unsafe_allow_html=True)
st.markdown('<div class="title-style">🧠 AI News Classifier</div>', unsafe_allow_html=True)

# Validasi data
if not data:
    st.error("Data pelatihan tidak valid!")
    st.stop()

# Model training
try:
    texts, labels = zip(*data)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    model = MultinomialNB()
    model.fit(X, labels)
except Exception as e:
    st.error(f"Error dalam melatih model: {e}")
    st.stop()

st.info("Kategori yang tersedia: Politik, Olahraga, Ekonomi, Otomotif, Teknologi")

# Contoh teks
with st.expander("Contoh Teks untuk Dicoba"):
    st.write("- **Politik**: 'Presiden mengumumkan kebijakan baru tentang pajak.'")
    st.write("- **Olahraga**: 'Tim nasional memenangkan pertandingan dengan skor 3-1.'")

# Input
input_text = st.text_area("Masukkan teks untuk diklasifikasikan:", height=100)

category_colors = {
    "politik": "#FFD700",  
    "olahraga": "#2E8B57", 
    "ekonomi": "#4169E1",  
    "otomotif": "#DC143C", 
    "teknologi": "#9370DB",
}

# Prediksi
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔍 Prediksi Kategori", use_container_width=True):
        if not input_text.strip():
            st.warning("Harap masukkan teks terlebih dahulu.")
        else:
            try:
                input_vec = vectorizer.transform([input_text])
                prediction = model.predict(input_vec)[0]
                proba = model.predict_proba(input_vec)[0]
                confidence = max(proba) * 100
                color = category_colors.get(prediction.lower(), "#999999")

                # Tampilkan hasil
                st.markdown(
                    f'<div class="result" style="background-color:{color};">'
                    f'<strong>Kategori:</strong> {prediction.capitalize()}<br>'
                    f'<strong>Confidence:</strong> {confidence:.2f}%'
                    '</div>',
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Error dalam prediksi: {e}")
# Footer
st.markdown("---")
st.markdown("""
        <div class="fixed-footer">
            <a href='https://www.instagram.com/alifarmadanii.lc.ma' target='_blank'>
            <i class="bi bi-instagram"></i>
        </a>
            <a href='mailto:armadanialif@gmail.com' target='_blank'>
            <i class="bi bi-envelope"></i>
        </a>
            <a href='https://www.linkedin.com/in/m-alif-armadani-lubis-5191a8366/' target='_blank'>
            <i class="bi bi-linkedin"></i>
        </a>
    <p>Created by M. Alif Armadani Lubis</p>
</div>
""", unsafe_allow_html=True)
