import streamlit as st
import re
from training_data import data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.markdown('<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.0/font/bootstrap-icons.css">', unsafe_allow_html=True)

def load_css(file_path):
    try:
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass
load_css("style.css")

def get_indonesian_stopwords():
    return {
        'yang', 'dan', 'di', 'ke', 'dari', 'dalam', 'untuk', 'pada', 'dengan', 
        'oleh', 'adalah', 'ini', 'itu', 'akan', 'telah', 'sudah', 'dapat',
        'atau', 'juga', 'tidak', 'ada', 'saya', 'kamu', 'dia', 'mereka',
        'kami', 'kita', 'anda', 'ia', 'nya', 'mu', 'ku', 'se', 'ter',
        'ber', 'per', 'an', 'kan', 'lah', 'pun', 'kah', 'tah', 'sangat',
        'lebih', 'paling', 'sekali', 'bisa', 'harus', 'ingin', 'mau',
        'seperti', 'karena', 'kalau', 'jika', 'bila', 'ketika', 'saat',
        'waktu', 'sebelum', 'sesudah', 'setelah', 'sambil', 'selama'
    }

def preprocess_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = text.lower().strip()
    if re.match(r'^https?://', text):
        parts = re.split(r'/|-|_', text)
        keywords = [word for word in parts if word.isalpha()]
        return ' '.join(keywords).strip()
    text = re.sub(r'http[s]?://\S+', '', text)
    return text.strip()

def preprocess_training_data(data):
    processed_data = []
    for text, label in data:
        processed_text = preprocess_text(text)
        if processed_text:  
            processed_data.append((processed_text, label))
    return processed_data

st.markdown('<div class="content">', unsafe_allow_html=True)
st.markdown('<div class="title-style">🧠 AI News Classifier</div>', unsafe_allow_html=True)

def train_model(processed_data):
    try:
        texts, labels = zip(*processed_data)
        vectorizer = CountVectorizer(min_df=2)
        X = vectorizer.fit_transform(texts)
        model = MultinomialNB(alpha=1.0)
        model.fit(X, labels)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error dalam melatih model: {e}")
        return None, None

processed_data = preprocess_training_data(data)
model, vectorizer = train_model(processed_data)

if model is None:
    st.stop()

st.info("Kategori yang tersedia: Politik, Olahraga, Ekonomi, Otomotif, Teknologi")

input_text = st.text_area("Masukkan teks untuk diklasifikasikan:", height=100)

category_colors = {
    "politik": "#FFD700", "olahraga": "#2E8B57", "ekonomi": "#4169E1",  
    "otomotif": "#DC143C", "teknologi": "#9370DB",
}

st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
if st.button("🔍 Prediksi Kategori", use_container_width=True):
        if not input_text.strip():
            st.warning("Harap masukkan teks terlebih dahulu.")
        else:
            try:
                processed_input = preprocess_text(input_text)
                if not processed_input.strip():
                    st.warning("Masukkan Teks Kembali")
                else:
                    input_vec = vectorizer.transform([processed_input])
                    prediction = model.predict(input_vec)[0]
                    proba = model.predict_proba(input_vec)[0]
                    confidence = max(proba) * 100
                    color = category_colors.get(prediction.lower(), "#999999")

                    st.markdown(
                        f'<div class="result" style="background-color:{color}; padding: 20px; border-radius: 10px; margin: 20px 0; text-align: center; color: white; font-size: 1.2em;">'
                        f'<strong>Kategori:</strong> {prediction.capitalize()}<br>'
                        f'<strong>Confidence:</strong> {confidence:.2f}%'
                        '</div>',
                        unsafe_allow_html=True
                    )
                    
                    with st.expander("Kata-kata Penting yang Terdeteksi"):
                        feature_names = vectorizer.get_feature_names_out()
                        input_features = input_vec.toarray()[0]
                        important_words = [(feature_names[i], score) for i, score in enumerate(input_features) if score > 0]
                        important_words.sort(key=lambda x: x[1], reverse=True)
                        
                        if important_words:
                            words_text = ", ".join([f"**{word}** ({int(count)})" for word, count in important_words[:10]])
                            st.markdown(f"Kata-kata yang mempengaruhi prediksi: {words_text}")
                        else:
                            st.info("Tidak ada kata penting yang terdeteksi dalam vocabulary model.")
                        
            except Exception:
                st.error("Pastikan teks mengandung kata-kata yang relevan dengan kategori yang tersedia.")

st.markdown("""
<div class="fixed-footer" style="text-align: center; padding: 20px;">
    <a href='https://www.instagram.com/alifarmadanii.lc.ma' target='_blank' style="margin: 0 10px;">
        <i class="bi bi-instagram"></i>
    </a>
    <a href='mailto:armadanialif@gmail.com' target='_blank' style="margin: 0 10px;">
        <i class="bi bi-envelope"></i>
    </a>
    <a href='https://www.linkedin.com/in/m-alif-armadani-lubis-5191a8366/' target='_blank' style="margin: 0 10px;">
        <i class="bi bi-linkedin"></i>
    </a>
    <p>Created by M. Alif Armadani Lubis</p>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
