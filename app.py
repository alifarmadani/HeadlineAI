import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css") 

st.markdown('<div class="content">', unsafe_allow_html=True)
st.markdown('<div class="title-style">🧠 AI News Classifier</div>', unsafe_allow_html=True)
# Training data
data = [
    # Politik
    ("Pemerintah menetapkan kebijakan baru tentang pajak", "politik"),
    ("Menteri keuangan umumkan anggaran negara", "politik"),
    ("Pemilu presiden akan digelar tahun depan", "politik"),
    ("Pemerintah bahas subsidi BBM dalam sidang kabinet", "politik"),
    ("DPR mengesahkan undang-undang baru", "politik"),
    ("Presiden menghadiri konferensi tingkat tinggi ASEAN", "politik"),
    ("Menteri luar negeri bertemu dengan duta besar Amerika", "politik"),
    ("Kebijakan luar negeri Indonesia semakin proaktif", "politik"),
    ("Partai politik mulai kampanye menjelang pemilu", "politik"),
    ("Isu korupsi kembali mencuat di parlemen", "politik"),
    ("Pemerintah menetapkan kebijakan baru tentang pajak", "politik"),
    ("Menteri keuangan umumkan anggaran negara", "politik"),
    ("Pemilu presiden akan digelar tahun depan", "politik"),
    ("Pemerintah bahas subsidi BBM dalam sidang kabinet", "politik"),
    ("DPR mengesahkan undang-undang baru", "politik"),
    ("Presiden menghadiri konferensi tingkat tinggi ASEAN", "politik"),
    ("Menteri luar negeri bertemu dengan duta besar Amerika", "politik"),
    ("Kebijakan luar negeri Indonesia semakin proaktif", "politik"),
    ("Partai politik mulai kampanye menjelang pemilu", "politik"),
    ("Isu korupsi kembali mencuat di parlemen", "politik"),
    ("KPU menetapkan daftar pemilih tetap untuk pemilu", "politik"),
    ("Polemik revisi UU KPK kembali mencuat", "politik"),    
    # Olahraga
    ("Barcelona menang 3-0 melawan Real Madrid", "olahraga"),
    ("Benzema mencetak hat-trick untuk Real Madrid", "olahraga"),
    ("Liverpool juara liga champions musim ini", "olahraga"),
    ("Timnas Indonesia lolos ke final AFF", "olahraga"),
    ("Cristiano Ronaldo pindah ke klub Arab Saudi", "olahraga"),
    ("Pemain muda Indonesia tampil impresif di SEA Games", "olahraga"),
    ("Turnamen bulu tangkis All England berlangsung seru", "olahraga"),
    ("Piala Dunia U-20 dibatalkan di Indonesia", "olahraga"),
    ("Persib Bandung menang dramatis lewat adu penalti", "olahraga"),
    ("Atlet renang Indonesia pecahkan rekor nasional", "olahraga"),
    ("Barcelona menang 3-0 melawan Real Madrid", "olahraga"),
    ("Benzema mencetak hat-trick untuk Real Madrid", "olahraga"),
    ("Liverpool juara liga champions musim ini", "olahraga"),
    ("Timnas Indonesia lolos ke final AFF", "olahraga"),
    ("Cristiano Ronaldo pindah ke klub Arab Saudi", "olahraga"),
    ("Pemain muda Indonesia tampil impresif di SEA Games", "olahraga"),
    ("Turnamen bulu tangkis All England berlangsung seru", "olahraga"),
    ("Piala Dunia U-20 dibatalkan di Indonesia", "olahraga"),
    ("Persib Bandung menang dramatis lewat adu penalti", "olahraga"),
    ("Atlet renang Indonesia pecahkan rekor nasional", "olahraga"),
    ("Pelatih baru timnas Indonesia resmi diumumkan", "olahraga"),
    ("Pertandingan tinju dunia berlangsung di Las Vegas", "olahraga"),
    ("Tim voli putri Indonesia raih perak di ASEAN Games", "olahraga"),
    ("Messi cetak gol kemenangan PSG", "olahraga"),
    ("Kejuaraan dunia MotoGP digelar di Mandalika", "olahraga"), 
    # Ekonomi
    ("Harga minyak dunia turun drastis hari ini", "ekonomi"),
    ("IHSG naik karena investor asing masuk", "ekonomi"),
    ("Kurs rupiah melemah terhadap dolar AS", "ekonomi"),
    ("Investasi asing meningkat di sektor teknologi", "ekonomi"),
    ("Pertumbuhan ekonomi kuartal kedua mencapai 5%", "ekonomi"),
    ("BPS mencatat inflasi turun pada bulan Mei", "ekonomi"),
    ("Bank Indonesia menaikkan suku bunga acuan", "ekonomi"),
    ("Nilai ekspor Indonesia meningkat tajam", "ekonomi"),
    ("Indeks harga konsumen stabil selama triwulan ini", "ekonomi"),
    ("Pasar saham Asia alami koreksi signifikan", "ekonomi"),
        ("Harga minyak dunia turun drastis hari ini", "ekonomi"),
    ("IHSG naik karena investor asing masuk", "ekonomi"),
    ("Kurs rupiah melemah terhadap dolar AS", "ekonomi"),
    ("Investasi asing meningkat di sektor teknologi", "ekonomi"),
    ("Pertumbuhan ekonomi kuartal kedua mencapai 5%", "ekonomi"),
    ("BPS mencatat inflasi turun pada bulan Mei", "ekonomi"),
    ("Bank Indonesia menaikkan suku bunga acuan", "ekonomi"),
    ("Nilai ekspor Indonesia meningkat tajam", "ekonomi"),
    ("Indeks harga konsumen stabil selama triwulan ini", "ekonomi"),
    ("Pasar saham Asia alami koreksi signifikan", "ekonomi"),
    ("Pemerintah keluarkan stimulus ekonomi baru", "ekonomi"),
    ("Perusahaan rintisan teknologi raih pendanaan besar", "ekonomi"),
    ("Nilai tukar rupiah menguat di akhir perdagangan", "ekonomi"),
    #otomotif
    ("Toyota meluncurkan mobil listrik terbaru di Indonesia", "otomotif"),
    ("Honda resmi rilis motor listrik untuk pasar Asia", "otomotif"),
    ("Penjualan mobil SUV meningkat pesat tahun ini", "otomotif"),
    ("Tesla buka pabrik baru di India untuk produksi mobil listrik", "otomotif"),
    ("Pameran otomotif GIIAS 2025 dipadati pengunjung", "otomotif"),
    ("BMW umumkan strategi elektrifikasi seluruh lini kendaraan", "otomotif"),
    ("Kendaraan hybrid semakin diminati di pasar Eropa", "otomotif"),
    ("Industri otomotif nasional tumbuh 7 persen", "otomotif"),
    ("Yamaha rilis motor sport edisi terbatas", "otomotif"),
    ("Pemerintah akan beri insentif untuk pembelian mobil listrik", "otomotif"),
    ("Suzuki luncurkan mobil keluarga dengan fitur keselamatan terbaru", "otomotif"),
    ("Kawasaki siapkan motor off-road terbaru untuk pasar Indonesia", "otomotif"),
    ("Mobil listrik Wuling Air EV laris manis di pasar lokal", "otomotif"),
    ("Ford kembangkan teknologi mobil swakemudi generasi baru", "otomotif"),
    ("Ban ramah lingkungan mulai diproduksi massal", "otomotif"),
    ("Mitsubishi kenalkan konsep mobil SUV masa depan", "otomotif"),
    ("Kemenperin dorong produksi mobil listrik lokal", "otomotif"),
    ("Harga mobil bekas turun menjelang lebaran", "otomotif"),
    ("Motor listrik lokal Esemka siap dipasarkan", "otomotif"), 
    #teknologi
    ("Google memperkenalkan fitur AI terbaru di Google Search", "teknologi"),
    ("Startup AI lokal raih pendanaan sebesar 10 juta dolar", "teknologi"),
    ("Apple rilis iPhone 16 dengan fitur kamera super canggih", "teknologi"),
    ("Indonesia akan bangun pusat data nasional tahun ini", "teknologi"),
    ("Microsoft umumkan pembaruan besar untuk Windows 12", "teknologi"),
    ("Facebook ubah nama menjadi Meta dan fokus ke metaverse", "teknologi"),
    ("Perusahaan rintisan bidang edutech semakin berkembang", "teknologi"),
    ("Pemerintah dorong transformasi digital di sektor UMKM", "teknologi"),
    ("Samsung luncurkan layar lipat generasi ketiga", "teknologi"),
    ("Kaspersky deteksi serangan siber besar-besaran ke Asia Tenggara", "teknologi"),
    ("ChatGPT kini bisa digunakan dalam lebih dari 30 bahasa", "teknologi"),
    ("Instagram uji coba fitur berbasis kecerdasan buatan", "teknologi"),
    ("TikTok tambah fitur keamanan untuk remaja", "teknologi"),
    ("Penetrasi internet di Indonesia capai 78 persen", "teknologi"),
    ("Teknologi 6G sedang dikembangkan oleh beberapa negara", "teknologi"),
    ("Huawei kenalkan chipset baru dengan performa tinggi", "teknologi"),
    ("Apple Vision Pro akan tersedia global akhir tahun ini", "teknologi"),
    ("Perusahaan teknologi berlomba-lomba masuk ke pasar AI", "teknologi"),
    ("Indonesia siapkan regulasi baru untuk AI dan big data", "teknologi"),
    ("Startup fintech tumbuh cepat pasca pandemi", "teknologi"),
]

# Model training
texts, labels = zip(*data)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)

st.info("Kategori yang tersedia: Politik, Olahraga, Ekonomi, Otomotif, Teknologi")

# Input
input_text = st.text_area("Masukkan teks untuk diklasifikasikan:", height=100)

# Category colors
category_colors = {
    "politik": "#f7db6c", 
    "olahraga": "#6ab39a",
    "ekonomi": "#e69999", 
    "otomotif": "#0000",   
    "teknologi": "#8b5cf6", 
}

# Prediksi
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔍 Prediksi Kategori", use_container_width=True):
        if not input_text.strip():
            st.warning("⚠️ Harap masukkan teks terlebih dahulu.")
        else:
            input_vec = vectorizer.transform([input_text])
            prediction = model.predict(input_vec)[0]
            proba = model.predict_proba(input_vec)[0]
            confidence = max(proba) * 100
            color = category_colors.get(prediction.lower(), "#999999")

            st.markdown(
                f'<div class="result" style="background-color:{color};">'
                f'<strong>Kategori:</strong> {prediction.capitalize()}<br>'
                f'<strong>Confidence:</strong> {confidence:.2f}%'
                '</div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

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


