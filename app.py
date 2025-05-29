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
    ("KPU menetapkan daftar pemilih tetap untuk pemilu", "politik"),
    ("Polemik revisi UU KPK kembali mencuat", "politik"),
    ("Sidang paripurna membahas RAPBN tahun depan", "politik"),
    ("Demonstrasi mahasiswa menolak kebijakan pendidikan", "politik"),
    ("Pemerintah bentuk tim khusus reformasi birokrasi", "politik"),
    ("Hubungan diplomatik RI dan Australia semakin erat", "politik"),
    ("Politisi senior mengundurkan diri dari partai", "politik"),
    ("Presiden melantik pejabat baru di kabinet", "politik"),
    ("Pemerintah luncurkan program bantuan sosial baru", "politik"),
    ("RUU Perlindungan Data Pribadi disahkan", "politik"),
    ("Isu reshuffle kabinet kembali berhembus", "politik"),
    ("Wakil rakyat bahas isu HAM dalam sidang", "politik"),
    ("Presiden teken Perpres tentang digitalisasi birokrasi", "politik"),
    ("Partai oposisi kritik keras kebijakan ekonomi", "politik"),
    ("Rapat kerja menteri dan DPR bahas defisit anggaran", "politik"),
    ("Anggota DPR usulkan revisi UU ITE", "politik"),
    ("Pemerintah fokus pada pembangunan ibu kota baru", "politik"),
    ("Wakil presiden kunjungi negara sahabat untuk kerja sama", "politik"),
    ("Gubernur umumkan program anti-korupsi daerah", "politik"),
    ("Diskusi publik bahas transparansi anggaran", "politik"),
    ("Presiden terbitkan Inpres tentang transformasi digital", "politik"),
    ("Demo buruh tuntut kenaikan upah minimum", "politik"),
    ("Komisi pemilihan umum bahas logistik pemilu", "politik"),
    ("Lembaga survei rilis elektabilitas calon presiden", "politik"),
    ("Isu dinasti politik mencuat di pilkada", "politik"),
    ("Politik identitas jadi sorotan menjelang pemilu", "politik"),
    ("RUU TNI-Polri memicu perdebatan di parlemen", "politik"),
    ("Partai besar koalisi dukung calon tunggal", "politik"),
    ("Pemerintah evaluasi otonomi daerah", "politik"),
    ("Debat capres perdana digelar malam ini", "politik"),   
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
    ("Pelatih baru timnas Indonesia resmi diumumkan", "olahraga"),
    ("Pertandingan tinju dunia berlangsung di Las Vegas", "olahraga"),
    ("Tim voli putri Indonesia raih perak di ASEAN Games", "olahraga"),
    ("Messi cetak gol kemenangan PSG", "olahraga"),
    ("Kejuaraan dunia MotoGP digelar di Mandalika", "olahraga"),
    ("Pebalap Indonesia raih podium di kejuaraan Asia", "olahraga"),
    ("Turnamen tenis Australia Open dimulai pekan depan", "olahraga"),
    ("Pembalap Formula 1 Max Verstappen menangi GP Monaco", "olahraga"),
    ("Indonesia raih emas di cabang panjat tebing", "olahraga"),
    ("Pelari muda cetak rekor nasional di kejuaraan atletik", "olahraga"),
    ("Real Madrid melaju ke semifinal liga champions", "olahraga"),
    ("Fans bola tumpah ruah rayakan kemenangan klub", "olahraga"),
    ("Pertandingan UFC ditunda karena cedera petarung", "olahraga"),
    ("Ganda putra Indonesia melaju ke final turnamen dunia", "olahraga"),
    ("Bintang NBA LeBron James cetak triple-double", "olahraga"),
    ("Turnamen catur internasional digelar di Jakarta", "olahraga"),
    ("Pemain Indonesia debut di liga Jepang", "olahraga"),
    ("Final liga Eropa berlangsung seru di Budapest", "olahraga"),
    ("Suporter antri tiket final Piala Presiden", "olahraga"),
    ("Atlet Indonesia ikuti pelatnas Olimpiade", "olahraga"),
    ("Petinju legendaris umumkan pensiun", "olahraga"),
    ("Liga basket nasional dimulai bulan depan", "olahraga"),
    ("Tim e-sport Indonesia juara dunia", "olahraga"),
    ("Wasit dikecam atas keputusan kontroversial", "olahraga"),
    ("Pemain cadangan tampil cemerlang", "olahraga"),
    ("Marathon internasional akan digelar di Bali", "olahraga"),
    ("Stadion baru akan dibangun di ibu kota", "olahraga"),
    ("Tim putra voli Indonesia menang 3-1 atas Malaysia", "olahraga"),
    ("Gelandang serang cetak dua gol", "olahraga"),
    ("Pelatih memuji kerja sama tim", "olahraga"), 
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
    ("Pemerintah keluarkan stimulus ekonomi baru", "ekonomi"),
    ("Perusahaan rintisan teknologi raih pendanaan besar", "ekonomi"),
    ("Nilai tukar rupiah menguat di akhir perdagangan", "ekonomi"),
    ("Harga sembako melonjak di pasar tradisional", "ekonomi"),
    ("Investor lokal dominasi bursa saham hari ini", "ekonomi"),
    ("BI proyeksi inflasi akan tetap terkendali", "ekonomi"),
    ("Kredit perbankan tumbuh stabil sepanjang tahun", "ekonomi"),
    ("Sektor pariwisata mulai pulih pasca pandemi", "ekonomi"),
    ("Ekonomi digital Indonesia berkembang pesat", "ekonomi"),
    ("Laju inflasi dipengaruhi harga BBM", "ekonomi"),
    ("Bursa Efek Indonesia catat peningkatan transaksi", "ekonomi"),
    ("Konsumsi rumah tangga dorong pertumbuhan ekonomi", "ekonomi"),
    ("Pemerintah targetkan swasembada pangan tahun depan", "ekonomi"),
    ("Pabrik otomotif asing ekspansi ke Indonesia", "ekonomi"),
    ("Permintaan properti meningkat di semester kedua", "ekonomi"),
    ("Pendapatan pajak meningkat signifikan", "ekonomi"),
    ("Impor barang konsumsi naik tajam", "ekonomi"),
    ("Produksi minyak sawit nasional naik 3%", "ekonomi"),
    ("BI jaga stabilitas nilai tukar rupiah", "ekonomi"),
    ("Sektor UMKM mendapat bantuan modal usaha", "ekonomi"),
    ("Harga logam mulia naik di tengah ketidakpastian", "ekonomi"),
    ("Volume perdagangan kripto alami kenaikan", "ekonomi"),
    ("Saham sektor energi jadi primadona investor", "ekonomi"),
    ("Badan Pusat Statistik rilis data pengangguran", "ekonomi"),
    ("Ekspor batubara ke Tiongkok meningkat", "ekonomi"),
    ("Harga gas industri disesuaikan pemerintah", "ekonomi"),
    ("Pemerintah salurkan BLT ke masyarakat miskin", "ekonomi"),
    ("Produksi pangan terganggu oleh cuaca ekstrem", "ekonomi"),
    ("Pertumbuhan ekonomi global diprediksi melambat", "ekonomi"),
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
    ("Konsumsi BBM kendaraan turun berkat teknologi hybrid", "otomotif"),
    ("Pabrikan mobil Cina ekspansi besar-besaran ke Asia Tenggara", "otomotif"),
    ("Penjualan mobil niaga naik seiring pemulihan ekonomi", "otomotif"),
    ("Hyundai kembangkan mobil terbang untuk masa depan", "otomotif"),
    ("Pasar otomotif Indonesia targetkan ekspor ke Afrika", "otomotif"),
    ("Dealer mobil berlomba beri diskon jelang akhir tahun", "otomotif"),
    ("Bajaj rilis kendaraan roda tiga bertenaga listrik", "otomotif"),
    ("Pencurian mobil meningkat, sistem keamanan digital disorot", "otomotif"),
    ("Motor klasik kembali digemari anak muda", "otomotif"),
    ("Uji emisi kendaraan bermotor diperketat tahun ini", "otomotif"),
    ("Teknologi self-driving masih terkendala regulasi", "otomotif"),
    ("Honda Brio jadi mobil terlaris di Indonesia", "otomotif"),
    ("Pabrik mobil Toyota di Karawang tambah lini produksi", "otomotif"),
    ("Pelanggan keluhkan naiknya harga suku cadang", "otomotif"),
    ("Komunitas otomotif gelar touring ke luar kota", "otomotif"),
    ("Tesla Model 3 dapat pembaruan software otomatis", "otomotif"),
    ("Permintaan kendaraan komersial tumbuh 12 persen", "otomotif"),
    ("Mobil otonom diuji coba di jalan tol Jakarta", "otomotif"),
    ("Nissan kenalkan platform baru untuk kendaraan listrik", "otomotif"),
    ("Sistem infotainment mobil kini terkoneksi ke smartphone", "otomotif"),
    ("Pemerintah dukung konversi motor bensin ke listrik", "otomotif"),
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
    ("Tesla kembangkan chip khusus untuk kendaraan otonom", "teknologi"),
    ("YouTube hadirkan fitur anti deepfake berbasis AI", "teknologi"),
    ("Industri game online berkembang pesat di Asia", "teknologi"),
    ("Pembaruan Android 15 bawa fitur privasi lebih kuat", "teknologi"),
    ("Perkembangan teknologi drone untuk pengiriman barang", "teknologi"),
    ("Google Translate kini bisa menerjemahkan percakapan langsung", "teknologi"),
    ("Layanan cloud computing jadi andalan perusahaan besar", "teknologi"),
    ("Perusahaan Indonesia mulai adopsi teknologi blockchain", "teknologi"),
    ("Apple Watch terbaru bisa pantau kadar gula darah", "teknologi"),
    ("Riset AI Indonesia masuk jurnal internasional", "teknologi"),
    ("Kecerdasan buatan bantu deteksi penyakit lebih cepat", "teknologi"),
    ("Cybersecurity jadi isu utama di era digital", "teknologi"),
    ("Startup teknologi hijau makin dilirik investor", "teknologi"),
    ("Teknologi virtual reality digunakan di dunia pendidikan", "teknologi"),
    ("Industri teknologi global alami lonjakan rekrutmen", "teknologi"),
    ("Kominfo batasi aplikasi asing yang langgar data pribadi", "teknologi"),
    ("Twitter kini dukung konten video panjang", "teknologi"),
    ("Indonesia targetkan jadi pusat teknologi digital ASEAN", "teknologi"),
    ("Robot pintar bantu pekerjaan rumah tangga", "teknologi"),
    ("Aplikasi AI bantu siswa belajar secara adaptif", "teknologi"),
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


