# Klasifikasi Penyakit pada Ayam Berdasarkan Citra Feses Menggunakan CNN

**Proyek Tugas 1 dan Ujian Tengah Semester (UTS) — Mata Kuliah Pembelajaran Mendalam**

Proyek ini mengembangkan sistem klasifikasi citra feses ayam menggunakan **Convolutional Neural Network (CNN)**. Pada Tugas 1, dilakukan pengolahan dataset, pelatihan, evaluasi, dan interpretasi model. Pada tahap UTS, model terlatih diintegrasikan ke dalam **aplikasi interaktif Gradio** agar pengguna dapat mengunggah gambar, membandingkan dua model, dan melihat visualisasi **Grad-CAM** tanpa melatih ulang model.


## 1. Tujuan Proyek

- Mengklasifikasikan citra feses ayam ke dalam **empat kelas**: `Coccidiosis`, `Healthy`, `New Castle Disease`, dan `Salmonella`.
- Membandingkan performa **CNN Baseline** dan **CNN dengan Class Weighting** untuk menangani ketidakseimbangan kelas.
- Menyediakan **Grad-CAM** untuk memperlihatkan area citra yang berkontribusi terhadap prediksi.
- Mengimplementasikan model dalam antarmuka Gradio sebagai pengembangan pada tahap UTS.

## 2. Dataset dan Prapemrosesan

**Sumber:** [Chicken Disease Dataset — Kaggle](https://www.kaggle.com/datasets/allandclive/chicken-disease-1)

| Tahap | Jumlah gambar |
|---|---:|
| Dataset awal | 8.067 |
| Dihapus karena konflik label dan duplikat identik | 283 |
| **Dataset setelah pembersihan** | **7.784** |

Dataset terdiri dari citra JPG berukuran **224 × 224 piksel** dengan label pada `train_data.csv`. Pemeriksaan menemukan **222 kelompok gambar identik**, dengan **280 salinan tambahan**, termasuk **3 kelompok berlabel konflik**. Sebanyak 6 gambar dari kelompok konflik dan 277 salinan duplikat tambahan dikeluarkan sebelum pembagian data.

**Distribusi kelas setelah pembersihan:**

| Kelas | Jumlah |
|---|---:|
| Coccidiosis | 2.342 |
| Healthy | 2.399 |
| New Castle Disease | 562 |
| Salmonella | 2.481 |
| **Total** | **7.784** |

Dataset dibagi secara **stratified** (70% training, 15% validation, 15% testing, `random_state=42`):

| Split | Jumlah |
|---|---:|
| Training | 5.448 |
| Validation | 1.168 |
| Testing | 1.168 |

Prapemrosesan model meliputi pembacaan citra RGB dengan TensorFlow, *resize* ke 224 × 224, dan normalisasi nilai piksel ke rentang 0–1. Augmentasi pada pelatihan mencakup *horizontal flip*, rotasi, dan zoom. **Aplikasi UTS menggunakan pembacaan file asli (`tf.io.read_file` dan `tf.image.decode_jpeg` untuk JPG)** agar hasil prediksi konsisten dengan evaluasi Tugas 1.

## 3. Arsitektur CNN

Kedua model menggunakan arsitektur identik, tetapi strategi pelatihannya berbeda.

1. Input gambar: **224 × 224 × 3**.
2. Empat blok **Conv2D + MaxPooling2D** dengan 32, 64, 128, dan 256 filter (aktivasi ReLU).
3. **GlobalAveragePooling2D**.
4. **Dense(128, ReLU)** dan **Dropout(0,3)**.
5. **Dense(4, Softmax)** sebagai output empat kelas.

Jumlah parameter setiap model: **421.828**.

Konfigurasi pelatihan: optimizer Adam, *learning rate* 0,001, *batch size* 32, maksimum 30 *epoch*, `sparse_categorical_crossentropy`, dan *early stopping* berdasarkan `val_loss` dengan `patience=5`.

**Perbedaan pendekatan:**

- **CNN Baseline:** pelatihan standar tanpa bobot kelas khusus.
- **CNN Class Weighting:** menggunakan bobot yang dihitung dari distribusi data training agar kesalahan pada kelas minoritas, terutama `New Castle Disease`, memperoleh perhatian lebih besar.

## 4. Hasil Evaluasi pada Data Testing

Evaluasi dilakukan pada **1.168 gambar testing** yang dipisahkan dari data training dan validation.

| Metrik | CNN Baseline | CNN Class Weighting |
|---|---:|---:|
| **Accuracy** | **90,24%** | 89,47% |
| Precision (macro) | 88,52% | 86,56% |
| Recall (macro) | 87,56% | **89,50%** |
| F1-score (macro) | **88,01%** | 87,76% |
| Recall kelas New Castle Disease | 76,19% (64/84) | **89,29% (75/84)** |

**Interpretasi:** CNN Baseline unggul dalam akurasi keseluruhan dan *macro F1*. Penggunaan *class weighting* meningkatkan *recall* kelas New Castle Disease sebesar sekitar **13,10 poin persentase**, tetapi sedikit menurunkan akurasi keseluruhan. Perubahan ini menunjukkan *trade-off* antara sensitivitas pada kelas minoritas dan performa agregat.

Berkas hasil evaluasi tersimpan pada folder [`results/`](results/).

## 5. Pengembangan Aplikasi pada UTS

Aplikasi Gradio menggunakan **dua model yang telah dilatih pada Tugas 1**, sehingga **tidak perlu training ulang** untuk melakukan inferensi.

**Fitur aplikasi:**

1. **Upload gambar:** menerima file JPG/JPEG atau PNG untuk diklasifikasikan.
2. **Perbandingan dua model:** menampilkan prediksi dan *confidence* CNN Baseline serta CNN Class Weighting secara bersamaan.
3. **Probabilitas per kelas:** menampilkan tabel probabilitas keempat kategori untuk kedua model.
4. **Grad-CAM:** pengguna dapat memilih model yang divisualisasikan, kemudian melihat *heatmap* dan gambar *overlay*.

### Alur penggunaan

```text
Upload gambar feses ayam (JPG/PNG)
                |
                v
Baca file dan preprocessing TensorFlow
(RGB, resize 224x224, normalisasi /255)
                |
                v
     Dua model CNN terlatih
       /                 \
      v                   v
 CNN Baseline       CNN Class Weighting
       \                 /
        v               v
 Prediksi, confidence, tabel probabilitas
                |
                v
 Pilih model untuk visualisasi Grad-CAM
                |
                v
       Heatmap dan overlay
```

### Contoh pengujian aplikasi

Kedua contoh berikut menggunakan gambar dari data testing dengan **label asli `Healthy`**.

| Contoh input | Label asli | Prediksi CNN Baseline | Confidence Baseline | Keterangan |
|---|---|---|---:|---|
| `healthy.1007.jpg` | Healthy | Healthy | 98,19% | Prediksi benar |
| `healthy.1620.jpg` | Healthy | New Castle Disease | 51,36% | Prediksi salah |

Contoh salah prediksi memperlihatkan bahwa sistem masih memiliki keterbatasan dan **confidence bukan jaminan kebenaran**. Dokumentasi tampilan aplikasi terdapat pada folder [`demo/screenshots/`](demo/screenshots/).

> **Catatan Grad-CAM:** Pada sejumlah contoh, aktivasi terlihat juga pada tepi atau latar belakang gambar. Visualisasi ini bersifat interpretasi kualitatif dan tidak membuktikan bahwa model mengidentifikasi penyebab penyakit secara klinis.

## 6. Struktur Repository

```text
chicken-disease-cnn/
├── Chicken_Disease_CNN.ipynb        # Eksperimen dan pelatihan Tugas 1
├── README.md                        # Dokumentasi Tugas 1 dan UTS
├── requirements.txt                 # Versi dependensi
├── app/
│   └── Chicken_Disease_CNN_UTS.ipynb  # Aplikasi Gradio dan pengujian UTS
├── models/
│   ├── best_cnn_baseline.keras
│   └── best_cnn_weighted.keras
├── demo/
│   └── screenshots/                 # Tangkapan layar demo aplikasi
├── results/                         # Grafik dan tabel hasil evaluasi
├── splits/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
└── data/
    └── train_data_clean.csv
```

**Catatan:** File CSV menyimpan metadata dan nama gambar, bukan seluruh file JPG dataset asli. Dataset lengkap diperoleh dari Kaggle. Gambar contoh pengujian dapat ditambahkan ke `demo/samples/` bila diperlukan.

## 7. Cara Menjalankan Aplikasi UTS di Google Colab

**Notebook aplikasi:** [`app/Chicken_Disease_CNN_UTS.ipynb`](app/Chicken_Disease_CNN_UTS.ipynb)

**Persiapan:**

1. Buka notebook UTS di [Google Colab](https://colab.research.google.com/).
2. Instal dependensi, khususnya Gradio. Notebook sudah menyertakan cell `!pip install -q gradio`. Versi seluruh library untuk lingkungan yang didokumentasikan terdapat pada [`requirements.txt`](requirements.txt).
3. Unduh dua model terlatih dari folder [`models/`](models/), lalu unggah ke Colab melalui cell `files.upload()` dengan nama:
   - `best_cnn_baseline.keras`
   - `best_cnn_weighted.keras`
4. Pastikan model berada pada path `/content/best_cnn_baseline.keras` dan `/content/best_cnn_weighted.keras`, sesuai kode notebook.
5. Jalankan cell pemuatan model dan definisi fungsi prediksi serta Grad-CAM.
6. **Gunakan definisi `preprocess_image()` versi terakhir** yang membaca file gambar menggunakan `tf.io.read_file()`, `tf.image.decode_jpeg()` atau `tf.image.decode_png()`, lalu `tf.image.resize()` dan normalisasi `/255`. Ini adalah perbaikan yang menjaga konsistensi dengan Tugas 1.
7. Jalankan cell pembuatan antarmuka `with gr.Blocks(...) as demo_uts:`. Pastikan komponen input menggunakan `type="filepath"` dan `image_mode=None`.
8. Jalankan cell peluncuran berikut, kemudian buka tautan `gradio.live` yang baru ditampilkan:

```python
demo_uts.launch(
    inline=False,
    share=True,
    debug=False,
    show_error=True
)
```

9. Upload gambar pengujian, klik **Prediksi Penyakit**, lalu amati hasil kedua CNN, tabel probabilitas, serta Grad-CAM.

**Penting untuk reproduksi:** Notebook UTS saat ini merupakan notebook pengembangan yang masih menyimpan beberapa cell eksperimen, definisi preprocessing lama, dan cell pengujian dataset. **Jangan langsung menjalankan semua cell dengan `Run all` pada runtime baru.** Pilih cell yang diperlukan sesuai urutan di atas; bila ada definisi fungsi yang berulang, gunakan versi terakhir yang sudah diperbaiki. Cell pengujian `test.csv` atau dataset ZIP tidak diperlukan untuk sekadar menjalankan antarmuka dan memprediksi satu gambar.

Tautan `gradio.live` hanya merupakan tautan sementara selama aplikasi berjalan. **Deployment permanen belum dilakukan** dan tidak termasuk tahapan UTS saat ini sesuai arahan dosen. Demonstrasi disajikan melalui antarmuka yang dijalankan dari Colab beserta screenshot dan penjelasan tertulis.

## 8. Dependensi

Dependensi utama proyek meliputi TensorFlow, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, Pillow, dan Gradio. File [`requirements.txt`](requirements.txt) berisi versi library yang digunakan pada runtime Colab UTS.

Untuk instalasi dari file versi tersebut, setelah berkas tersedia pada lingkungan Python:

```bash
pip install -r requirements.txt
```

Versi paket mencerminkan lingkungan Colab saat file dibuat dan belum merupakan jaminan kompatibilitas pada semua sistem operasi atau versi Python.

## 9. Keterbatasan dan Rencana Pengembangan

- Kelas dataset tidak seimbang; jumlah citra New Castle Disease jauh lebih sedikit.
- Duplikat byte-identik telah ditangani, tetapi duplikat yang *hampir identik* dan kedekatan sumber pengambilan gambar belum sepenuhnya diverifikasi.
- Grad-CAM memperlihatkan bahwa latar belakang mungkin ikut memengaruhi beberapa prediksi.
- Performa pada data testing internal belum menjamin generalisasi terhadap foto baru, kondisi pencahayaan lain, atau lingkungan peternakan yang berbeda.
- Aplikasi belum memiliki validasi klinis, mekanisme penolakan input yang tidak relevan, atau deployment permanen.

Pengembangan berikutnya dapat mencakup pembersihan notebook agar lebih mudah direproduksi, evaluasi pada dataset eksternal, penambahan contoh gambar yang lebih beragam, dan deployment setelah tahap akademik mengizinkan.

## 10. Anggota Kelompok

- **Alvin Darmawan**
- **Valentino Damar**
- **Vallent Thery Lim**
- **Alvern Wilbert Tio**

## 11. Referensi

- [Chicken Disease Dataset — Kaggle](https://www.kaggle.com/datasets/allandclive/chicken-disease-1)
- [TensorFlow / Keras](https://www.tensorflow.org/)
- [Gradio](https://www.gradio.app/)
- [Grad-CAM — Selvaraju et al. (2017)](https://arxiv.org/abs/1610.02391)
- [Scikit-learn](https://scikit-learn.org/)

---

**Repository:** https://github.com/vallentuy/chicken-disease-cnn
