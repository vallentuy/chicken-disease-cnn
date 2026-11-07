# Klasifikasi Penyakit pada Ayam Berdasarkan Citra Feses Menggunakan CNN

Proyek kelompok mata kuliah **Pembelajaran Mendalam (Deep Learning)** yang mengembangkan model **Convolutional Neural Network (CNN)** untuk mengklasifikasikan citra feses ayam ke dalam empat kategori. Proyek ini membandingkan CNN baseline dengan CNN menggunakan **class weighting** serta menerapkan **Grad-CAM** untuk membantu menjelaskan area gambar yang berkontribusi terhadap prediksi.

> **Catatan:** Prediksi model hanya untuk keperluan pembelajaran/penelitian dan **bukan diagnosis dokter hewan**.

## Dataset

- **Sumber:** [Chicken Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/allandclive/chicken-disease-1)
- **Data awal:** 8.067 citra; label disimpan pada `train_data.csv` dan gambar pada folder `Train/` dalam dataset asli.
- **Resolusi dan format:** 224 × 224 piksel, RGB, berkas JPG.
- **Pemeriksaan kualitas:** 0 gambar rusak; ditemukan 222 grup duplikat identik (280 salinan tambahan), termasuk 3 grup dengan label bertentangan.
- **Pembersihan:** 6 gambar dalam grup konflik label dikeluarkan, lalu 277 salinan duplikat identik dikeluarkan. Tersisa **7.784 gambar**.

Distribusi kelas **setelah pembersihan**:

| Kelas | Jumlah gambar |
|---|---:|
| Coccidiosis | 2.342 |
| Healthy | 2.399 |
| New Castle Disease | 562 |
| Salmonella | 2.481 |
| **Total** | **7.784** |

Dataset dibagi secara *stratified* menjadi **70% training, 15% validation, dan 15% testing**, menggunakan `random_state=42`. Data dinormalisasi ke rentang piksel 0–1; augmentasi saat training meliputi horizontal flip, rotasi, dan zoom. Pemeriksaan duplikat identik dilakukan sebelum pembagian untuk mengurangi risiko *data leakage*.

## Arsitektur dan pelatihan

Arsitektur CNN baseline dan CNN class weighting memiliki susunan yang sama:

1. Input RGB **224 × 224 × 3**, dengan augmentasi selama training.
2. Empat blok `Conv2D` (32, 64, 128, 256 filter; aktivasi ReLU) dan `MaxPooling2D`.
3. `GlobalAveragePooling2D`, `Dense(128, relu)`, dan `Dropout(0.3)`.
4. Output `Dense(4, softmax)` untuk empat kelas.

**Jumlah parameter:** 421.828. Pelatihan memakai optimizer **Adam**, *learning rate* **0.001**, *batch size* **32**, maksimum **30 epoch**, `sparse_categorical_crossentropy`, `EarlyStopping(patience=5, monitor='val_loss')`, serta penyimpanan model terbaik berdasarkan *validation loss*.

## Dua fitur unik

**1. Class Weighting:** Menggunakan bobot kelas yang dihitung dari data training untuk memberi perhatian lebih besar pada kelas minoritas, khususnya New Castle Disease. Kinerja model dievaluasi terhadap CNN baseline dengan data yang sama.

**2. Grad-CAM:** Menghasilkan *heatmap* area citra yang berkontribusi terhadap prediksi. Visualisasi menunjukkan bahwa pada sebagian contoh, model memberi perhatian pada tepi atau latar belakang foto; ini merupakan temuan kualitatif tentang keterbatasan model, bukan bukti bahwa prediksi diagnosis benar.

## Hasil evaluasi akhir (testing)

Pengujian dilakukan pada **1.168 gambar testing** yang tidak dipakai untuk melatih atau memilih model.

| Metrik | CNN Baseline | CNN Class Weighting |
|---|---:|---:|
| Accuracy | **90,24%** | 89,47% |
| Precision (macro) | **88,52%** | 86,56% |
| Recall (macro) | 87,56% | **89,50%** |
| F1-score (macro) | **88,01%** | 87,76% |

Pada kelas **New Castle Disease**, *recall* meningkat dari **76,19% (64/84)** pada baseline menjadi **89,29% (75/84)** dengan class weighting. Sebaliknya, akurasi keseluruhan dan macro F1 sedikit menurun. Hasil ini menunjukkan *trade-off* antara sensitivitas terhadap kelas minoritas dan performa agregat.

## Struktur repository

```text
chicken-disease-cnn/
├── Chicken_Disease_CNN_UAS.ipynb   # Notebook eksperimen lengkap
├── README.md                       # Dokumentasi proyek
├── requirements.txt                # Dependensi Python
├── models/
│   ├── best_cnn_baseline.keras
│   └── best_cnn_weighted.keras
├── results/
│   ├── confusion_matrix_testing.png
│   ├── gradcam_baseline_4kelas.png
│   ├── hasil_evaluasi_testing.csv
│   └── perbandingan_model.png
├── splits/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
└── data/
    └── train_data_clean.csv
```

> Folder di atas menggambarkan **susunan yang disarankan**. Pastikan nama notebook dan semua file sesuai dengan yang benar-benar diunggah ke GitHub. Berkas CSV di `data/` dan `splits/` berisi metadata/nama gambar, **bukan** berkas citra aslinya.

## Cara menjalankan

1. Unduh dataset dari tautan Kaggle di atas (ZIP). Dataset mentah **tidak disertakan** dalam repository.
2. Buka `Chicken_Disease_CNN_UAS.ipynb` di [Google Colab](https://colab.research.google.com/).
3. Jika dibutuhkan, pasang dependensi dari file `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
4. Ikuti cell notebook secara berurutan: upload dan ekstrak ZIP, periksa struktur folder, pembersihan, EDA, pembagian dataset, preprocessing, training, dan evaluasi.
5. Jika hanya ingin menggunakan model terlatih, muat berkas `.keras` dari folder `models/` pada runtime yang sesuai. Gunakan pemetaan label yang sama dengan notebook: `Coccidiosis`, `Healthy`, `New Castle Disease`, `Salmonella`.

**Penting:** Path pada notebook awal menggunakan lokasi `/content/...` di Google Colab. Sesuaikan path jika dataset ditempatkan di direktori lain. Notebook mungkin memerlukan penyesuaian path berkas model atau CSV ketika dijalankan kembali dari awal.

## Keterbatasan

- Dataset tidak seimbang; kelas New Castle Disease memiliki sampel jauh lebih sedikit.
- Duplikat byte-identik sudah ditangani, tetapi gambar yang nyaris identik atau berasal dari sumber pengambilan yang sama belum diverifikasi sepenuhnya. Hal ini masih berpotensi menyebabkan kebocoran informasi antarsplit.
- Beberapa contoh Grad-CAM memperlihatkan aktivasi pada latar belakang, sehingga model mungkin memanfaatkan ciri yang tidak terkait langsung dengan feses.
- Hasil pada dataset ini belum membuktikan kemampuan generalisasi untuk kondisi peternakan atau gambar baru di luar dataset.

## Anggota dan kontribusi kelompok

Tambahkan nama, NIM, serta kontribusi **sesuai pekerjaan yang benar-benar dilakukan** oleh setiap anggota kelompok. Pembagian tugas lengkap juga dicantumkan dalam laporan PDF.

## Referensi

- Dataset: https://www.kaggle.com/datasets/allandclive/chicken-disease-1
- Dokumentasi TensorFlow/Keras: https://www.tensorflow.org/
- Dokumentasi Scikit-learn: https://scikit-learn.org/
