# Predictive Analytics: Prediksi Perilaku Pegawai

Proyek ini bertujuan untuk memprediksi apakah seorang karyawan akan resign atau tetap bekerja di perusahaan berdasarkan data historis. Proyek ini menggunakan algoritma machine learning seperti **Random Forest Classifier**, **Naive Bayes**, dan **XGBoost** untuk membangun model prediksi.

## Dataset

Dataset yang digunakan adalah [HR Analytics Job Prediction](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction/data) yang berisi informasi tentang karyawan, seperti tingkat kepuasan kerja, evaluasi terakhir, jumlah proyek, rata-rata jam kerja bulanan, dan lainnya.

## Fitur Dataset

- **satisfaction_level**: Tingkat kepuasan kerja karyawan.
- **last_evaluation**: Skor evaluasi terakhir karyawan.
- **number_project**: Jumlah proyek yang telah dikerjakan.
- **average_monthly_hours**: Rata-rata jam kerja bulanan.
- **time_spend_company**: Lama waktu bekerja di perusahaan (tahun).
- **work_accident**: Apakah karyawan pernah mengalami kecelakaan kerja (0: tidak, 1: iya).
- **promotion_last_5years**: Apakah karyawan mendapatkan promosi dalam 5 tahun terakhir (0: tidak, 1: iya).
- **department**: Departemen tempat karyawan bekerja.
- **salary**: Tingkat gaji karyawan (low, medium, high).
- **left**: Status resign karyawan (0: tidak, 1: iya).

## Fitur Utama Proyek

1. **Exploratory Data Analysis (EDA)**:
   - Analisis univariate dan multivariate untuk memahami pola data.
   - Deteksi outliers, missing values, dan duplikasi data.

2. **Modeling**:
   - Penggunaan algoritma **Random Forest Classifier**, **Naive Bayes**, dan **XGBoost**.
   - Hyperparameter tuning menggunakan **Bayesian Optimization** untuk meningkatkan performa model.

3. **Evaluation**:
   - Evaluasi model menggunakan metrik seperti **accuracy**, **precision**, **recall**, **F1-score**, dan **confusion matrix**.

4. **Model Interpretation**:
   - Analisis feature importance untuk memahami fitur yang paling berpengaruh terhadap prediksi.

## Cara Menjalankan Proyek

### Prasyarat

Pastikan Anda memiliki Python 3.7 atau versi lebih baru dan telah menginstal `pip`. Anda juga memerlukan Jupyter Notebook untuk menjalankan file `.ipynb`.

### Langkah-langkah

1. **Clone Repository**
   ```bash
   git clone https://github.com/drewjd27/predictiveAnalysis_HrAnalyticsJobPrediction.git
   cd repository-name
   ```

2. **Install Dependencies**
   Instal semua dependensi yang diperlukan menggunakan file `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan Jupyter Notebook**
   Buka file `notebook.ipynb` menggunakan Jupyter Notebook:
   ```bash
   jupyter notebook notebook.ipynb
   ```

4. **Jalankan Script Python**
   Jika Anda ingin menjalankan proyek menggunakan script Python, jalankan file `main.py`:
   ```bash
   python main.py
   ```

## Struktur Direktori

```
.
├── HR_comma_sep.csv          # Dataset
├── main.py                   # Script utama untuk menjalankan proyek
├── notebook.ipynb            # Notebook Jupyter untuk analisis dan modeling
├── requirements.txt          # Daftar dependensi
├── readme.md                 # Dokumentasi proyek
├── laporan.md                # Laporan proyek
```

## Dependensi

Berikut adalah daftar library yang digunakan dalam proyek ini:
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- xgboost
- scikit-optimize

Semua dependensi dapat diinstal menggunakan file `requirements.txt`.

Jalankan script berikut di terminal untuk menginstall dependensi
```
pip install -r requirements.txt
```

## Hasil dan Evaluasi

Model terbaik yang digunakan adalah **Random Forest Classifier** dengan hyperparameter tuning. Model ini memiliki akurasi **98.62%** pada data testing dan menunjukkan performa yang stabil berdasarkan validasi silang.

## Kontribusi

Jika Anda ingin berkontribusi pada proyek ini, silakan buat pull request atau buka issue di repository ini.
