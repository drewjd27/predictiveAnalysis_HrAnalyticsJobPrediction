# %% [markdown]
# # Predictive Analytics: Prediksi Perilaku Pegawai
# **Domain**: Ekonomi dan Bisnis

# %% [markdown]
# ## Data Understanding

# %% [markdown]
# ### Tentang Dataset

# %% [markdown]
# Dataset dalam penelitian ini diperoleh dari Kaggle dengan nama dataset 
# [Hr Analytics Job Prediction](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction/data). Dataset ini berisi data pegawai yang bekerja di sebuah perusahaan, dengan 14999 baris data. Tujuan utama penelitian ini terhadap dataset ini adalah untuk membantu departemen HR dalam memprediksi perilaku karyawan, yaitu apakah pegawai resign atau tetap bekerja pada perusahaan tersebut.
# 
# Dataset ini tersedia dalam format CSV dan mencakup berbagai atribut yang relevan untuk analisis HR, antara lain:
# 
# - **satisfaction_level**: Tingkat kepuasan karyawan terhadap pegawaian mereka.
# - **last_evaluation**: Skor evaluasi terakhir yang diterima oleh karyawan.
# - **number_projects**: Jumlah proyek yang telah dikerjakan oleh karyawan.
# - **average_montly_hours**: Rata-rata jam kerja bulanan karyawan.
# - **time_spent_company**: Lama waktu (dalam tahun) karyawan bekerja di perusahaan.
# - **Work_accident**: Apakah karyawan pernah mengalami kecelakaan kerja (0: tidak, 1: iya).
# - **promotion_last_5years**: Apakah karyawan mendapatkan promosi dalam 5 tahun terakhir (0: tidak, 1: iya).
# - **Department**: Departemen tempat karyawan bekerja (sales, technical, support, IT, RandD, product_mng, marketing, accounting, hr, management).
# - **salary**: Tingkat gaji karyawan (low, medium, high).
# - **left**: Fitur target dari dataset ini, sebagai indikator apakah karyawan telah meninggalkan perusahaan (0: tidak, 1: iya).

# %% [markdown]
# ### Import Libraries

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from scipy.stats import zscore
import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import learning_curve, LearningCurveDisplay
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV

# %% [markdown]
# ### Load Dataset

# %%
# Load dataset
df = pd.read_csv('HR_comma_sep.csv')
df.head()

# %% [markdown]
# ### Deskripsi Variabel

# %%
print("Total Baris dan Kolom:", df.shape)
print("Total Data:", df.size)

# %%
#Cek tipe data dari setiap kolom
df.info()

# %%
#Cek rangkuman statistik untuk setiap fitur numerik 
df.describe()

# %% [markdown]
# ### 📌 **Insight dari Data Understanding**
# 
# Sebagian besar tipe data sudah benar. Saya akan mengonversi tipe data objek tertentu, seperti `salary` dan `Department` ke dalam format numerik, mungkin melalui encoding, untuk meningkatkan akurasi model. Untuk saat ini saya akan mengonversinya ke dalam tipe data kategorikal untuk EDA nanti. Saya juga akan memperbaiki typo pada penamaan kolom `average_montly_hours`. 

# %% [markdown]
# ## Exploratory Data Analysis

# %% [markdown]
# ### Exploratory Data Analysis - Data Cleaning

# %% [markdown]
# Dalam tahap ini, saya akan mengecek dan mengatasi data yang :
# - Ubah nama kolom
# - Missing values
# - Duplikat
# - Tipe data tidak tepat
# - Outliers

# %% [markdown]
# #### Ubah Nama Kolom

# %%
# Ubah nama kolom menjadi huruf kecil, dan perbaiki typo pada nama kolom
df = df.rename(columns={'average_montly_hours': 'average_monthly_hours',
                        'Work_accident': 'work_accident',
                        'Department': 'department'})

# tampilkan nama kolom setelah diubah
df.columns

# %% [markdown]
# #### Missing Values

# %%
# cek missing value
print("Missing Value:\n", df.isnull().sum())

# %% [markdown]
# #### Data Duplikat

# %%
# cek data duplikat
print("Data Duplikat:", df.duplicated().sum())

# %%
# Lihat data duplikat

duplicates = df[df.duplicated(keep=False)]
duplicates.head()

pd.concat([df, duplicates]).sort_values(by=df.columns.tolist())

# %%
#Cek persentase dan jumlah data duplikat
percentage_duplicates = (df.duplicated().sum() / len(df)) * 100
count_duplicates = df.duplicated().sum()
print(f"Persen baris duplikat: {percentage_duplicates:.2f}%")
print(f"Jumlah baris duplikat: {count_duplicates}")

# %% [markdown]
# Saya menemukan 3008 baris data yang terduplikasi yang merupakan 20,05% dari 14999 baris data. Nilai yang terduplikasi ini tidak menambah nilai atau informasi apa pun dan berpotensi menghambat proses visualisasi dan analisis data. Duplikasi tersebut juga dapat menyebabkan bias, tidak hanya dalam visualisasi, tetapi juga pada model.

# %%
#copy data df ke df_cleaned_duplicates
df_cleaned_duplicates = df.copy()

#Hapus data yang duplikat
df_cleaned_duplicates.drop_duplicates(inplace=True)
# data sebelum dihapus duplikat
print("Data sebelum dihapus duplikat:", df.shape[0])
# data setelah dihapus duplikat
print("Data setelah dihapus duplikat:", df_cleaned_duplicates.shape[0])
#cek apakah ada data duplikat setelah dihapus
print("Apakah masih ada data duplikat setelah dihapus?", df_cleaned_duplicates.duplicated().any())

# %% [markdown]
# #### Ubah Tipe Data

# %%
df_cleaned_duplicates['work_accident'] = df_cleaned_duplicates['work_accident'].astype('category')
df_cleaned_duplicates['promotion_last_5years'] = df_cleaned_duplicates['promotion_last_5years'].astype('category')
df_cleaned_duplicates['left'] = df_cleaned_duplicates['left'].astype('category')

# %%
for col in df_cleaned_duplicates.select_dtypes(include='object').columns:
    df_cleaned_duplicates[col] = df[col].astype('category')

# %%
# Cek tipe data setelah diubah
df_cleaned_duplicates.info()

# %% [markdown]
# #### Outliers

# %%
# Atur tema warna hijau
sns.set_theme(style="whitegrid")

# Ambil kolom numerik, kecualikan kolom tertentu
numeric_cols = df_cleaned_duplicates.select_dtypes(include=['float64', 'int64']).columns
excluded_cols = ['work_accident', 'left', 'promotion_last_5years']
numeric_cols_filtered = [col for col in numeric_cols if col not in excluded_cols]

# Hitung baris dan kolom untuk subplot
n_cols = 3
n_rows = math.ceil(len(numeric_cols_filtered) / n_cols)

# Buat figure
plt.figure(figsize=(5 * n_cols, 4 * n_rows))
for i, col in enumerate(numeric_cols_filtered):
    if i < n_rows * n_cols:  # Ensure the subplot index is within bounds
        plt.subplot(n_rows, n_cols, i + 1)
        sns.boxplot(data=df_cleaned_duplicates, x=col, color='limegreen')
        plt.title(f'Boxplot dari {col}')
plt.tight_layout()
plt.show()

# %%
# Hitung IQR untuk kolom 'time_spend_company'
Q1 = df_cleaned_duplicates['time_spend_company'].quantile(0.25)
Q3 = df_cleaned_duplicates['time_spend_company'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Hitung jumlah outliers
outliers = df_cleaned_duplicates[(df_cleaned_duplicates['time_spend_company'] < lower_bound) | 
                                  (df_cleaned_duplicates['time_spend_company'] > upper_bound)]
jumlah_outliers = outliers.shape[0]
print(f"Jumlah outliers pada fitur 'time_spend_company': {jumlah_outliers}")

# %% [markdown]
# Terdapat outliers pada kolom `time_spend_company`. Namun outlier nya tidaklah extreme. Jadi menurut saya tidak perlu menangani outlier tersebut. Mari kita lihat nanti performa model yang akan kita latih dengan dataset ini.

# %% [markdown]
# 📌 **Insight dari Data Cleaning**
# 
# 1. **Standardisasi Nama Kolom:**
#    - Nama kolom diubah menjadi huruf kecil seluruhnya.
#    - Typo diperbaiki, contohnya:
#      - `average_montly_hours` → `average_monthly_hours`
#      - `Work_accident` → `work_accident`
#      - `Department` → `department`
#    - Ini penting agar analisis berikutnya lebih konsisten, mencegah error karena typo atau perbedaan huruf kapital.
# 
# 2. **Pengecekan dan Penanganan Missing Value:**
#    - Dicek jumlah data kosong di semua kolom.
#    - Dari hasilnya, **tidak ada missing value** di dataset ini, sehingga tidak perlu dilakukan imputasi atau pembuangan data.
# 
# 3. **Cek dan Tangani Data Duplikat:**
#    - Data diperiksa apakah ada yang duplikat.
#    - Ditemukan beberapa data duplikat, lalu **duplikat dihapus**.
#    - Setelah data duplikat dihapus, jumlah baris pada data adalah 11991.
#    - Ini penting agar analisis tidak bias karena contoh data yang berulang.
# 
# 
# 4. **Ubah Tipe Data untuk Konsistensi:**
#    - Ada pengubahan tipe data untuk kolom bertipe kategorikal agar konsisten formatnya (misal, merapikan kategori di kolom `department` dan `salary`).
#    - Hal ini penting untuk memudahkan proses encoding nantinya.
# 
# 5. **Deteksi Data Outlier:**
#    - Dilakukan pengecekan distribusi untuk beberapa kolom numerik menggunakan boxplot. Terdapat outlier pada fitur `time_spend_company`
#    - Kemudian outlier dideteksi dengan metode IQR. Jumlah data outlier pada fitur `time_spend_company`adalah sebanyak 824 baris
#    - Namun, karena data outlier masih tergolong tidak extreme, maka data outlier dibiarkan saja. Ini juga berguna agar tidak ada pola yang hilang
# 
# 

# %% [markdown]
# ### Exploratory Data Analysis - Univariate Analysis

# %%
df_cleaned = df_cleaned_duplicates.copy()

# %%
# Masukkan data numerik dan data kategorikal ke dalam list variabel
num_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()

cat_cols = df_cleaned.select_dtypes(include=['category']).columns.tolist()

# %%
num_cols

# %%
cat_cols

# %% [markdown]
# #### Categorical Features

# %% [markdown]
# ##### Fitur left

# %%
# Buat label sementara untuk plotting
temp_labels = df_cleaned['left'].map({0: 'tidak', 1: 'iya'})

feature = cat_cols[1]
count = temp_labels.value_counts()
percent = 100 * temp_labels.value_counts(normalize=True)
df_temp = pd.DataFrame({'jumlah sampel': count, 'persentase': percent.round(1)})
print(df_temp)
count.plot(kind='bar', title="Distribusi pegawai Yang Resign", color='limegreen');

# %% [markdown]
# Dari grafik Distribusi pegawai Yang Resign dapat disimpulkan beberapa hal:
# - Jumlah pegawai yang **tidak resign** jauh lebih banyak dibandingkan dengan yang **resign**.
# - Jumlah pegawai yang **tidak resign** adalah **10000** dengan persentase **83.4**, sedangkan yang **resign** sekitar **1991** dengan persentase **16.6**.

# %% [markdown]
# ##### Fitur work_accident

# %%
# Buat label sementara untuk plotting
temp_labels = df_cleaned['work_accident'].map({0: 'tidak', 1: 'iya'})

# Hitung jumlah dan persentase
count = temp_labels.value_counts()
percent = 100 * temp_labels.value_counts(normalize=True)
df_temp = pd.DataFrame({'jumlah sampel': count, 'persentase': percent.round(1)})

# Tampilkan dataframe
print(df_temp)

# Plot data
count.plot(kind='bar', title="Distribusi pegawai Yang Mengalami Kecelakaan Kerja", color='limegreen');

# %% [markdown]
# Dari analisa fitur work_accident:
# - Jumlah pegawai yang **tidak pernah mengalami kecelakaan** jauh lebih banyak dibandingkan dengan yang **pernah mengalami kecelakaan**.
# - Jumlah pegawai yang **tidak pernah mengalami kecelakaan** adalah **10141** dengan persentase **84.6**, sedangkan yang **pernah mengalami kecelakaan** sekitar **1850** dengan persentase **15.4**.

# %% [markdown]
# ##### Fitur promotion_last_5years

# %%
# Buat label sementara untuk plotting
temp_labels = df_cleaned['promotion_last_5years'].map({0: 'tidak', 1: 'iya'})

# Hitung jumlah dan persentase
count = temp_labels.value_counts()
percent = 100 * temp_labels.value_counts(normalize=True)
df_temp = pd.DataFrame({'jumlah sampel': count, 'persentase': percent.round(1)})

# Tampilkan dataframe
print(df_temp)

# Plot data
count.plot(kind='bar', title="Distribusi pegawai Yang Dipromosikan 5 Tahun Terakhir", color='limegreen');

# %% [markdown]
# Dari analisa fitur promotion_last_5years dapat disimpulkan beberapa hal:
# - Jumlah pegawai yang **tidak pernah dipromosikan 5 tahun terakhir** jauh lebih banyak dibandingkan dengan yang **pernah dipromosikan 5 tahun terakhir**.
# - Jumlah pegawai yang **tidak pernah dipromosikan 5 tahun terakhir** adalah **11788** dengan persentase **98.3**, sedangkan yang **pernah dipromosikan 5 tahun terakhir** sekitar **203** dengan persentase **1.7**.

# %% [markdown]
# ##### Fitur department

# %%
feature = cat_cols[3]
count = df_cleaned[feature].value_counts()
percent = 100*df_cleaned[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title="Distribusi pegawai Berdasarkan Divisinya", color='limegreen');

# %% [markdown]
# Dari analisa fitur Department, dapat disimpulkan beberapa hal, yaitu:
# - Divisi dengan jumlah pekerja **terbanyak** adalah **sales** (lebih dari 3000 karyawan).
# - Disusul oleh **technical** dan **support**, keduanya juga memiliki proporsi yang signifikan.
# - Divisi dengan jumlah karyawan **paling sedikit** adalah **management**, hanya sekitar **3.6%**, yang bisa dimaklumi karena biasanya posisi manajerial memang lebih sedikit.
# - Tiga divisi teratas (**Sales, Technical, Support**) menyumbang lebih dari **60%** dari total tenaga kerja. Hal ini bisa menunjukkan bahwa perusahaan memiliki fokus besar pada **penjualan**, **dukungan teknis**, dan **layanan pelanggan**.
# - **Divisi IT, R&D, dan Product Management** tergolong menengah, dan ini bisa menunjukkan perusahaan juga cukup memperhatikan sisi pengembangan teknologi dan produk — meski tidak sebesar tenaga sales dan support.

# %% [markdown]
# ##### Fitur salary

# %%
feature = cat_cols[4]
count = df_cleaned[feature].value_counts()
percent = 100 * df_cleaned[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel': count, 'persentase': percent.round(1)})
print(df)
count.plot(kind='bar', title="Distribusi pegawai Berdasarkan Gajinya", color='limegreen');

# %% [markdown]
# Berikut ini beberapa kesimpulan berdasarkan analisa fitur salary:
# - Mayoritas karyawan berada di kategori gaji **rendah** (`low`), yaitu **5.740 orang** atau **47.9%** dari total populasi.
# - Disusul oleh kategori **gaji menengah** (`medium`) sebanyak **5.261 orang** (**43.9%**).
# - Sedangkan karyawan dengan **gaji tinggi** (`high`) hanya sekitar **990 orang** atau **8.3%**.
# - Hampir 92% dari total karyawan memiliki gaji di level `low` dan `medium`, sedangkan hanya 8% yang menikmati gaji tinggi. Ini menunjukkan adanya kesenjangan distribusi kompensasi dalam perusahaan.
# - Karena kelompok `low salary` memiliki proporsi terbesar, kelompok ini kemungkinan besar juga menyumbang sebagian besar kasus **resign** (yang perlu dikonfirmasi melalui analisis lebih lanjut). Gaji rendah umumnya dikaitkan dengan kepuasan kerja rendah dan kemungkinan lebih besar untuk mencari peluang kerja di tempat lain.
# - Jika ternyata banyak pekerja dari kelompok gaji rendah yang resign, maka strategi peningkatan kesejahteraan, pelatihan, atau insentif bisa difokuskan ke kelompok ini untuk menurunkan angka turnover.
# - Persentase pegawai yang resign semakin tinggi apabila gaji yang diberikan kepada pegawai semakin rendah.

# %% [markdown]
# #### Numerical Features

# %%
# Atur tema warna hijau
sns.set_theme(style="whitegrid")

# Hitung baris dan kolom untuk subplot
n_cols = 3
n_rows = math.ceil(len(num_cols) / n_cols)

# Buat figure
plt.figure(figsize=(5 * n_cols, 4 * n_rows))
for i, col in enumerate(num_cols):
    if i < n_rows * n_cols:  # Ensure the subplot index is within bounds
        plt.subplot(n_rows, n_cols, i + 1)
        sns.histplot(data=df_cleaned_duplicates, x=col, kde=True, color='limegreen')
        plt.title(f'Distribusi dari {col}')
plt.tight_layout()
plt.show()

# %% [markdown]
# Berdasarkan visualisasi grafik distribusi data numerik di atas, dapat disimpulkan beberapa hal.
# 1. **Satisfaction Level**
# - **Distribusi** agak bimodal dan sedikit condong ke kanan.
# - Ada dua puncak besar:
#   - Sekitar **0.4–0.5** (rendah)
#   - Sekitar **0.7–0.8** (tinggi)
# - **Insight**:
#   - Ada kelompok karyawan yang sangat puas dan cukup banyak yang kurang puas.
#   - Karyawan dengan tingkat kepuasan rendah ini patut diawasi karena mereka berpotensi lebih besar untuk resign.
# 
# ---
# 
# 2. **Last Evaluation**
# - Distribusi relatif **merata** dari 0.5 sampai 1, dengan sedikit puncak di area **0.55–0.6**.
# - **Insight**:
#   - Evaluasi kerja karyawan cenderung tersebar merata, artinya sistem penilaian cukup bervariasi.
#   - Tidak ada indikasi bahwa karyawan yang performanya sangat rendah mendominasi.
# 
# ---
# 
# 3. **Number of Projects**
# - Ada puncak jelas pada angka:
#   - **3 proyek** (terbanyak)
#   - **4 proyek** dan **5 proyek** menyusul
# - Hanya sedikit karyawan yang menangani **lebih dari 5 proyek**.
# - **Insight**:
#   - Kebanyakan karyawan menangani 3–5 proyek.
#   - Jumlah proyek ekstrem (sedikit atau banyak) sangat jarang dan bisa menjadi faktor stres/resign yang perlu dianalisis lebih lanjut.
# 
# ---
# 
# 4. **Average Monthly Hours**
# - Distribusi **bimodal**:
#   - Puncak di sekitar **150 jam** dan **250 jam** per bulan.
# - Ada juga karyawan yang bekerja di atas **300 jam** per bulan, meskipun jumlahnya sedikit.
# - **Insight**:
#   - Ada dua pola kerja: yang cenderung normal dan yang bekerja sangat intens.
#   - Karyawan yang bekerja lebih dari 250 jam kemungkinan mengalami **overwork**, berpotensi meningkatkan risiko resign atau burnout.
# 
# ---
# 
# 5. **Time Spent in Company**
# - Mayoritas karyawan berada di:
#   - **2 tahun** dan **3 tahun**
# - Setelah itu jumlahnya turun drastis, kecuali ada lonjakan kecil di tahun ke-4 dan ke-5.
# - **Insight**:
#   - Karyawan cenderung keluar atau tidak bertahan lama setelah tahun ke-3.
#   - Daya tahan karyawan terhadap lingkungan kerja menurun seiring waktu, bisa jadi indikator kurangnya prospek atau kenaikan karier.
# 
# ---
# 
# **Kesimpulan Umum:**
# - Beberapa variabel seperti **satisfaction_level**, **average_monthly_hours**, dan **time_spend_company** menunjukkan distribusi yang bisa dikaitkan langsung dengan kemungkinan resign.
# - Fitur-fitur ini **sangat potensial** untuk dimasukkan dalam model prediksi karena menggambarkan kondisi kerja dan kepuasan secara langsung.

# %% [markdown]
# ### Exploratory Data Analysis - Multivariate Analysis

# %% [markdown]
# #### Analisis Left terhadap Data Kategori

# %%
cat_cols_filtered = [col for col in cat_cols if col != 'left']
# Atur tema warna hijau
sns.set_theme(style="whitegrid")

# Buat figure
n_cols = 2  # Tetapkan jumlah kolom tetap menjadi 2
n_rows = math.ceil(len(cat_cols_filtered) / n_cols)  # Hitung jumlah baris
plt.figure(figsize=(5 * n_cols, 4 * n_rows))

# Iterasi melalui setiap kolom kategori
for i, col in enumerate(cat_cols_filtered):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.countplot(data=df_cleaned, x=col, hue='left', palette='viridis', 
                  hue_order=[0, 1])  # Pastikan urutan hue sesuai dengan data
    plt.title(f'Perbandingan {col} terhadap left')
    plt.legend(title='Left', loc='upper right', labels=['tidak', 'iya'])  # Ubah label legenda
    plt.xticks(rotation=45)  # Rotasi label x sebesar 45 derajat

plt.tight_layout()
plt.show()

# %% [markdown]
# Berikut adalah **analisis dan insight** dari Multivariate Analysis pada keempat grafik bar (countplot). Masing-masing menunjukkan hubungan antara variabel kategori terhadap status resign (`left`):
# 
# 1. **Work Accident vs Left**
# - **Insight:** Pegawai yang **tidak mengalami kecelakaan kerja (work_accident = 0)** memiliki jumlah resign yang **jauh lebih tinggi** dibandingkan yang mengalami kecelakaan.
# - **Interpretasi:** Ini cukup menarik karena biasanya kita berpikir kecelakaan bisa membuat karyawan keluar, tapi justru yang **tidak mengalami kecelakaan lebih banyak resign**. Ini bisa jadi karena mereka kurang merasa “terikat” atau karena faktor lain seperti lingkungan kerja atau beban kerja.
# 
# 2. **Promotion in Last 5 Years vs Left**
# - **Insight:** Karyawan yang **tidak mendapat promosi dalam 5 tahun terakhir (promotion_last_5years = 0)** memiliki angka resign yang sangat tinggi.
# - **Interpretasi:** Ini konsisten dengan dugaan bahwa **kurangnya penghargaan atau perkembangan karir memicu resign**. Promosi tampaknya jadi faktor penting dalam retensi karyawan.
# 
# 3. **Department vs Left**
# - **Insight:** Departemen seperti **sales, technical, dan support** memiliki jumlah resign yang lebih tinggi secara absolut.
# - **Interpretasi:**
#   - **Sales:** Mungkin karena tekanan target yang tinggi.
#   - **Technical & Support:** Bisa jadi karena beban kerja atau kurangnya jenjang karir.
#   - Departemen seperti **management dan R&D** memiliki tingkat resign yang relatif rendah, menunjukkan stabilitas yang lebih tinggi.
# 
# 4. **Salary vs Left**
# - **Insight:** Mayoritas karyawan yang resign berasal dari kelompok gaji **rendah (low)**, diikuti oleh **medium**, dan hampir tidak ada yang dari kelompok **high salary**.
# - **Interpretasi:** Gaji jelas berpengaruh terhadap loyalitas. **Semakin rendah gaji, semakin tinggi potensi resign.** Ini bisa dijadikan pertimbangan dalam strategi kompensasi perusahaan.
# 
# **Kesimpulan Umum:**
# Variabel-variabel berikut punya **korelasi kuat dengan keputusan resign**:
# - Tidak mendapat promosi
# - Gaji rendah
# - Bekerja di departemen tertentu (sales, support, technical)
# - Tidak mengalami kecelakaan kerja (mungkin korelasi tidak langsung)

# %% [markdown]
# #### Analisis Left terhadap Data Numerik

# %%
# List fitur numerik yang ingin dibandingkan
features = ['satisfaction_level', 'last_evaluation', 'average_monthly_hours', 'time_spend_company', 'number_project']

plt.figure(figsize=(15, 20))

for i, feature in enumerate(features, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(x='left', y=feature, data=df_cleaned, palette='Greens')
    plt.title(f'Boxplot {feature} berdasarkan Status Resign')
    plt.xlabel('Resign (0 = Tidak, 1 = Iya)')
    plt.ylabel(feature)

plt.tight_layout()
plt.show()


# %% [markdown]
# Berikut adalah analisis dari lima **boxplot** yang membandingkan berbagai fitur numerik terhadap status **resign** (`left`: 0 = tidak resign, 1 = resign):
# 
# ---
# 
# 1. **Satisfaction Level**
# - **Insight**: Pegawai yang resign (1) cenderung memiliki tingkat kepuasan kerja yang jauh lebih rendah dibandingkan yang tidak resign (0).
# - **Interpretasi**: Kepuasan kerja rendah menjadi salah satu faktor utama seseorang memutuskan keluar dari perusahaan.
# 
# ---
# 
# 2. **Last Evaluation**
# - **Insight**: Pegawai yang resign memiliki nilai evaluasi yang sedikit lebih tinggi secara median dibandingkan yang tidak resign.
# - **Interpretasi**: Bisa jadi beberapa pegawai yang performanya tinggi merasa kurang dihargai atau tidak diberi tantangan/prospek karir, sehingga mereka memilih keluar.
# 
# ---
# 
# 3. **Average Monthly Hours**
# - **Insight**: Pegawai yang resign bekerja lebih lama per bulan (jam kerja lebih tinggi).
# - **Interpretasi**: Jam kerja yang terlalu tinggi bisa menyebabkan burnout dan meningkatkan kemungkinan resign.
# 
# ---
# 
# 4. **Time Spend Company**
# - **Insight**: Pegawai yang resign umumnya telah bekerja lebih lama (median sekitar 4 tahun), sedangkan yang bertahan banyak di kisaran 3 tahun.
# - **Interpretasi**: Masa kerja yang lama tanpa promosi atau perubahan kondisi kerja bisa menyebabkan kejenuhan, sehingga memicu resign.
# 
# ---
# 
# 5. **Number of Projects**
# - **Insight**: Pegawai yang resign cenderung memiliki jumlah proyek yang lebih beragam (dari yang sedikit hingga sangat banyak).
# - **Interpretasi**:
#   - Resign bisa dipicu karena **kurangnya proyek (underutilization)** → merasa tidak berkembang.
#   - Atau karena **terlalu banyak proyek (overload)** → burnout.
# 
# ---
# 
# **Kesimpulan Umum**
# Faktor-faktor yang paling berkorelasi dengan **resign** adalah:
# - **Kepuasan kerja rendah**
# - **Jam kerja yang tinggi**
# - **Lama masa kerja**
# - **Jumlah proyek terlalu sedikit atau terlalu banyak (kedua ekstrem)**

# %%
# Atur tema warna hijau
sns.set_theme(style="whitegrid")

# Buat figure dengan dua subplot
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Scatter plot untuk pegawai yang tidak resign (left = 0)
sns.scatterplot(data=df_cleaned[df_cleaned['left'] == 0], 
                x='average_monthly_hours', y='satisfaction_level', 
                color='limegreen', alpha=0.6, ax=axes[0])
axes[0].set_title('Pegawai yang Tidak Resign', fontsize=14)
axes[0].set_xlabel('Rata-rata Jam Kerja Bulanan', fontsize=12)
axes[0].set_ylabel('Tingkat Kepuasan', fontsize=12)

# Scatter plot untuk pegawai yang resign (left = 1)
sns.scatterplot(data=df_cleaned[df_cleaned['left'] == 1], 
                x='average_monthly_hours', y='satisfaction_level', 
                color='orange', alpha=0.6, ax=axes[1])
axes[1].set_title('Pegawai yang Resign', fontsize=14)
axes[1].set_xlabel('Rata-rata Jam Kerja Bulanan', fontsize=12)
axes[1].set_ylabel('')  # Tidak perlu label y di subplot kedua

# Tampilkan plot
plt.tight_layout()
plt.show()


# %% [markdown]
# Berikut adalah insight dari grafik scatter plot yang membandingkan **pegawai yang tidak resign** (kiri) dan **pegawai yang resign** (kanan), berdasarkan **rata-rata jam kerja bulanan** dan **tingkat kepuasan kerja**:
# 
# | Kelompok Pegawai         | Insight                                                                                             |
# |--------------------------|-----------------------------------------------------------------------------------------------------|
# | Pegawai yang Tidak Resign | - Penyebaran tingkat kepuasan cukup merata di berbagai jam kerja, menunjukkan variasi yang luas.    |
# |                           | - Kebanyakan memiliki kepuasan di atas 0.5 dan jam kerja bulanan antara 150-250 jam.                |
# |                           | - Tidak ada pola yang menunjukkan bahwa jam kerja tinggi atau rendah secara langsung mendorong ketidakresign-an. |
# | Pegawai yang Resign       | - Terlihat tiga kelompok dominan:                                                                  |
# |                           |   1. Tingkat kepuasan rendah (~0.1 - 0.4) & jam kerja sekitar 130-160 jam.                          |
# |                           |   2. Tingkat kepuasan tinggi (~0.7 - 1.0) & jam kerja tinggi (~225 - 275 jam).                      |
# |                           |   3. Tingkat kepuasan sangat rendah (~0.1) dengan jam kerja sangat tinggi (~250 - 310 jam).         |
# |                           | - Artinya, pegawai resign bisa terjadi pada dua ekstrem: overworked meskipun puas, atau sangat tidak puas meskipun tidak terlalu sibuk. |
# 
# **Insight:**
# 
# * Resign tidak selalu disebabkan oleh kepuasan rendah saja, tapi juga bisa karena **jam kerja yang terlalu tinggi**, bahkan untuk pegawai yang puas sekalipun.
# * Ada indikasi bahwa manajemen beban kerja dan distribusi proyek sangat penting untuk mencegah turnover.
# 

# %%
# Atur tema warna hijau
sns.set_theme(style="whitegrid")

# Buat figure dengan dua subplot
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Scatter plot untuk pegawai yang tidak resign (left = 0)
sns.scatterplot(data=df_cleaned[df_cleaned['left'] == 0], 
                x='average_monthly_hours', y='last_evaluation', 
                color='limegreen', alpha=0.6, ax=axes[0])
axes[0].set_title('Pegawai yang Tidak Resign', fontsize=14)
axes[0].set_xlabel('Rata-rata Jam Kerja Bulanan', fontsize=12)
axes[0].set_ylabel('Skor Evaluasi Terbaru pada Pegawai', fontsize=12)

# Scatter plot untuk pegawai yang resign (left = 1)
sns.scatterplot(data=df_cleaned[df_cleaned['left'] == 1], 
                x='average_monthly_hours', y='last_evaluation', 
                color='orange', alpha=0.6, ax=axes[1])
axes[1].set_title('Pegawai yang Resign', fontsize=14)
axes[1].set_xlabel('Skor Evaluasi Terbaru pada Pegawai', fontsize=12)
axes[1].set_ylabel('')  # Tidak perlu label y di subplot kedua

# Tampilkan plot
plt.tight_layout()
plt.show()


# %% [markdown]
# Berikut adalah hasil analisis dan insight dari grafik scatter plot yang memperlihatkan hubungan antara **`average_montly_hours`** dan **`last_evaluation`**, dibandingkan berdasarkan status resign (**`left`**):
# 
# | Kelompok Pegawai         | Insight                                                                                             |
# |--------------------------|-----------------------------------------------------------------------------------------------------|
# | Pegawai yang Tidak Resign | - Data tersebar merata di seluruh rentang evaluasi dan jam kerja.                                   |
# |                           | - Tidak ada pola yang jelas menunjukkan hubungan kuat antara evaluasi kerja dengan jam kerja bulanan. |
# |                           | - Artinya, baik evaluasi tinggi maupun rendah tidak terlalu mempengaruhi keputusan untuk tetap bekerja. |
# | Pegawai yang Resign       | - Terlihat dua kelompok mencolok:                                                                   |
# |                           |   1. Evaluasi rendah (~0.4–0.6) & jam kerja rendah (~130–160 jam): kemungkinan karena performa kurang. |
# |                           |   2. Evaluasi tinggi (~0.8–1.0) & jam kerja tinggi (~230–300 jam): kemungkinan karena overwork meskipun berkinerja tinggi. |
# |                           | - Sedikit pegawai resign dengan skor evaluasi sedang (~0.6–0.7), menunjukkan bahwa kelompok ini lebih stabil. |
# 
# **Insight:**
# 
# * Pegawai dengan evaluasi rendah dan jam kerja rendah cenderung resign, kemungkinan karena dianggap tidak perform.
# * Pegawai dengan evaluasi tinggi dan jam kerja tinggi juga cenderung resign, mungkin akibat kelelahan atau beban kerja berlebihan.
# * Pegawai yang "aman" dari risiko resign berada di tengah-tengah: evaluasi sedang & jam kerja seimbang.
# 

# %% [markdown]
# #### Analisis Left terhadap Data Numerik - Kategori

# %%
# Atur tema warna hijau
sns.set_theme(style="whitegrid")

# Buat figure dengan dua subplot
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Plot untuk pegawai yang resign (left = 'iya')
sns.countplot(data=df_cleaned[df_cleaned['left'] == 1], 
              x='time_spend_company', hue='salary', palette='viridis', 
              hue_order=['low', 'medium', 'high'], ax=axes[0])
axes[0].set_title('pegawai yang Resign', fontsize=14)
axes[0].set_xlabel('Lama Bekerja (time_spend_company)', fontsize=12)
axes[0].set_ylabel('Jumlah pegawai', fontsize=12)
axes[0].legend(title='Tingkat Gaji (salary)', loc='upper right')

# Plot untuk pegawai yang menetap (left = 'tidak')
sns.countplot(data=df_cleaned[df_cleaned['left'] == 0], 
              x='time_spend_company', hue='salary', palette='viridis', 
              hue_order=['low', 'medium', 'high'], ax=axes[1])
axes[1].set_title('pegawai yang Menetap', fontsize=14)
axes[1].set_xlabel('Lama Bekerja (time_spend_company)', fontsize=12)
axes[1].set_ylabel('')  # Tidak perlu label y di subplot kedua
axes[1].legend(title='Tingkat Gaji (salary)', loc='upper right')

# Tampilkan plot
plt.tight_layout()
plt.show()

# %% [markdown]
# **Visualisasi Perbandingan: Pegawai yang Resign vs Menetap berdasarkan Lama Bekerja dan Tingkat Gaji**
# 
# * **Sumbu X:** Lama bekerja di perusahaan (`time_spend_company`)
# * **Sumbu Y:** Jumlah pegawai
# * **Warna Batang:**
# 
#   * **Biru tua:** Gaji rendah (`low`)
#   * **Hijau tua:** Gaji sedang (`medium`)
#   * **Hijau terang:** Gaji tinggi (`high`)
# * Dibagi menjadi dua panel:
# 
#   * **Kiri:** Pegawai yang *resign*
#   * **Kanan:** Pegawai yang *menetap*
# 
# **Insight dari Pegawai yang Resign:**
# 
# 1. **Mayoritas yang resign berada di level gaji rendah**, khususnya:
# 
#    * Tahun ke-3 paling tinggi → Pegawai yang bekerja 3 tahun dengan gaji rendah paling banyak resign.
#    * Disusul tahun ke-4 dan ke-5, juga didominasi gaji rendah.
#    * Hampir tidak ada pegawai bergaji tinggi yang resign.
# 
# 2. **Resign paling jarang terjadi di tahun ke-2 atau ke-6**, apalagi dengan gaji tinggi.
# 
# **Insight dari Pegawai yang Menetap:**
# 
# 1. **Tingkat retensi paling tinggi terjadi pada pegawai yang telah bekerja selama 3 tahun**, terutama dengan:
# 
#    * Gaji rendah dan sedang.
#    * Menunjukkan bahwa setelah 3 tahun, sebagian besar pegawai tetap bertahan.
# 
# 2. **Pegawai dengan gaji tinggi memiliki kecenderungan lebih besar untuk bertahan**, meskipun jumlahnya tidak sebanyak pegawai dengan gaji rendah/menengah.
# 
# 3. **Mereka yang sudah bekerja 6 tahun ke atas (hingga 10 tahun)** cenderung tetap tinggal di perusahaan, walaupun jumlahnya kecil.
# 
# **Kesimpulan:**
# 
# * Fokus pada pegawai **gaji rendah dan sedang** yang telah bekerja **3–5 tahun**, karena mereka paling rentan resign.
# * Pertimbangkan peningkatan gaji atau insentif setelah masa kerja 2–3 tahun sebagai bentuk retensi.
# * Pegawai **bergaji tinggi jarang resign**, jadi mempertahankan dan memotivasi mereka penting untuk jangka panjang.
# * Buat program loyalitas atau kenaikan jenjang karier yang dimulai sejak tahun ke-2 atau ke-3.
# * Evaluasi apakah gaji rendah di tahun-tahun krusial (3–5 tahun) memicu rasa tidak dihargai hingga pegawai memilih keluar.
# 
# 

# %%
# Atur tema warna hijau
sns.set_theme(style="whitegrid")

# Buat figure dengan dua subplot
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Scatter plot untuk pegawai yang tidak resign (left = 0)
sns.scatterplot(data=df_cleaned[df_cleaned['left'] == 0], 
                x='average_monthly_hours', y='promotion_last_5years', 
                color='limegreen', alpha=0.6, ax=axes[0])
axes[0].set_title('Pegawai yang Tidak Resign', fontsize=14)
axes[0].set_xlabel('Rata-rata Jam Kerja Bulanan', fontsize=12)
axes[0].set_ylabel('Mendapatkan Promosi 5 Tahun Terakhir', fontsize=12)

# Scatter plot untuk pegawai yang resign (left = 1)
sns.scatterplot(data=df_cleaned[df_cleaned['left'] == 1], 
                x='average_monthly_hours', y='promotion_last_5years', 
                color='orange', alpha=0.6, ax=axes[1])
axes[1].set_title('Pegawai yang Resign', fontsize=14)
axes[1].set_xlabel('Rata-rata Jam Kerja Bulanan', fontsize=12)
axes[1].set_ylabel('')  # Tidak perlu label y di subplot kedua

# Tampilkan plot
plt.tight_layout()
plt.show()


# %% [markdown]
# Berikut adalah hasil **analisis** dan **insight** dari grafik scatter plot yang menunjukkan hubungan antara **`rata-rata jam kerja bulanan`** dan status **`mendapatkan promosi dalam 5 tahun terakhir`**, dipisahkan berdasarkan status resign:
# 
# **Pegawai yang Tidak Resign**
# 
# * Terlihat cukup banyak pegawai **yang tidak mendapatkan promosi (nilai = 0)** namun **tetap bertahan** di perusahaan.
# * Pegawai yang **mendapat promosi (nilai = 1)** juga cukup banyak dan tersebar merata dalam rentang jam kerja antara \~100 sampai \~280 jam per bulan.
# * Artinya, **promosi mungkin berperan dalam menjaga loyalitas**, tapi **bukan satu-satunya faktor** yang membuat pegawai tetap tinggal.
# 
# **Pegawai yang Resign**
# 
# * Mayoritas **pegawai yang resign tidak mendapatkan promosi** (nilai = 0).
# * Hanya **beberapa** pegawai yang resign **pernah mendapatkan promosi**, dan mereka tersebar di jam kerja rendah hingga tinggi (\~130–290 jam/bulan).
# * Ini mengindikasikan bahwa:
# 
#   * **Kurangnya promosi** bisa menjadi salah satu faktor penyebab resign.
#   * **Promosi tidak menjamin pegawai bertahan**, terutama jika dikombinasikan dengan **beban kerja yang tinggi** atau **faktor lain** seperti kepuasan kerja, kompensasi, atau keseimbangan kerja-hidup.
# 
# **Insight**
# 
# * Pegawai yang tidak dipromosikan lebih rentan untuk resign, khususnya mereka dengan beban kerja tinggi.
# * Pegawai yang mendapatkan promosi cenderung bertahan, namun bukan berarti mereka tidak bisa resign. Ada kemungkinan bahwa **pegawai berprestasi tetap resign jika merasa tidak puas secara keseluruhan**.
# * **Strategi retensi** tidak cukup hanya dengan promosi. Perusahaan juga perlu **mengatur beban kerja** dan **menciptakan lingkungan kerja yang suportif**.
# 
# 

# %% [markdown]
# #### Analisis Matriks Korelasi

# %%
df_cleaned.head()

# %%
# Data Encoding
# Salin dataframe
df_encoded = df_cleaned.copy()

# Ordinal encoding untuk salary
salary_order = ['low', 'medium', 'high']
ordinal_enc = OrdinalEncoder(categories=[salary_order])
df_encoded['salary'] = ordinal_enc.fit_transform(df_cleaned[['salary']])

# One-hot encoding untuk department
df_encoded = pd.get_dummies(df_encoded, columns=['department'], drop_first=True)

# Optional: ubah kategori lain ke integer (jika belum)
for col in ['work_accident', 'promotion_last_5years', 'left']:
    df_encoded[col] = df_encoded[col].astype(int)

df_encoded.head()

# %%
# Hitung korelasi
correlation_matrix = df_encoded.corr(numeric_only=True)

# Set tema warna hijau
plt.figure(figsize=(16, 10))
sns.heatmap(correlation_matrix, 
            annot=True, 
            fmt=".2f", 
            cmap="Greens", 
            linewidths=0.5, 
            linecolor='gray')

plt.title("Matriks Korelasi Fitur", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# %% [markdown]
# Berikut beberapa insight penting terhadap `left`:
# 
# | Fitur                     | Korelasi terhadap `left` | Keterangan                                                                 |
# |--------------------------|--------------------------|----------------------------------------------------------------------------|
# | `satisfaction_level`     | **-0.35**                | Korelasi negatif kuat → makin puas, makin kecil kemungkinan resign         |
# | `time_spend_company`     | 0.17                     | Korelasi positif ringan → makin lama di perusahaan, sedikit lebih rentan resign |
# | `number_project`         | 0.03                     | Korelasi sangat kecil                                                      |
# | `average_monthly_hours`  | 0.07                     | Korelasi sangat kecil                                                      |
# | `last_evaluation`        | 0.01                     | Hampir tidak berkorelasi                                                   |
# | `promotion_last_5years` | **-0.13**                | Korelasi negatif → yang pernah dipromosi, cenderung tidak resign           |
# | `work_accident`          | **-0.13**                | Korelasi negatif → yang pernah kecelakaan kerja, cenderung bertahan        |
# | `salary`                 | -0.12                    | Korelasi negatif → makin tinggi gaji, makin kecil kemungkinan resign        |
# 
# ---
# 
# **🟩 Interpretasi Singkat:**
# 
# - **Fitur paling penting terhadap `left`** berdasarkan korelasi:  
#   ✅ `satisfaction_level`  
#   ✅ `promotion_last_5years`  
#   ✅ `work_accident`  
#   ✅ `salary`
# 

# %% [markdown]
# ### 📌 Insight dari **Exploratory Data Analysis**
# 
# 1. **Data Cleaning**
# - **Standardisasi Nama Kolom**: Nama kolom diubah menjadi huruf kecil dan typo diperbaiki (contoh: `average_montly_hours` → `average_monthly_hours`).
# - **Missing Values**: Tidak ada nilai yang hilang dalam dataset.
# - **Data Duplikat**: Ditemukan 3008 baris duplikat (20,05%) yang dihapus, menyisakan 11.991 baris data.
# - **Outliers**: Terdapat outliers pada `time_spend_company`, tetapi tidak ekstrem sehingga dibiarkan untuk menjaga pola data.
# 
# ---
# 
# 2. **Univariate Analysis**
# - **Categorical Features**:
#   - **`left`**: 16.6% karyawan resign, mayoritas (83.4%) tetap bekerja.
#   - **`work_accident`**: 84.6% tidak pernah mengalami kecelakaan kerja.
#   - **`promotion_last_5years`**: Hanya 1.7% karyawan yang pernah dipromosikan.
#   - **`department`**: Divisi terbesar adalah **sales**, diikuti **technical** dan **support**. Divisi **management** memiliki jumlah karyawan paling sedikit.
#   - **`salary`**: Mayoritas karyawan memiliki gaji rendah (47.9%) atau menengah (43.9%), hanya 8.3% bergaji tinggi.
# 
# - **Numerical Features**:
#   - **`satisfaction_level`**: Distribusi bimodal, dengan kelompok puas tinggi (~0.7–0.8) dan rendah (~0.4–0.5).
#   - **`last_evaluation`**: Distribusi merata, dengan puncak kecil di ~0.55–0.6.
#   - **`number_projects`**: Mayoritas menangani 3–5 proyek.
#   - **`average_monthly_hours`**: Distribusi bimodal (~150 jam dan ~250 jam).
#   - **`time_spend_company`**: Mayoritas bekerja 2–3 tahun, dengan lonjakan kecil di tahun ke-4 dan ke-5.
# 
# ---
# 
# 3. **Multivariate Analysis**
# - **Categorical Features vs `left`**:
#   - **`work_accident`**: Karyawan yang tidak mengalami kecelakaan lebih banyak resign.
#   - **`promotion_last_5years`**: Karyawan yang tidak dipromosikan lebih banyak resign.
#   - **`department`**: Divisi **sales**, **technical**, dan **support** memiliki jumlah resign tertinggi.
#   - **`salary`**: Resign lebih banyak terjadi pada karyawan bergaji rendah.
# 
# - **Numerical Features vs `left`**:
#   - **`satisfaction_level`**: Karyawan yang resign memiliki tingkat kepuasan lebih rendah.
#   - **`last_evaluation`**: Karyawan yang resign memiliki evaluasi sedikit lebih tinggi.
#   - **`average_monthly_hours`**: Karyawan yang resign bekerja lebih lama per bulan.
#   - **`time_spend_company`**: Resign lebih banyak terjadi pada karyawan dengan masa kerja 3–5 tahun.
#   - **`number_projects`**: Resign terjadi pada karyawan dengan proyek terlalu sedikit atau terlalu banyak.
# 
# - **Scatterplot Insight**:
#   - **`satisfaction_level` vs `average_monthly_hours`**: Resign terjadi pada karyawan dengan kepuasan rendah, baik dengan jam kerja normal (~130–160 jam) maupun tinggi (~250–310 jam).
#   - **`last_evaluation` vs `average_monthly_hours`**: Resign terjadi pada karyawan dengan evaluasi rendah (~0.45–0.6) dan jam kerja normal, atau evaluasi tinggi (>0.8) dan jam kerja tinggi (>250 jam).
#   - **`time_spend_company` vs `salary`**: Resign paling banyak terjadi pada karyawan bergaji rendah dengan masa kerja 3–5 tahun.
# 
# ---
# 
# 4. **Correlation Analysis**
# - **Fitur dengan Korelasi Tinggi terhadap `left`**:
#   - **Negatif**: `satisfaction_level` (-0.35), `promotion_last_5years` (-0.13), `work_accident` (-0.13), `salary` (-0.12).
#   - **Positif**: `time_spend_company` (0.17).
# - **Fitur seperti `number_project` dan `last_evaluation` memiliki korelasi sangat kecil.**
# 
# ---
# 
# **Kesimpulan Strategis**
# 1. **Faktor Utama Resign**:
#    - Kepuasan kerja rendah.
#    - Tidak mendapat promosi.
#    - Gaji rendah.
#    - Jam kerja tinggi (burnout).
#    - Lama bekerja (3–5 tahun).
# 
# **Rekomendasi**:
#    - Tingkatkan kepuasan kerja melalui insentif, promosi, dan pengakuan.
#    - Fokus pada retensi karyawan bergaji rendah dengan masa kerja 3–5 tahun.
#    - Hindari overload pekerjaan pada karyawan berkinerja tinggi.

# %% [markdown]
# ## Data Preparation

# %% [markdown]
# ### Train-Test Split

# %%
# 1. Pisahkan fitur (X) dan target (y)
X = df_encoded.drop(columns='left')
y = df_encoded['left']

# 2. Lakukan pembagian data (misalnya 80% untuk train dan 20% untuk test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# %%
X_train.head()

# %%
y_train.value_counts()

# %%
portion_train = X_train.shape[0] / X.shape[0]
portion_test = X_test.shape[0] / X.shape[0]

print(f'shape of X_train: {X_train.shape}')
print(f'shape of y_train: {y_train.shape}')
print(f'shape of X_test: {X_test.shape}')
print(f'shape of y_test: {y_test.shape}')
print(f'percentage of train size: {portion_train:.2f}%')
print(f'percentage of test size: {portion_test:.2f}%')

# %% [markdown]
# ## Modelling

# %% [markdown]
# ### Base Model Development

# %%
nb = GaussianNB()
rf = RandomForestClassifier(random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

nb.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

classification_models = [nb, rf, xgb]
classification_models


# %% [markdown]
# ### Base Model Evaluation

# %%
# Definisi fungsi evaluasi model
def models_evaluation(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted')
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    return accuracy, f1, cm

# Inisialisasi list untuk menyimpan hasil evaluasi
model_names = []
accuracies_train = []
f1_scores_train = []
cms_train = []

accuracies_test = []
f1_scores_test = []
cms_test = []

# Evaluasi untuk setiap model
for model in classification_models:
    # Evaluasi pada data train
    accuracy_train, f1_train, cm_train = models_evaluation(model, X_train, y_train)
    accuracies_train.append(accuracy_train)
    f1_scores_train.append(f1_train)
    cms_train.append(cm_train)
    
    # Evaluasi pada data test
    accuracy_test, f1_test, cm_test = models_evaluation(model, X_test, y_test)
    accuracies_test.append(accuracy_test)
    f1_scores_test.append(f1_test)
    cms_test.append(cm_test)
    
    # Nama model
    model_names.append(model.__class__.__name__)

# Menampilkan hasil evaluasi dalam tabel
evaluations = pd.DataFrame({
    'Model': model_names,
    'Train Accuracy': accuracies_train,
    'Train F1 Score': f1_scores_train,
    'Test Accuracy': accuracies_test,
    'Test F1 Score': f1_scores_test,
})

display(evaluations)  # Pastikan tabel muncul di Jupyter Notebook

# %%
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Buat objek StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Buat dictionary model
models_cv = {
    "GaussianNB": nb,
    "RandomForest": rf,
    "XGBoost": xgb
}

# Evaluasi cross-validation (scoring pakai 'accuracy' atau 'f1_weighted')
for name, model in models_cv.items():
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')  # bisa diganti jadi 'f1_weighted'
    print(f"{name} - Mean Accuracy: {scores.mean():.4f}, Std Dev: {scores.std():.4f}")


# %% [markdown]
# Dari ketiga model yang dibangung, dapat dilihat RandomForestClassifier memiliki nilai akurasi yang lebih tinggi. Berdasarkan nilai Cross-Validation pun, model RandomForestClassifier memiliki nilai mean accuracy yang paling tinggi, dan memiliki stabilitas yang lebih tinggi karena std dev lebih kecil, yaitu 0.0018. Namun masih ada terdapat indikasi overfitting, walau memang selisih nilai akurasi pada training set dan test set tidaklah banyak. Jadi saya akan memilih model RandomForestClassifier untuk dilakukan Hyperparameter Tuning dengan tujuan agar gap atau selisih antara nilai akurasi training dan akurasi tes berkurang.

# %% [markdown]
# ### Model Hyperparameter Tuning

# %%
# Definisi ruang pencarian hyperparameter untuk setiap model
search_spaces = {
    'RandomForestClassifier': {
        'n_estimators': (10, 200),
        'max_depth': (1, 50),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 20)
    }
}

# Inisialisasi hasil tuning
tuned_models = {}

# Cross-validation dengan StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lakukan tuning untuk setiap model
for model in classification_models:
    model_name = model.__class__.__name__
    if model_name in search_spaces:
        print(f"Tuning hyperparameter untuk {model_name}...")
        bayes_search = BayesSearchCV(
            estimator=model,
            search_spaces=search_spaces[model_name],
            n_iter=30,  # Jumlah iterasi pencarian
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )
        bayes_search.fit(X_train, y_train)
        tuned_models[model_name] = bayes_search.best_estimator_
        print(f"Best parameters for {model_name}: {bayes_search.best_params_}")
        print(f"Best score for {model_name}: {bayes_search.best_score_:.4f}\n")

# %% [markdown]
# ### Tuned Model Evaluation

# %%
# Evaluasi ulang model klasifikasi setelah tuning
tuned_model_names = []
tuned_accuracies_train = []
tuned_f1_scores_train = []
tuned_accuracies_test = []
tuned_f1_scores_test = []
tuned_cms = []

rf_before = rf
train_accuracy_before = rf_before.score(X_train, y_train)
test_accuracy_before = rf_before.score(X_test, y_test)

for model_name, tuned_model in tuned_models.items():
    # Ubah nama model RandomForestClassifier setelah tuning
    if model_name == "RandomForestClassifier":
        model_name = "RandomForestClassifier (Setelah Tuning)"
    
    # Evaluasi pada training set
    accuracy_train, f1_train, _ = models_evaluation(tuned_model, X_train, y_train)
    tuned_accuracies_train.append(accuracy_train)
    tuned_f1_scores_train.append(f1_train)
    
    # Evaluasi pada test set
    accuracy_test, f1_test, cm_test = models_evaluation(tuned_model, X_test, y_test)
    tuned_accuracies_test.append(accuracy_test)
    tuned_f1_scores_test.append(f1_test)
    tuned_cms.append(cm_test)
    
    # Nama model
    tuned_model_names.append(model_name)

# Menampilkan hasil evaluasi setelah tuning dalam tabel
tuned_evaluations = pd.DataFrame({
    'Model': tuned_model_names,
    'Train Accuracy': tuned_accuracies_train,
    'Train F1 Score': tuned_f1_scores_train,
    'Test Accuracy': tuned_accuracies_test,
    'Test F1 Score': tuned_f1_scores_test
})

# Tambahkan evaluasi model sebelum tuning ke dalam tabel
rf_before_tuning = {
    'Model': 'RandomForestClassifier (Sebelum Tuning)',  # Nama model sebelum tuning
    'Train Accuracy': accuracies_train[1],  # Akurasi training sebelum tuning
    'Train F1 Score': f1_scores_train[1],   # F1 Score training sebelum tuning
    'Test Accuracy': accuracies_test[1],    # Akurasi testing sebelum tuning
    'Test F1 Score': f1_scores_test[1],     # F1 Score testing sebelum tuning
}

# Tambahkan ke tabel evaluasi setelah tuning
tuned_evaluations = pd.concat([
    pd.DataFrame([rf_before_tuning]),  # Data sebelum tuning
    tuned_evaluations                  # Data setelah tuning
], ignore_index=True)

# Tampilkan tabel evaluasi yang diperbarui
display(tuned_evaluations)

y_pred_before = rf_before.predict(X_test)
cm_before = confusion_matrix(y_test, y_pred_before)

# Menentukan ukuran grid subplot untuk confusion matrix sebelum dan sesudah tuning
fig, axes = plt.subplots(figsize=(15, 6), nrows=1, ncols=2)

# Confusion matrix sebelum tuning (warna hijau)
sns.heatmap(cm_before, annot=True, fmt='d', ax=axes[0], cmap="Greens", 
            xticklabels=["tidak resign", "resign"], 
            yticklabels=["tidak resign", "resign"])
axes[0].set_title("RandomForestClassifier (Sebelum Tuning)", fontsize=12)

# Confusion matrix sesudah tuning (warna hijau)
sns.heatmap(tuned_cms[0], annot=True, fmt='d', ax=axes[1], cmap="Greens", 
            xticklabels=["tidak resign", "resign"], 
            yticklabels=["tidak resign", "resign"])
axes[1].set_title("RandomForestClassifier (Setelah Tuning)", fontsize=12)

plt.tight_layout()
plt.show()


# %%
# Prediksi pada data test
y_pred_tuned = tuned_models['RandomForestClassifier'].predict(X_test)

# Buat classification report
report = classification_report(y_test, y_pred_tuned, target_names=['Tidak Resign', 'Resign'])
print(report)

# %%
# Identifikasi underfitting atau overfitting dengan Learning Curve untuk model sebelum tuning
train_sizes_before, train_scores_before, test_scores_before = learning_curve(
    rf_before, X_train, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

train_mean_before = np.mean(train_scores_before, axis=1)
test_mean_before = np.mean(test_scores_before, axis=1)

# Identifikasi underfitting atau overfitting dengan Learning Curve untuk model setelah tuning
train_sizes_after, train_scores_after, test_scores_after = learning_curve(
    tuned_models['RandomForestClassifier'], X_train, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

train_mean_after = np.mean(train_scores_after, axis=1)
test_mean_after = np.mean(test_scores_after, axis=1)

# Plot perbandingan Learning Curve dengan warna kontras untuk sebelum tuning dan hijau untuk sesudah tuning
# Plot perbandingan Learning Curve dengan tema hijau dan ungu
plt.figure(figsize=(12, 8))

# Sebelum tuning - Ungu
plt.plot(train_sizes_before, train_mean_before, 
         label="Training Score (Sebelum Tuning)", 
         marker='o', linestyle='--', color='#7a68a6')  # Ungu muda
plt.plot(train_sizes_before, test_mean_before, 
         label="Cross-Validation Score (Sebelum Tuning)", 
         marker='o', linestyle='--', color='#54278f')  # Ungu tua

# Setelah tuning - Hijau
plt.plot(train_sizes_after, train_mean_after, 
         label="Training Score (Setelah Tuning)", 
         marker='o', linestyle='-', color='#2ca25f')  # Hijau muda
plt.plot(train_sizes_after, test_mean_after, 
         label="Cross-Validation Score (Setelah Tuning)", 
         marker='o', linestyle='-', color='#006d2c')  # Hijau tua

# Tambahkan judul dan label
plt.title("Perbandingan Learning Curve - RandomForestClassifier\nSebelum dan Setelah Tuning", fontsize=14)
plt.xlabel("Training Set Size", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# 
# ### Insight dari Tahap Modelling
# 
# 1. **Base Model Development**:
#     - Tiga model yang digunakan: `GaussianNB`, `RandomForestClassifier`, dan `XGBClassifier`.
#     - Model `RandomForestClassifier` menunjukkan performa terbaik dengan akurasi dan F1 score yang tinggi pada data train dan test.
# 
# 2. **Base Model Evaluation**:
#     - `RandomForestClassifier` memiliki akurasi test sebesar **98.58%** dan F1 score test sebesar **98.56%**.
#     - Namun, terdapat indikasi **overfitting** karena akurasi train mencapai **100%**.
# 
# 3. **Hyperparameter Tuning**:
#     - Dilakukan tuning pada `RandomForestClassifier` menggunakan `BayesSearchCV`.
#     - Parameter terbaik: `max_depth=21`, `min_samples_split=3`, `n_estimators=200`.
# 
# 4. **Tuned Model Evaluation**:
#     - Setelah tuning, akurasi test meningkat menjadi **98.62%** dan F1 score test menjadi **98.60%**.
#     - Overfitting berkurang dengan akurasi train turun menjadi **99.71%**.
# 
# 5. **Kesimpulan**:
#     - Model `RandomForestClassifier` dengan hyperparameter tuning memberikan performa terbaik dan lebih stabil.
#     - Model ini dipilih untuk digunakan dalam prediksi karena memiliki keseimbangan antara akurasi tinggi dan risiko overfitting yang rendah.

# %% [markdown]
# ## Model Interpretation

# %%
# IMPORT LIBRARY
importances_rf = tuned_models['RandomForestClassifier'].feature_importances_
importances_xgb = xgb.feature_importances_

# Buat DataFrame untuk mempermudah visualisasi
features = X_train.columns

# DataFrame untuk Random Forest
feature_importance_rf = pd.DataFrame({
    'Feature': features,
    'Importance': importances_rf
}).sort_values(by='Importance', ascending=False)


# Plot Random Forest
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_rf, x='Importance', y='Feature', palette='Greens_d')
plt.title('Feature Importance - Random Forest')
plt.tight_layout()
plt.show()

# %% [markdown]
# 1. 🎯 **Fokus utama prediksi adalah pada perilaku dan performa karyawan**
# 
# * Fitur yang berkaitan langsung dengan **produktivitas, keterlibatan, dan kepuasan kerja** sangat dominan:
# 
#   * `satisfaction_level`, `number_project`, `average_monthly_hours`, `last_evaluation`.
# * Ini mengindikasikan bahwa **indikator psikologis dan beban kerja** sangat memengaruhi keputusan karyawan untuk bertahan atau keluar.
# 
# 2. 💼 **Fitur demografis & departemen tidak terlalu penting**
# 
# * Fitur-fitur seperti `department_*`, `promotion_last_5years`, dan `salary` punya pengaruh rendah.
# * Mungkin karena:
# 
#   * Distribusinya tidak terlalu bervariasi, atau
#   * Tidak berkontribusi langsung dalam membedakan antara karyawan yang resign dan yang bertahan.

# %% [markdown]
# ## Insight dan Rekomendasi

# %% [markdown]
# ### Insight
# 1. **Faktor Utama Resign**:
#     - **Kepuasan Kerja Rendah**: Pegawai dengan tingkat kepuasan rendah memiliki kemungkinan resign yang tinggi.
#     - **Jam Kerja Tinggi**: Pegawai dengan jam kerja bulanan tinggi (>250 jam) cenderung resign karena burnout.
#     - **Kurangnya Promosi**: Pegawai yang tidak dipromosikan dalam 5 tahun terakhir lebih rentan resign.
#     - **Gaji Rendah**: Pegawai dengan gaji rendah memiliki tingkat resign yang jauh lebih tinggi dibandingkan dengan gaji menengah atau tinggi.
#     - **Masa Kerja 3–5 Tahun**: Pegawai dengan masa kerja 3–5 tahun memiliki risiko resign tertinggi.
# 
# 2. **Departemen dengan Tingkat Resign Tinggi**:
#     - **Sales**, **Technical**, dan **Support** memiliki jumlah resign tertinggi, kemungkinan karena tekanan kerja atau kurangnya jenjang karir.
# 
# 3. **Model Prediksi**:
#     - Model **RandomForestClassifier** dengan hyperparameter tuning memberikan akurasi tinggi (98.62%) dan dapat digunakan untuk memprediksi pegawai yang berisiko resign.
# 
# ---
# 
# ### Rekomendasi
# 1. **Tingkatkan Kepuasan Kerja**:
#     - Berikan program kesejahteraan, pelatihan, dan pengakuan untuk meningkatkan kepuasan kerja.
#     - Fokus pada pegawai dengan kepuasan rendah untuk mencegah resign.
# 
# 2. **Manajemen Beban Kerja**:
#     - Kurangi jam kerja berlebih (>250 jam) untuk menghindari burnout.
#     - Distribusikan proyek secara merata untuk menghindari underutilization atau overload.
# 
# 3. **Promosi dan Penghargaan**:
#     - Tingkatkan peluang promosi, terutama untuk pegawai dengan masa kerja 3–5 tahun.
#     - Berikan insentif atau bonus untuk pegawai yang berprestasi.
# 
# 4. **Kompensasi yang Kompetitif**:
#     - Evaluasi ulang struktur gaji, terutama untuk pegawai dengan gaji rendah.
#     - Berikan kenaikan gaji atau insentif tambahan untuk meningkatkan loyalitas.
# 
# 5. **Fokus pada Departemen Rentan**:
#     - Lakukan evaluasi mendalam pada departemen **Sales**, **Technical**, dan **Support** untuk mengidentifikasi penyebab resign.
#     - Berikan pelatihan atau program pengembangan karir untuk meningkatkan retensi di departemen tersebut.
# 
# Dengan langkah-langkah ini, perusahaan dapat meningkatkan retensi karyawan, mengurangi biaya turnover, dan menciptakan lingkungan kerja yang lebih produktif.


