# Laporan Proyek Machine Learning - Andrew Jonatan Damanik

## 1. Domain Proyek

Dalam dunia bisnis, memahami perilaku karyawan adalah salah satu aspek penting untuk meningkatkan efisiensi dan produktivitas perusahaan [1]. Penelitian ini berfokus pada analisis data karyawan untuk memprediksi apakah seorang karyawan akan berhenti bekerja (resign) atau tetap bekerja di perusahaan. Dataset yang digunakan dalam penelitian ini berasal dari Kaggle dengan nama "HR Analytics and Job Prediction" [2]. Dataset ini mencakup 14.999 baris data dengan berbagai atribut yang relevan, seperti tingkat kepuasan kerja, evaluasi terakhir, jumlah proyek, rata-rata jam kerja bulanan, dan lainnya.

Masalah ini penting untuk diselesaikan karena tingkat turnover karyawan yang tinggi dapat menyebabkan kerugian finansial dan operasional bagi perusahaan [3]. Dengan menggunakan pendekatan machine learning, perusahaan dapat mengidentifikasi faktor-faktor yang memengaruhi keputusan resign dan mengambil langkah-langkah preventif untuk meningkatkan retensi karyawan.

## 2. Business Understanding

### 2.1. Problem Statements

Penelitian ini bertujuan untuk menjawab beberapa pertanyaan utama:
1. Faktor apa saja yang memengaruhi keputusan karyawan untuk resign?
2. Dapatkah model machine learning yang digunakan memprediksi status resign karyawan dengan akurasi tinggi?

### 2.2. Goals

Tujuan dari penelitian ini adalah:
1. Mengidentifikasi faktor-faktor utama yang memengaruhi keputusan resign karyawan.
2. Mengembangkan model prediksi yang dapat memprediksi status resign karyawan dengan akurasi tinggi.
3. Memberikan rekomendasi berbasis data untuk meningkatkan retensi karyawan.

### 2.3. Solution Statements

- Melakukan Exploratory Data Analysis (EDA) untuk memahami pola dan hubungan antar fitur, serta mengidentifikasi fitur yang paling memengaruhi pegawai untuk resign menggunakan analisis korelasi, dan visualisasi.
- Untuk prediksi, akan membangun dan membandingkan beberapa algoritma machine learning untuk klasifikasi, yaitu Naive Bayes, Random Forest Classifier, dan XGBoost. Ketiga model ini dipilih karena kemampuannya dalam menangani data dengan fitur yang kompleks, dan tidak terlalu membutuhkan scaling pada datasetnya. Setelah itu, akan dipilih satu model untuk dilakukan hyperparameter tuning untuk meningkatkan performa model. 
- Mengevaluasi model menggunakan metrik evaluasi klasifikasi seperti akurasi, f1-score, precission, recall, cross validation, dan learning curve.
- Melakukan interpretasi model untuk menganalisis hasil model dengan metode Features Importance

## 3. Data Understanding

Dataset yang digunakan merupakan dataset sekunder yang didapat dari situs Kaggle dengan nama [Hr Analytics Job Prediction](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction/data). Dataset ini terdiri dari 14999 baris dan 10 fitur. Fitur tersebut yaitu:
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

### 3.1. Deskripsi Variabel

Adapun rangkuman tipe data dari dataset ini dapat dilihat pada gambar berikut

| ![Screenshot (144)](https://github.com/user-attachments/assets/88fa8b9b-fb37-4a76-af96-413a526a4bb1) | 
|:--:| 
| *Rangkuman Tipe Data pada Dataset Awal* |

Data didominasi oleh tipe data numerik. Peneliti melihat beberapa penamaan label yang perlu diperbaiki, seperti `Work_accident`, `average_montly_hours`, dan `Department`.

Untuk rangkuman dekskripsi statistik dari datasetnya dapat dilihat dari gambar berikut

| ![Screenshot (144)](https://github.com/user-attachments/assets/514a7c89-1518-4f88-ac81-734c7b56b3bb) | 
|:--:| 
| *Rangkuman Dekskripsi Statistik Dataset Awal* |

Berdasarkan gambar Rangkuman Dekskripsi Statistik Dataset Awal di atas, peneliti menyimpulkan beberapa hal, yaitu:

| Fitur                    | Insight                                                                                  |
|--------------------------|------------------------------------------------------------------------------------------|
| **Satisfaction Level**   | Rata-rata 0.61, distribusi sedang. Ada karyawan sangat tidak puas (0.09) hingga sangat puas (1.0). |
| **Last Evaluation**      | Rata-rata 0.72, menunjukkan sebagian besar memiliki evaluasi baik. Distribusinya cenderung normal. |
| **Number of Projects**   | Rata-rata 3.8 proyek. Kisaran 2–7 proyek. Perlu analisis apakah beban kerja berlebih memicu churn. |
| **Average Monthly Hours**| Rata-rata 201 jam/bulan. Beberapa bekerja sangat banyak (hingga 310 jam/bulan). Indikasi overwork. |
| **Time Spent at Company**| Rata-rata 3.5 tahun. Sebagian besar berada di 2–4 tahun masa kerja.                        |
| **Work Accident**        | 14% karyawan mengalami kecelakaan kerja. Mayoritas tidak pernah mengalami kecelakaan.     |
| **Left (Turnover)**      | 24% karyawan telah keluar dari perusahaan. Bisa menjadi target variabel dalam prediksi churn. |
| **Promotion (5 years)**  | Hanya 2.1% yang mendapat promosi dalam 5 tahun. Menunjukkan peluang promosi sangat kecil.  |


## 4. Exploratory Data Analysis (EDA)

Pada tahap ini, peneliti melakukan ekplorasi pada dataset untuk memahami distribusi data dan hubungan antar fitur. 

### 4.1. Data Cleaning
Sebelum melakukan analisis data, peneliti melakukan pengecekan kebersihan data terlebih dahulu. Seperti mengubah nama kolom, mengecek missing values, mengecek data duplikat, mengecek tipe data yang tidak tepat, dan mengecek outliers.

#### 4.1.1. Ubah Nama Kolom
Peneliti mengubah nama kolom dari fitur `Work_accident`, `average_montly_hours`, dan `Department` menjadi `work_accident`, `average_monthly_hours`, dan `department`. Tujuannya hanya sebagai keseragaman nama fitur, dan tidak ada typo pada nama fitur.

#### 4.1.2. Missing Values
Peneliti melakukan pengecekan missing values pada dataset. Namun tidak terdapat missing values pada dataset.

#### 4.1.3. Data Duplikat
Peneliti melakukan pengecekan data duplikat pada dataset. Penanganan data duplikat sangat penting, karena bisa menyebabkan bias, menghambar visualisasi dan analisis data, terutama pada model. Hasilnya, terdapat 3008 baris data yang duplikat. Peneliti melakukan penanganan pada data duplikat dengan cara menghapus data duplikat tersebut, karena data duplikat relatif sedikit.

#### 4.1.4. Ubah Tipe Data
Peneliti melakukan pengubahan tipe data pada dataset, agar dapat memudahkan analisis pada data. Adapun fitur yang diubah tipe datanya adalah sebagai berikut:
- work_accident menjadi category
- promotion_last_5years menjadi kategori
- left menjadi category
- department menjadi category
- salary menjadi category

#### 4.1.5. Deteksi Outlier 
Peneliti melakukan pengecekan outlier pada fitur numerik dengan metode IQR. Ini penting karena outlier dapat mengganggu model. Hasilnya seperti pada gambar berikut ini.

| ![Screenshot (144)](https://github.com/user-attachments/assets/365b06fe-6724-49a2-a3d2-895bd3f24a04) | 
|:--:| 
| *Cek Outlier Pada Fitur Numerik* |

Dari gambar tersebut, dapat dilihat ada outliers pada fitur `time_spend_company`. Data outliers nya berjumlah 824 data. Outlier yang terdapat pada dataset ini tidaklah ekstrem, dan bukan tipe anomali, melainkan pola alami dari dataset ini. Jadi peneliti memutuskan untuk melanjutkan analisis pada dataset ini.


Awalnya, dataset ini terdiri atas 14999 baris. Kemudian, setelah proses pembersihan data, total data pada dataset ini adalah 11991 baris.

### 4.2. Univariate Analysis

Univariate analysis adalah analisis data yang hanya melibatkan satu variabel/ fitur. Fokus utamanya adalah untuk memahami karakteristik dasar dari variabel tersebut.

#### 4.2.1. Categorical Features

##### 4.2.1.1. Fitur left

| ![Screenshot (144)](https://github.com/user-attachments/assets/4f7a04c0-854d-4807-ad8e-155a7ccc86a4) | 
|:--:| 
| *Distribusi Pegawai yang Resign* |

Dari grafik Distribusi pegawai Yang Resign dapat disimpulkan beberapa hal:
- Jumlah pegawai yang **tidak resign** jauh lebih banyak dibandingkan dengan yang **resign**.
- Jumlah pegawai yang **tidak resign** adalah **10000** dengan persentase **83.4**, sedangkan yang **resign** sekitar **1991** dengan persentase **16.6**.

##### 4.2.1.2. Fitur work_accident

| ![Screenshot (144)](https://github.com/user-attachments/assets/b9c6a2e5-e654-46bc-8dd8-68465f3ebdf6) | 
|:--:| 
| *Distribusi Pegawai yang Pernah Mengalami Kecelakaan Kerja* |

Dari grafik Distribusi Pegawai yang Pernah Mengalami Kecelakaan Kerja dapat disimpulkan bahwa mayoritas pegawai tidak pernah mengalami kecelakaan kerja selama bekerja di perusahaan tersebut. Yang pernah mengalami kecelakaan kerja hanya 15.4% saja.

##### 4.2.1.3. Fitur promotion_last_5years

| ![Screenshot (144)](https://github.com/user-attachments/assets/9a61dfc1-f2ce-468b-995f-5b1045e9f68f) | 
|:--:| 
| *Distribusi Pegawai yang Pernah Mendapatkan Promosi Selama Bekerja* |

Dapat dilihat bahwa pegawai yang pernah mendapatkan promosi di perusahaan tersebut hanya 203 orang dari belasan ribu pegawai. Ini sangat sedikit.

##### 4.2.1.4. Fitur department

| ![Screenshot (144)](https://github.com/user-attachments/assets/468ea793-9d9e-44b7-8aa7-cd5f9c151ac9) | 
|:--:| 
| *Distribusi Pegawai Berdasarkan Divisinya* |

Dari grafik di atas yang memperlihatkan Distribusi Pegawai Berdasarkan Divisinya dapat disimpulkan beberapa hal:
- **Distribusi karyawan tidak merata antar divisi**:
  
  - Tiga divisi teratas (**Sales, Technical, Support**) menyumbang lebih dari **60%** dari total tenaga kerja.
    
  - Hal ini bisa menunjukkan bahwa perusahaan memiliki fokus besar pada **penjualan**, **dukungan teknis**, dan **layanan pelanggan**.
    
- **Divisi manajemen** merupakan yang paling sedikit, hanya sekitar **3.6%**, yang bisa dimaklumi karena biasanya posisi manajerial memang lebih sedikit.
  
- **Divisi IT, R\&D, dan Product Management** tergolong menengah, dan ini bisa menunjukkan perusahaan juga cukup memperhatikan sisi pengembangan teknologi dan produk — meski tidak sebesar tenaga sales dan support.


##### 4.2.1.5. Fitur salary

| ![Screenshot (144)](https://github.com/user-attachments/assets/03bdd743-639c-42d4-9a5a-a4ccd7f1e314) | 
|:--:| 
| *Distribusi Pegawai Berdasarkan Kategori Gajinya* |

Dari grafik di atas yang memperlihatkan Distribusi Pegawai Berdasarkan Divisinya dapat disimpulkan beberapa hal:

- Mayoritas karyawan berada di kategori gaji **rendah** (`low`), yaitu **5.740 orang** atau **47.9%** dari total populasi.
- Disusul oleh kategori **gaji menengah** (`medium`) sebanyak **5.261 orang** (**43.9%**).
- Sedangkan karyawan dengan **gaji tinggi** (`high`) hanya sekitar **990 orang** atau **8.3%**.
- Hampir 92% dari total karyawan memiliki gaji di level `low` dan `medium`, sedangkan hanya 8% yang menikmati gaji tinggi. Ini menunjukkan adanya kesenjangan distribusi kompensasi dalam perusahaan.

#### 4.2.2. Numerical Features

Pada tahap ini peneliti melakukan analisis univariate terhadap fitur numerik dari dataset, yaitu `satisfaction_level`, `last_evaluation`, `number_project`, `average_monthly_hours`, dan `time_spend_company`.

| ![Screenshot (144)](https://github.com/user-attachments/assets/1d87a9ee-8dac-4663-842b-af7b7691e01b) | 
|:--:| 
| *Distribusi Fitur Numerik* |

Berdasarkan visualisasi grafik distribusi data numerik di atas, dapat disimpulkan beberapa hal.
1. **Satisfaction Level**
- **Distribusi** sedikit bimodal dan sedikit condong ke kanan.
- Ada dua puncak besar:
  - Sekitar **0.4–0.5** (rendah)
  - Sekitar **0.7–0.8** (tinggi)
- **Insight**:
  - Ada kelompok karyawan yang sangat puas dan cukup banyak yang kurang puas.
  - Karyawan dengan tingkat kepuasan rendah ini patut diawasi karena mereka berpotensi lebih besar untuk resign.

2. **Last Evaluation**
- Distribusi relatif **merata** dari 0.5 sampai 1, dengan sedikit puncak di area **0.55–0.6**.
- **Insight**:
  - Evaluasi kerja karyawan cenderung tersebar merata, artinya sistem penilaian cukup bervariasi.
  - Tidak ada indikasi bahwa karyawan yang performanya sangat rendah mendominasi.

3. **Number of Projects**
- Ada puncak jelas pada angka:
  - **3 proyek** (terbanyak)
  - **4 proyek** dan **5 proyek** menyusul
- Hanya sedikit karyawan yang menangani **lebih dari 5 proyek**.
- **Insight**:
  - Kebanyakan karyawan menangani 3–5 proyek.
  - Jumlah proyek ekstrem (sedikit atau banyak) sangat jarang dan bisa menjadi faktor stres/resign yang perlu dianalisis lebih lanjut.

4. **Average Monthly Hours**
- Distribusi **bimodal**:
  - Puncak di sekitar **150 jam** dan **250 jam** per bulan.
- Ada juga karyawan yang bekerja di atas **300 jam** per bulan, meskipun jumlahnya sedikit.
- **Insight**:
  - Ada dua pola kerja: yang cenderung normal dan yang bekerja sangat intens.
  - Karyawan yang bekerja lebih dari 250 jam kemungkinan mengalami **overwork**, berpotensi meningkatkan risiko resign atau burnout.

5. **Time Spent in Company**
- Mayoritas karyawan berada di:
  - **2 tahun** dan **3 tahun**
- Setelah itu jumlahnya turun drastis, kecuali ada lonjakan kecil di tahun ke-4 dan ke-5.
- **Insight**:
  - Karyawan cenderung keluar atau tidak bertahan lama setelah tahun ke-3.
  - Daya tahan karyawan terhadap lingkungan kerja menurun seiring waktu, bisa jadi indikator kurangnya prospek atau kenaikan karier.

### 4.3. Multivariate Analysis

Pada tahap ini peneliti akan melakukan multivariate analysis untuk menganalisis data yang melibatkan lebih dari satu variabel secara simultan. Tujuannya adalah untuk memahami hubungan antar variabel dan bagaimana variabel-variabel tersebut secara kolektif mempengaruhi suatu hasil atau fenomena, terutama yang mempengaruhi pegawai yang resign (`left`).

#### 4.3.1. Analisis Left terhadap Data Kategori

Peneliti melakukan analisis multivariate terhadap fitur kategorikal dari dataset terhadap label target, yaitu fitur `left`. Adapun kombinasinya yaitu `work_accident`-`left`, `promotion_last_5years`-`left`, `department`-`left`, dan `salary`-`left`.

| ![Screenshot (144)](https://github.com/user-attachments/assets/e819b3ca-7303-4c0d-8af7-1648b3945fc4) | 
|:--:| 
| *Analisis Pegawai yang Resign Terhadap Fitur Kategorikal* |

Berikut adalah **analisis dan insight** dari Multivariate Analysis pada keempat grafik bar (countplot). Masing-masing menunjukkan hubungan antara variabel kategori terhadap status resign (`left`):

1. **Work Accident vs Left**
- **Insight:** Pegawai yang **tidak mengalami kecelakaan kerja (work_accident = 0)** memiliki jumlah resign yang **jauh lebih tinggi** dibandingkan yang mengalami kecelakaan.
- **Interpretasi:** Ini cukup menarik karena biasanya kita berpikir kecelakaan bisa membuat karyawan keluar, tapi justru yang **tidak mengalami kecelakaan lebih banyak resign**. Ini bisa jadi karena mereka kurang merasa “terikat” atau karena faktor lain seperti lingkungan kerja atau beban kerja.

2. **Promotion in Last 5 Years vs Left**
- **Insight:** Karyawan yang **tidak mendapat promosi dalam 5 tahun terakhir (promotion_last_5years = 0)** memiliki angka resign yang sangat tinggi.
- **Interpretasi:** Ini konsisten dengan dugaan bahwa **kurangnya penghargaan atau perkembangan karir memicu resign**. Promosi tampaknya jadi faktor penting dalam retensi karyawan.

3. **Department vs Left**
- **Insight:** Departemen seperti **sales, technical, dan support** memiliki jumlah resign yang lebih tinggi secara absolut.
- **Interpretasi:**
  - **Sales:** Mungkin karena tekanan target yang tinggi.
  - **Technical & Support:** Bisa jadi karena beban kerja atau kurangnya jenjang karir.
  - Departemen seperti **management dan R&D** memiliki tingkat resign yang relatif rendah, menunjukkan stabilitas yang lebih tinggi.

4. **Salary vs Left**
- **Insight:** Mayoritas karyawan yang resign berasal dari kelompok gaji **rendah (low)**, diikuti oleh **medium**, dan hampir tidak ada yang dari kelompok **high salary**.
- **Interpretasi:** Gaji jelas berpengaruh terhadap loyalitas. **Semakin rendah gaji, semakin tinggi potensi resign.** Ini bisa dijadikan pertimbangan dalam strategi kompensasi perusahaan.

**Kesimpulan Umum:**
Variabel-variabel berikut punya **korelasi kuat dengan keputusan resign**:
- Tidak mendapat promosi
- Gaji rendah
- Bekerja di departemen tertentu (sales, support, technical)
- Tidak mengalami kecelakaan kerja (mungkin korelasi tidak langsung)

#### 4.3.2. Analisis Left terhadap Data Numerik

Peneliti melakukan analisis multivariate terhadap fitur kategorikal dari dataset terhadap label target, yaitu fitur `left`. Adapun kombinasinya yaitu `satisfaction_level`-`left`, `last_evaluation`-`left`, `average_monthly_hours`-`left`, `time_spend_company`-`left`, dan `number_project`-`left`.

| ![Screenshot (144)](https://github.com/user-attachments/assets/92d3987f-8b90-4743-9f1e-ed5573333d15) | 
|:--:| 
| *Analisis Pegawai yang Resign Terhadap Fitur Numerikal* |


Berikut adalah analisis dari lima **boxplot** yang membandingkan berbagai fitur numerik terhadap status **resign** (`left`: 0 = tidak resign, 1 = resign):

1. **Satisfaction Level**
- **Insight**: Pegawai yang resign (1) cenderung memiliki tingkat kepuasan kerja yang jauh lebih rendah dibandingkan yang tidak resign (0).
- **Interpretasi**: Kepuasan kerja rendah menjadi salah satu faktor utama seseorang memutuskan keluar dari perusahaan.

2. **Last Evaluation**
- **Insight**: Pegawai yang resign memiliki nilai evaluasi yang sedikit lebih tinggi secara median dibandingkan yang tidak resign.
- **Interpretasi**: Bisa jadi beberapa pegawai yang performanya tinggi merasa kurang dihargai atau tidak diberi tantangan/prospek karir, sehingga mereka memilih keluar.

3. **Average Monthly Hours**
- **Insight**: Pegawai yang resign bekerja lebih lama per bulan (jam kerja lebih tinggi).
- **Interpretasi**: Jam kerja yang terlalu tinggi bisa menyebabkan burnout dan meningkatkan kemungkinan resign.

4. **Time Spend Company**
- **Insight**: Pegawai yang resign umumnya telah bekerja lebih lama (median sekitar 4 tahun), sedangkan yang bertahan banyak di kisaran 3 tahun.
- **Interpretasi**: Masa kerja yang lama tanpa promosi atau perubahan kondisi kerja bisa menyebabkan kejenuhan, sehingga memicu resign.

5. **Number of Projects**
- **Insight**: Pegawai yang resign cenderung memiliki jumlah proyek yang lebih beragam (dari yang sedikit hingga sangat banyak).
- **Interpretasi**:
  - Resign bisa dipicu karena **kurangnya proyek (underutilization)** → merasa tidak berkembang.
  - Atau karena **terlalu banyak proyek (overload)** → burnout.

**Kesimpulan Umum**
Faktor-faktor yang paling berkorelasi dengan **resign** adalah:
- **Kepuasan kerja rendah**
- **Jam kerja yang tinggi**
- **Lama masa kerja**
- **Jumlah proyek terlalu sedikit atau terlalu banyak (kedua ekstrem)**

| ![Screenshot (144)](https://github.com/user-attachments/assets/1b4a156d-1126-43fe-af3c-2f8beb433546) | 
|:--:| 
| *Analisis Pegawai yang Resign Terhadap Rata-Rata Jam Kerja Bulanan - Tingkat Kepuasan Kerja* |

Berikut adalah insight dari grafik scatter plot yang membandingkan **pegawai yang tidak resign** (kiri) dan **pegawai yang resign** (kanan), berdasarkan **rata-rata jam kerja bulanan** dan **tingkat kepuasan kerja**:

| Kelompok Pegawai         | Insight                                                                                             |
|--------------------------|-----------------------------------------------------------------------------------------------------|
| Pegawai yang Tidak Resign | - Penyebaran tingkat kepuasan cukup merata di berbagai jam kerja, menunjukkan variasi yang luas.    |
|                           | - Kebanyakan memiliki kepuasan di atas 0.5 dan jam kerja bulanan antara 150-250 jam.                |
|                           | - Tidak ada pola yang menunjukkan bahwa jam kerja tinggi atau rendah secara langsung mendorong ketidakresign-an. |
| Pegawai yang Resign       | - Terlihat tiga kelompok dominan:                                                                  |
|                           |   1. Tingkat kepuasan rendah (~0.1 - 0.4) & jam kerja sekitar 130-160 jam.                          |
|                           |   2. Tingkat kepuasan tinggi (~0.7 - 1.0) & jam kerja tinggi (~225 - 275 jam).                      |
|                           |   3. Tingkat kepuasan sangat rendah (~0.1) dengan jam kerja sangat tinggi (~250 - 310 jam).         |
|                           | - Artinya, pegawai resign bisa terjadi pada dua ekstrem: overworked meskipun puas, atau sangat tidak puas meskipun tidak terlalu sibuk. |

**Insight:**

* Resign tidak selalu disebabkan oleh kepuasan rendah saja, tapi juga bisa karena **jam kerja yang terlalu tinggi**, bahkan untuk pegawai yang puas sekalipun.
* Ada indikasi bahwa manajemen beban kerja dan distribusi proyek sangat penting untuk mencegah turnover.

| ![Screenshot (144)](https://github.com/user-attachments/assets/48ff399e-ec27-4fe7-8f3f-2697228dc994) | 
|:--:| 
| *Analisis Pegawai yang Resign Terhadap Skor Evaluasi Terbaru pada Pegawai - Rata-Rata Jam Kerja Bulanan* |


Berikut adalah hasil analisis dan insight dari grafik scatter plot yang memperlihatkan hubungan antara **`average_montly_hours`** dan **`last_evaluation`**, dibandingkan berdasarkan status resign (**`left`**):

| Kelompok Pegawai         | Insight                                                                                             |
|--------------------------|-----------------------------------------------------------------------------------------------------|
| Pegawai yang Tidak Resign | - Data tersebar merata di seluruh rentang evaluasi dan jam kerja.                                   |
|                           | - Tidak ada pola yang jelas menunjukkan hubungan kuat antara evaluasi kerja dengan jam kerja bulanan. |
|                           | - Artinya, baik evaluasi tinggi maupun rendah tidak terlalu mempengaruhi keputusan untuk tetap bekerja. |
| Pegawai yang Resign       | - Terlihat dua kelompok mencolok:                                                                   |
|                           |   1. Evaluasi rendah (~0.4–0.6) & jam kerja rendah (~130–160 jam): kemungkinan karena performa kurang. |
|                           |   2. Evaluasi tinggi (~0.8–1.0) & jam kerja tinggi (~230–300 jam): kemungkinan karena overwork meskipun berkinerja tinggi. |
|                           | - Sedikit pegawai resign dengan skor evaluasi sedang (~0.6–0.7), menunjukkan bahwa kelompok ini lebih stabil. |

**Insight:**

* Pegawai dengan evaluasi rendah dan jam kerja rendah cenderung resign, kemungkinan karena dianggap tidak perform.
* Pegawai dengan evaluasi tinggi dan jam kerja tinggi juga cenderung resign, mungkin akibat kelelahan atau beban kerja berlebihan.
* Pegawai yang "aman" dari risiko resign berada di tengah-tengah: evaluasi sedang & jam kerja seimbang.

#### 4.3.3. Analisis Left terhadap Data Numerik - Kategori

| ![Screenshot (144)](https://github.com/user-attachments/assets/0a5ea5c7-8838-4f30-b587-50a09966ccc1) | 
|:--:| 
| *Analisis Pegawai yang Resign vs Menetap berdasarkan Lama Bekerja dan Tingkat Gaji* |

**Visualisasi Perbandingan: Pegawai yang Resign vs Menetap berdasarkan Lama Bekerja dan Tingkat Gaji**

* **Sumbu X:** Lama bekerja di perusahaan (`time_spend_company`)
* **Sumbu Y:** Jumlah pegawai
* **Warna Batang:**

  * **Biru tua:** Gaji rendah (`low`)
  * **Hijau tua:** Gaji sedang (`medium`)
  * **Hijau terang:** Gaji tinggi (`high`)
* Dibagi menjadi dua panel:

  * **Kiri:** Pegawai yang *resign*
  * **Kanan:** Pegawai yang *menetap*

**Insight dari Pegawai yang Resign:**

1. **Mayoritas yang resign berada di level gaji rendah**, khususnya:

   * Tahun ke-3 paling tinggi → Pegawai yang bekerja 3 tahun dengan gaji rendah paling banyak resign.
   * Disusul tahun ke-4 dan ke-5, juga didominasi gaji rendah.
   * Hampir tidak ada pegawai bergaji tinggi yang resign.

2. **Resign paling jarang terjadi di tahun ke-2 atau ke-6**, apalagi dengan gaji tinggi.

**Insight dari Pegawai yang Menetap:**

1. **Tingkat retensi paling tinggi terjadi pada pegawai yang telah bekerja selama 3 tahun**, terutama dengan:

   * Gaji rendah dan sedang.
   * Menunjukkan bahwa setelah 3 tahun, sebagian besar pegawai tetap bertahan.

2. **Pegawai dengan gaji tinggi memiliki kecenderungan lebih besar untuk bertahan**, meskipun jumlahnya tidak sebanyak pegawai dengan gaji rendah/menengah.

3. **Mereka yang sudah bekerja 6 tahun ke atas (hingga 10 tahun)** cenderung tetap tinggal di perusahaan, walaupun jumlahnya kecil.

**Kesimpulan:**

* Fokus pada pegawai **gaji rendah dan sedang** yang telah bekerja **3–5 tahun**, karena mereka paling rentan resign.
* Pertimbangkan peningkatan gaji atau insentif setelah masa kerja 2–3 tahun sebagai bentuk retensi.
* Pegawai **bergaji tinggi jarang resign**, jadi mempertahankan dan memotivasi mereka penting untuk jangka panjang.
* Buat program loyalitas atau kenaikan jenjang karier yang dimulai sejak tahun ke-2 atau ke-3.
* Evaluasi apakah gaji rendah di tahun-tahun krusial (3–5 tahun) memicu rasa tidak dihargai hingga pegawai memilih keluar.


| ![Screenshot (144)](https://github.com/user-attachments/assets/d3e19fa8-c2c8-4597-bf20-ca6fbbb14e8d) | 
|:--:| 
| *Analisis Pegawai yang Resign vs Rata-Rata Jam Kerja Bulanan - Mendapatkan Promosi Dalam 5 Tahun Terakhir* |

Berikut adalah hasil **analisis** dan **insight** dari grafik scatter plot yang menunjukkan hubungan antara **`rata-rata jam kerja bulanan`** dan status **`mendapatkan promosi dalam 5 tahun terakhir`**, dipisahkan berdasarkan status resign:

**Pegawai yang Tidak Resign**

* Terlihat cukup banyak pegawai **yang tidak mendapatkan promosi (nilai = 0)** namun **tetap bertahan** di perusahaan.
* Pegawai yang **mendapat promosi (nilai = 1)** juga cukup banyak dan tersebar merata dalam rentang jam kerja antara \~100 sampai \~280 jam per bulan.
* Artinya, **promosi mungkin berperan dalam menjaga loyalitas**, tapi **bukan satu-satunya faktor** yang membuat pegawai tetap tinggal.

**Pegawai yang Resign**

* Mayoritas **pegawai yang resign tidak mendapatkan promosi** (nilai = 0).
* Hanya **beberapa** pegawai yang resign **pernah mendapatkan promosi**, dan mereka tersebar di jam kerja rendah hingga tinggi (\~130–290 jam/bulan).
* Ini mengindikasikan bahwa:

  * **Kurangnya promosi** bisa menjadi salah satu faktor penyebab resign.
  * **Promosi tidak menjamin pegawai bertahan**, terutama jika dikombinasikan dengan **beban kerja yang tinggi** atau **faktor lain** seperti kepuasan kerja, kompensasi, atau keseimbangan kerja-hidup.

**Insight**

* Pegawai yang tidak dipromosikan lebih rentan untuk resign, khususnya mereka dengan beban kerja tinggi.
* Pegawai yang mendapatkan promosi cenderung bertahan, namun bukan berarti mereka tidak bisa resign. Ada kemungkinan bahwa **pegawai berprestasi tetap resign jika merasa tidak puas secara keseluruhan**.
* **Strategi retensi** tidak cukup hanya dengan promosi. Perusahaan juga perlu **mengatur beban kerja** dan **menciptakan lingkungan kerja yang suportif**.


#### 4.3.4. Analisis Matriks Korelasi

Dalam tahap ini, sebelum melakukan analisis terhadap matriks korelasi, peneliti terlebih dahulu melakukan encoding pada data kategorikal. Ini karena agar peneliti dapat melihat korelasi seluruh fitur, termasuk data kategorikal yang sudah dilakukan encoding, terhadap fitur target, yaitu `left`. Selain itu, encoding perlu dilakukan agar data kategorikal diubah menjadi data numerikal, sehingga model dapat memproses data, karena model hanya dapat memproses data numerik saja. Encoding yang dilakukan oleh peneliti pada dataset ini adalah ordinal encoding, dan one hot encoding. One Hot Encoding diterapkan pada fitur `department`, karena nilai kategori pada fitur tersebut tidak memiliki urutan. Sedangkan Ordinal Encoding diterapkan pada fitur `salary`, karena memiliki urutan yang logis (`low` < `medium` < `high`). 

Di bawah ini adalah code snippet yang dilakukan peneliti untuk melakukan encoding pada fitur `department` dan  `salary`.
```python
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
```

Setelah itu matriks korelasinya dapat dilihat pada gambar berikut.

| ![Screenshot (144)](https://github.com/user-attachments/assets/c57e53a2-f3ba-4680-9b9b-f8f9d088f4a1) | 
|:--:| 
| *Analisis Matriks Korelasi* |

Berikut beberapa insight penting terhadap `left`:

| Fitur                     | Korelasi terhadap `left` | Keterangan                                                                 |
|--------------------------|--------------------------|----------------------------------------------------------------------------|
| `satisfaction_level`     | **-0.35**                | Korelasi negatif kuat → makin puas, makin kecil kemungkinan resign         |
| `time_spend_company`     | 0.17                     | Korelasi positif ringan → makin lama di perusahaan, sedikit lebih rentan resign |
| `number_project`         | 0.03                     | Korelasi sangat kecil                                                      |
| `average_monthly_hours`  | 0.07                     | Korelasi sangat kecil                                                      |
| `last_evaluation`        | 0.01                     | Hampir tidak berkorelasi                                                   |
| `promotion_last_5years` | **-0.13**                | Korelasi negatif → yang pernah dipromosi, cenderung tidak resign           |
| `work_accident`          | **-0.13**                | Korelasi negatif → yang pernah kecelakaan kerja, cenderung bertahan        |
| `salary`                 | -0.12                    | Korelasi negatif → makin tinggi gaji, makin kecil kemungkinan resign        |

---

** Interpretasi Singkat:**

- **Fitur paling penting terhadap `left`** berdasarkan korelasi:  
   `satisfaction_level`  
   `promotion_last_5years`  
   `work_accident`  
   `salary`

## 5. Data Preparation
Pada bagian **Data Preparation**, dilakukan beberapa tahapan penting untuk mempersiapkan data sebelum digunakan dalam proses modeling. Berikut adalah tahapan-tahapan yang dilakukan:

### 5.1. Ubah Nama Kolom
- Nama kolom diubah menjadi huruf kecil seluruhnya untuk konsistensi.
- Typo pada nama kolom diperbaiki, contohnya:
  - `average_montly_hours` → `average_monthly_hours`
  - `Work_accident` → `work_accident`
  - `Department` → `department`

### 5.2. Missing Values
- Dilakukan pengecekan terhadap nilai kosong (missing values) pada dataset. Penanganan missing values pada data sangat penting, karena data yang tidak lengkap bisa memengaruhi kualitas dan akurasi model yang dibangun
- Hasilnya menunjukkan **tidak ada missing values**, sehingga tidak diperlukan langkah imputasi atau penghapusan data.

### 5.3. Data Duplikat
- Dataset diperiksa untuk mendeteksi data duplikat.
- Ditemukan **3008 baris duplikat** (20,05% dari total data), yang kemudian dihapus. Penanganan data duplikat sangat penting, karena bisa menyebabkan bias, menghambat visualisasi dan analisis data, terutama pada model.
- Setelah penghapusan data duplikat, dataset menyisakan **11.991 baris data**.

### 5.4. Ubah Tipe Data
- Beberapa kolom diubah menjadi tipe data yang sesuai untuk memudahkan analisis dan proses encoding:
- Pada tahap EDA, kolom `department`, `salary`, `work_accident`, `promotion_last_5years`, dan `left` diubah menjadi tipe kategorikal untuk memudahkan peneliti menganalisis data.
- Kemudian saat selesai tahap EDA, dan menuju tahap encoding, peneliti mengubah kolom `work_accident`, `promotion_last_5years`, dan `left` menjadi tipe integer.

### 5.5. Outliers
- Dilakukan pengecekan distribusi data numerik menggunakan boxplot. Ini penting karena outlier dapat mengganggu model.
- Terdapat outliers pada kolom `time_spend_company`, tetapi karena tidak ekstrem, outliers dibiarkan untuk menjaga pola data.
- Boxplot dapat dilihat di bawah ini

| ![Screenshot (144)](https://github.com/user-attachments/assets/365b06fe-6724-49a2-a3d2-895bd3f24a04) | 
|:--:| 
| *Cek Outlier Pada Fitur Numerik* |


### 5.6. Encoding
- Data kategorikal diubah menjadi format numerik agar dapat digunakan oleh algoritma machine learning:
- **Ordinal Encoding**: Kolom `salary` diubah menjadi nilai ordinal berdasarkan urutan `low`, `medium`, `high`.
- **One-Hot Encoding**: Kolom `department` diubah menjadi beberapa kolom biner (dummy variables), dengan satu kolom di-drop untuk menghindari multikolinearitas.
- Kolom kategorikal lainnya (`work_accident`, `promotion_last_5years`, `left`) diubah menjadi tipe integer.
- Encoding perlu dilakukan agar data kategorikal diubah menjadi data numerikal, sehingga model dapat memproses data, karena model hanya dapat memproses data numerik saja.

Di bawah ini adalah code snippet yang dilakukan peneliti untuk melakukan encoding pada fitur `department` dan  `salary`.
```python
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
```

### 5.6. Pemisahan Fitur
- Ini penting agar model tau untuk mempelajari fitur apa (X), dan memprediksi fitur apa (y).
- Dataset dipisahkan menjadi dua bagian:
  - **Fitur (X)**: Semua kolom kecuali kolom target `left`.
  - **Target (y)**: Kolom `left`, yang menunjukkan apakah karyawan resign (1) atau tidak (0).
Berikut ini adalah code snippet dalam melakukan pemisahan fitur pada dataset penelitian ini.
```python
X = df_encoded.drop(columns='left')
y = df_encoded['left']
```

### 5.7. Train-Test Split 
Tahap ini adalah proses membagi dataset menjadi Training Set untuk melatih model, dan Testing Set untuk menguji performa model untuk memastikan model dapat diandalkan pada data baru. 
Tahap ini penting karena beberapa hal, yaitu:
- **Evaluasi Generalisasi**: Mengukur kemampuan model pada data baru.
- **Deteksi Overfitting/Underfitting**: Membandingkan performa di training dan testing set.
- **Validasi Model**: Membandingkan performa model secara adil.

Peneliti melakukan pembagian data Train-Test dengan ratio 80% : 20%. Berikut ini adalah code snippet dalam melakukan train-test split pada penelitian ini.
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

Adapun shape dari dataset setelah dibagi menjadi Train-Test dapat dilihat di bawah ini.
```
shape of X_train: (9592, 17)
shape of y_train: (9592,)
shape of X_test: (2399, 17)
shape of y_test: (2399,)
percentage of train size: 0.80%
percentage of test size: 0.20%
```

## 6. Modelling
Pada tahap ini, peneliti menggunakan 3 algoritma klasifikasi. Kemudian peneliti memilih satu algoritma yang terbaik, dan melakukan proses hyperparameter tuning pada model tersebut.
### Tahapan dan Parameter yang Digunakan
Pada tahap pemodelan, dilakukan pengembangan tiga algoritma klasifikasi, yaitu:
1. **Gaussian Naive Bayes (GaussianNB)**:
    - Algoritma ini menggunakan pendekatan probabilistik berdasarkan Teorema Bayes.
    - Tidak memerlukan banyak parameter untuk diatur, sehingga cocok untuk baseline model.
    - **Parameter yang digunakan**: `default`

2. **Random Forest Classifier (RandomForestClassifier)**:
    - Algoritma ensemble berbasis pohon keputusan.
    - **Parameter utama yang digunakan**: `default`. Peneliti juga menggunakan `random_state` dengan nilai 42 untuk memastikan hasil yang konsisten.

3. **Extreme Gradient Boosting (XGBoost)**:
    - Algoritma boosting berbasis pohon keputusan.
    - **Parameter utama yang digunakan**: `default`. Peneliti juga menggunakan `random_state` dengan nilai 42 untuk memastikan hasil yang konsisten.
### Kelebihan dan Kekurangan Algoritma
1. **Gaussian Naive Bayes**:
    - **Kelebihan**:
      - Cepat dan efisien untuk dataset besar.
      - Tidak memerlukan banyak parameter untuk diatur.
    - **Kekurangan**:
      - Asumsi independensi antar fitur seringkali tidak realistis.
      - Kurang cocok untuk dataset dengan fitur yang saling berkorelasi.

2. **Random Forest Classifier**:
    - **Kelebihan**:
      - Mampu menangani dataset dengan fitur yang saling berkorelasi.
      - Tidak mudah overfitting karena menggunakan ensemble learning.
      - Memberikan interpretasi pentingnya fitur.
    - **Kekurangan**:
      - Memerlukan waktu komputasi lebih lama dibandingkan GaussianNB.
      - Hasil prediksi bisa sulit diinterpretasikan karena kompleksitas model.

3. **XGBoost**:
    - **Kelebihan**:
      - Sangat efisien untuk menangani dataset besar.
      - Memiliki kemampuan untuk menangani missing values.
      - Memberikan hasil yang sangat baik pada kompetisi data science.
    - **Kekurangan**:
      - Memerlukan waktu tuning parameter yang lebih lama.
      - Lebih kompleks dibandingkan algoritma lain.

### Pemilihan Model Terbaik
Berdasarkan evaluasi awal, model **RandomForestClassifier** dipilih sebagai model terbaik karena:
- Memiliki akurasi yang tinggi pada data training dan testing.
- Memiliki stabilitas yang lebih tinggi karena std dev lebih kecil.
- Memberikan keseimbangan yang baik antara performa dan interpretasi.
- Meskipun terdapat indikasi overfitting, selisih akurasi antara data training dan testing relatif kecil dibandingkan model lain.

### Proses Improvement (Hyperparameter Tuning)
Karena base model RandomForestClassifier masih memiliki indikasi overfittin, maka model tersebut akan dilanjutkan ke tahap hyperparameter tuning. Untuk meningkatkan performa model **RandomForestClassifier**, dilakukan proses hyperparameter tuning menggunakan **Bayesian Optimization** melalui `BayesSearchCV`. Hyperparameter tuning dengan Bayesian Optimization adalah metode untuk mencari kombinasi hyperparameter terbaik dalam suatu model machine learning dengan cara memodelkan proses pencarian sebagai masalah probabilistik. Parameter yang dioptimalkan meliputi:
- `n_estimators`: Jumlah pohon dalam hutan (10 hingga 200).
- `max_depth`: Kedalaman maksimum pohon (1 hingga 50).
- `min_samples_split`: Jumlah minimum sampel untuk membagi simpul (2 hingga 20).
- `min_samples_leaf`: Jumlah minimum sampel di daun simpul (1 hingga 20).

Proses tuning dilakukan dengan:
1. Menggunakan **StratifiedKFold** dengan 5 lipatan untuk menjaga distribusi kelas yang seimbang.
2. Menggunakan metrik evaluasi `accuracy` untuk memilih kombinasi parameter terbaik.
3. Melakukan iterasi sebanyak 30 kali untuk menemukan parameter optimal.

Setelah tuning, model menunjukkan peningkatan performa dengan akurasi yang lebih baik pada data testing dan pengurangan indikasi overfitting. Parameter terbaik yang ditemukan adalah:
- `n_estimators`: 100
- `max_depth`: 21
- `min_samples_split`: 3
- `min_samples_leaf`: 1

## 7. Evaluation

### Metrik Evaluasi yang Digunakan

Pada proyek ini, kinerja model dievaluasi menggunakan beberapa metrik utama, yaitu:

1. **Accuracy**: Mengukur proporsi prediksi yang benar terhadap total data. Formula untuk menghitung akurasi adalah:

$$
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
$$

Di mana:

* TP: True Positive (prediksi benar untuk kelas positif)
* TN: True Negative (prediksi benar untuk kelas negatif)
* FP: False Positive (prediksi salah untuk kelas positif)
* FN: False Negative (prediksi salah untuk kelas negatif)

2. **Precision**: Mengukur proporsi prediksi positif yang benar. Formula precision adalah:

$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$

3. **Recall**: Mengukur proporsi data positif yang berhasil diprediksi dengan benar. Formula recall adalah:

$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

4. **F1-Score**: Merupakan rata-rata harmonis dari precision dan recall. F1-Score digunakan untuk menangani ketidakseimbangan data. Formula F1-Score adalah:

$$
\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

5. **Confusion Matrix**: Matriks ini memberikan gambaran detail tentang jumlah prediksi benar dan salah untuk setiap kelas. Matriks ini terdiri dari empat elemen utama: TP, TN, FP, dan FN.

6. **Cross-Validation**: Untuk memastikan stabilitas model, dilakukan validasi silang (cross-validation) dengan metrik akurasi. Proses ini membagi data menjadi beberapa lipatan (folds) untuk melatih dan menguji model secara bergantian.



### Hasil Evaluasi Base Model Development

Berikut adalah hasil evaluasi ketiga model pada tahap base model development:

| Model                  | Train Accuracy | Train F1 Score | Test Accuracy | Test F1 Score | Test Precision | Test Recall |
| ---------------------- | -------------- | -------------- | ------------- | ------------- | -------------- | ----------- |
| GaussianNB             | 0.8225         | 0.8346         | 0.8420        | 0.8515        | 0.8512         | 0.8420      |
| RandomForestClassifier | 1.0000         | 1.0000         | 0.9858        | 0.9856        | 0.9860         | 0.9858      |
| XGBClassifier          | 0.9960         | 0.9960         | 0.9825        | 0.9824        | 0.9827         | 0.9825      |

**Insight**:

* Model **RandomForestClassifier** memiliki performa terbaik pada data test dengan akurasi dan F1-Score tertinggi (0.9858 dan 0.9856).
* Model **GaussianNB** memiliki performa paling rendah dibandingkan model lainnya.
* Model **XGBClassifier** mendekati performa RandomForestClassifier, namun sedikit lebih rendah.



### Hasil Evaluasi Model Setelah Tuning

Berdasarkan evaluasi, berikut adalah hasil kinerja model sebelum dan setelah tuning:

| Model                                   | Train Accuracy | Train F1 Score | Test Accuracy | Test F1 Score | Test Precision | Test Recall |
| --------------------------------------- | -------------- | -------------- | ------------- | ------------- | -------------- | ----------- |
| RandomForestClassifier (Sebelum Tuning) | 1.0000         | 1.0000         | 0.9858        | 0.9856        | 0.9860         | 0.9858      |
| RandomForestClassifier (Setelah Tuning) | 0.9971         | 0.9971         | 0.9862        | 0.9860        | 0.9864         | 0.9862      |

1. **Accuracy**:

   * Model RandomForestClassifier sebelum tuning memiliki akurasi sangat tinggi pada data train (1.000) dan test (0.9858). Namun, terdapat indikasi overfitting karena akurasi train jauh lebih tinggi dibanding test.
   * Setelah tuning, akurasi test meningkat menjadi 0.9862, sementara akurasi train sedikit menurun menjadi 0.9971. Ini menunjukkan bahwa tuning berhasil mengurangi overfitting.

2. **F1-Score**:

   * F1-Score sebelum tuning adalah 1.000 (train) dan 0.9856 (test). Setelah tuning, F1-Score test naik menjadi 0.9860, sedangkan train turun sedikit ke 0.9971. Ini menunjukkan keseimbangan precision dan recall yang lebih baik.

3. **Precision dan Recall**:

   * Sebelum tuning: precision 0.9860 dan recall 0.9858.
   * Setelah tuning: precision naik ke 0.9864 dan recall ke 0.9862. Artinya, model semakin baik dalam mengenali kelas positif dan negatif.

4. **Confusion Matrix**:

   | ![Screenshot (144)](https://github.com/user-attachments/assets/512f1962-7833-49c2-865f-275d96e383af) | 
   |:--:| 
   | *Perbandingan Matriks Korelasi Sebelum dan Sesudah Tuning* |


   Tuning berhasil mengurangi kesalahan False Positive dari 4 menjadi 3.

4. **Cross-Validation**:
   Validasi silang menunjukkan model cukup stabil, dengan rata-rata akurasi mendekati hasil data test. Ini membuktikan bahwa model dapat bekerja dengan baik pada data yang belum pernah dilihat.

   | ![Screenshot (144)](https://github.com/user-attachments/assets/bf8054e5-293f-47f3-957d-74923bdb93e5) | 
   |:--:| 
   | *Grafik Cross Validation* |

   

Model **RandomForestClassifier** menunjukkan performa sangat baik untuk memprediksi apakah karyawan akan resign. Setelah tuning, model menjadi lebih seimbang dan mampu melakukan generalisasi lebih baik ke data test. Metrik seperti accuracy, F1-Score, precision, recall, dan confusion matrix menunjukkan bahwa model ini layak digunakan dalam pengambilan keputusan.


## 8. Model Interpretation 
Pada tahap ini, peneliti akan melakukan interpretasi hasil model dengan menggunakan metode features importance. 

| ![Screenshot (144)](https://github.com/user-attachments/assets/e6931806-3e1d-4df9-8411-9de007589661) | 
   |:--:| 
   | *Grafik Features Importance terhadap Klasifikasi `left` dari model Random Forest Classifier* |

Grafik ini menunjukkan seberapa besar kontribusi masing-masing fitur terhadap keputusan model Random Forest dalam memprediksi apakah seorang pegawai akan resign atau tidak.

| **No.** | **Fitur**                                      | **Interpretasi**                                                                                                                                                      |
| ------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1       | **satisfaction\_level**                        |  Fitur paling penting. Tingkat kepuasan kerja sangat berpengaruh terhadap kemungkinan resign. Pegawai dengan tingkat kepuasan rendah lebih cenderung resign.        |
| 2       | **number\_project**                            | Jumlah proyek yang dikerjakan berkorelasi kuat. Bisa jadi indikasi beban kerja atau keterlibatan pegawai.                                                             |
| 3       | **time\_spend\_company**                       | Lama bekerja di perusahaan berpengaruh—pegawai yang terlalu lama tanpa perkembangan cenderung resign.                                                                 |
| 4       | **average\_monthly\_hours**                    | Rata-rata jam kerja bulanan signifikan. Jam kerja yang terlalu tinggi bisa menyebabkan burnout dan memicu resign.                                                     |
| 5       | **last\_evaluation**                           | Skor evaluasi terakhir memengaruhi prediksi. Evaluasi rendah bisa menunjukkan performa buruk, atau evaluasi tinggi tapi tanpa penghargaan juga bisa memicu frustrasi. |
| 6       | **salary**                                     | Meskipun tidak sepenting lima besar, gaji tetap menjadi pertimbangan dalam keputusan resign.                                                                          |
| 7       | **work\_accident**                             | Ada pengaruh kecil; bisa jadi pegawai yang mengalami kecelakaan merasa tidak aman atau tidak puas.                                                                    |
| 8 sampai 16    | **department\_* dan promotion\_last\_5years*\* | Semua fitur ini memiliki pengaruh yang sangat kecil terhadap prediksi. Menunjukkan bahwa faktor departemen dan promosi bukan penentu utama dalam model ini.           |

## 9. Kesimpulan dan Saran:

### Kesimpulan
1. **Faktor Utama Resign**:
    - **Kepuasan Kerja Rendah**: Pegawai dengan tingkat kepuasan rendah memiliki kemungkinan resign yang tinggi.
    - **Jam Kerja Tinggi**: Pegawai dengan jam kerja bulanan tinggi (>250 jam) cenderung resign karena burnout.
    - **Kurangnya Promosi**: Pegawai yang tidak dipromosikan dalam 5 tahun terakhir lebih rentan resign.
    - **Gaji Rendah**: Pegawai dengan gaji rendah memiliki tingkat resign yang jauh lebih tinggi dibandingkan dengan gaji menengah atau tinggi.
    - **Masa Kerja 3–5 Tahun**: Pegawai dengan masa kerja 3–5 tahun memiliki risiko resign tertinggi.
    - - Fitur-fitur seperti departemen atau apakah pernah dipromosikan ternyata kurang relevan dalam konteks prediksi ini, **menurut model Random Forest**.

2. **Departemen dengan Tingkat Resign Tinggi**:
    - **Sales**, **Technical**, dan **Support** memiliki jumlah resign tertinggi, kemungkinan karena tekanan kerja atau kurangnya jenjang karir.

3. **Model Prediksi**:
    - Model **RandomForestClassifier** dengan hyperparameter tuning memberikan akurasi tinggi (98.62%) dan dapat digunakan untuk memprediksi pegawai yang berisiko resign.

### Saran
- Fokus retensi pegawai bisa dimulai dari peningkatan kepuasan kerja, pengaturan beban kerja yang seimbang, dan pengakuan performa melalui promosi atau penghargaan.
- Berikan program kesejahteraan, pelatihan, dan pengakuan untuk meningkatkan kepuasan kerja.
- Fokus pada pegawai dengan kepuasan rendah untuk mencegah resign.
- Kurangi jam kerja berlebih (>250 jam) untuk menghindari burnout.
- Distribusikan proyek secara merata untuk menghindari underutilization atau overload.
- Tingkatkan peluang promosi, terutama untuk pegawai dengan masa kerja 3–5 tahun.
- Berikan insentif atau bonus untuk pegawai yang berprestasi.
- Evaluasi ulang struktur gaji, terutama untuk pegawai dengan gaji rendah.
- Berikan kenaikan gaji atau insentif tambahan untuk meningkatkan loyalitas.
- Lakukan evaluasi mendalam pada departemen **Sales**, **Technical**, dan **Support** untuk mengidentifikasi penyebab resign.
- Berikan pelatihan atau program pengembangan karir untuk meningkatkan retensi di departemen tersebut.

Dengan langkah-langkah ini, perusahaan dapat meningkatkan retensi karyawan, mengurangi biaya turnover, dan menciptakan lingkungan kerja yang lebih produktif.

## Referensi

[1] M. Patterson, P. Warr, and M. West, “Organizational climate and company productivity: The role of employee affect and employee level,” Journal of Occupational and Organizational Psychology, vol. 77, no. 2, pp. 193–216, Jun. 2004, doi: https://doi.org/10.1348/096317904774202144.
‌

[2] M. Faisal Qureshi, "HR Analytics and Job Prediction," Kaggle, 2023. [Online]. Available: https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction/data. [Accessed: 2025-04-15].


[3] A. C. Glebbeek and E. H. Bax, “IS HIGH EMPLOYEE TURNOVER REALLY HARMFUL? AN EMPIRICAL TEST USING COMPANY RECORDS.,” Academy of Management Journal, vol. 47, no. 2, pp. 277–286, Apr. 2004, doi: https://doi.org/10.2307/20159578.
‌
