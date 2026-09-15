# MODUL PERKULIAHAN: SISTEM KENDALI CERDAS
## Topik: Fuzzy PID Controller (Pengendali PID Adaptif Berbasis Logika Fuzzy)

---

## 1. Capaian Pembelajaran (Learning Objectives)

Setelah mempelajari materi ini, mahasiswa diharapkan mampu:

1. Memahami keterbatasan kendali PID konvensional dan kendali Fuzzy murni
2. Menjelaskan konsep dan arsitektur *Fuzzy Adaptive PID* (Fuzzy Gain Scheduling)
3. Merancang basis aturan (*rule base*) untuk penalaan parameter PID secara *online*
4. Mensimulasikan dan membandingkan kinerja PID konvensional dengan Fuzzy PID pada sistem non-linear

---

## 2. Pendahuluan & Motivasi

### Mengapa kita membutuhkan Fuzzy PID?

**Kendali PID Konvensional:**
- Sangat baik untuk sistem linear
- Memiliki akurasi tinggi pada keadaan tunak (*steady-state*) berkat aksi Integral
- **Kelemahan fatal:** Tidak optimal pada sistem non-linear, memiliki *time-delay* besar, atau parameter sistem yang berubah-ubah (*time-varying*)
- PID dengan gain tetap ($K_p, K_i, K_d$) tidak bisa beradaptasi

**Kendali Fuzzy Murni (FLC):**
- Sangat tangguh (*robust*) untuk sistem non-linear
- Tidak memerlukan model matematis yang presisi
- **Kelemahan:** Sering memiliki kesalahan keadaan tunak (*steady-state error* / offset) dan sulit mencapai presisi tinggi pada sistem yang membutuhkan akurasi absolut

**Solusi: Fuzzy PID**
- Menggabungkan keunggulan keduanya
- Menggunakan struktur PID untuk menjamin akurasi *steady-state* (melalui aksi Integral)
- Menggunakan Logika Fuzzy untuk **menala (*tuning*) nilai $K_p, K_i,$ dan $K_d$ secara otomatis dan *real-time*** berdasarkan kondisi error saat itu

---

## 3. Konsep Dasar Fuzzy Adaptive PID

Pada arsitektur *Fuzzy Gain Scheduling*, Logika Fuzzy tidak menggantikan blok PID, melainkan bertindak sebagai **supervisor** yang mengubah parameter PID di setiap langkah waktu (*sampling time*).

### Persamaan Dasar PID Diskrit

$$u(k) = K_p e(k) + K_i \sum e(i) + K_d \Delta e(k)$$

### Pada Fuzzy PID, Parameter Tersebut Berubah Menjadi:

- $K_p = K_{p0} + \Delta K_p$
- $K_i = K_{i0} + \Delta K_i$
- $K_d = K_{d0} + \Delta K_d$

**Dimana:**
- $K_{p0}, K_{i0}, K_{d0}$ = Nilai awal PID (bisa dari tuning Ziegler-Nichols atau Cohen-Coon)
- $\Delta K_p, \Delta K_i, \Delta K_d$ = Nilai koreksi yang dihasilkan oleh *Fuzzy Inference System*

---

## 4. Tahapan Perancangan Fuzzy PID

### A. Penentuan Variabel Input dan Output Fuzzy

**Input Fuzzy (2 Variabel):**
1. **Error ($e$):** Selisih antara *setpoint* dan nilai aktual
2. **Change of Error ($ec$ atau $\Delta e$):** Turunan/perubahan error terhadap waktu

**Output Fuzzy (3 Variabel):**
1. $\Delta K_p$
2. $\Delta K_i$
3. $\Delta K_d$

### B. Fuzzifikasi (Himpunan Fuzzy)

Variabel input ($e$ dan $ec$) serta output ($\Delta K$) dibagi menjadi himpunan fuzzy. Standar yang sering digunakan dalam literatur adalah **7 level linguistik**:

- **NB** (*Negative Big*)
- **NM** (*Negative Medium*)
- **NS** (*Negative Small*)
- **ZO** (*Zero*)
- **PS** (*Positive Small*)
- **PM** (*Positive Medium*)
- **PB** (*Positive Big*)

*Fungsi keanggotaan (Membership Function)* yang umum digunakan adalah bentuk **Segitiga (Triangular)** atau **Gaussian**.

### C. Perancangan Basis Aturan (Rule Base)

Ini adalah "otak" dari Fuzzy PID. Aturan dibuat berdasarkan prinsip penalaan PID klasik. Berikut adalah prinsip logika untuk membuat tabel aturan:

#### 1. Aturan untuk $\Delta K_p$ (Fokus pada Kecepatan Respons & Overshoot)
- Jika $|e|$ **Besar** → $\Delta K_p$ harus **Besar** (agar respons cepat)
- Jika $|e|$ **Kecil** → $\Delta K_p$ harus **Kecil** (untuk mencegah *overshoot*)

#### 2. Aturan untuk $\Delta K_i$ (Fokus pada Steady-State Error & Windup)
- Jika $|e|$ **Besar** → $\Delta K_i$ harus **Kecil/Nol** (mencegah *Integral Windup* dan overshoot berlebih)
- Jika $|e|$ **Kecil** → $\Delta K_i$ harus **Besar** (menghilangkan *steady-state error* / offset)

#### 3. Aturan untuk $\Delta K_d$ (Fokus pada Antisipasi & Redaman)
- Jika $|e|$ **Besar** → $\Delta K_d$ harus **Kecil** (agar aksi turunan tidak terlalu sensitif terhadap noise)
- Jika $|ec|$ **Besar** (error berubah cepat) → $\Delta K_d$ harus **Besar** (untuk memberikan efek pengereman/redaman)

*Catatan: Tabel aturan lengkapnya berukuran 7×7 = 49 aturan untuk masing-masing $\Delta K$.*

### D. Inferensi dan Defuzzifikasi

- **Inferensi:** Menggunakan metode *Mamdani* (Min-Max) atau *Sugeno*
- **Defuzzifikasi:** Menggunakan metode *Centroid* (Center of Gravity) untuk mengubah nilai fuzzy $\Delta K_p, \Delta K_i, \Delta K_d$ menjadi nilai *crisp* (angka riil)

### E. Update Parameter PID

Nilai *crisp* hasil defuzzifikasi ditambahkan ke nilai PID awal, lalu dikirim ke blok PID untuk menghitung sinyal kendali $u(t)$ ke plant.

---

## 5. Studi Kasus Penerapan

### Kasus: Pengendalian Suhu pada Furnace (Oven) Industri

**Karakteristik Plant:**
- Non-linear
- Memiliki *time-delay* (dead time) yang besar
- Kapasitas panas berubah tergantung banyaknya material di dalam oven

**Masalah dengan PID Biasa:**
- Jika disetel untuk beban penuh (agar tidak *overshoot*), maka saat beban kosong, respons akan sangat lambat
- Sebaliknya, jika disetel untuk beban kosong, saat beban penuh suhu akan *overshoot* parah

**Solusi Fuzzy PID:**
- Saat suhu masih jauh dari *setpoint* ($e$ besar), Fuzzy akan memperbesar $K_p$ dan mematikan $K_i$ agar suhu naik cepat tanpa *windup*
- Saat suhu mendekati *setpoint* ($e$ kecil), Fuzzy akan memperkecil $K_p$, memperbesar $K_i$ untuk mengunci suhu tepat di *setpoint*, dan memperbesar $K_d$ untuk menahan laju kenaikan suhu agar tidak *overshoot*

---

## 6. Kelebihan dan Kekurangan Fuzzy PID

### Kelebihan

1. **Adaptif & Robust:** Mampu menangani perubahan parameter plant dan gangguan (*disturbance*) secara *real-time*
2. **Akurasi Tinggi:** Mempertahankan keunggulan aksi Integral dari PID sehingga *steady-state error* mendekati nol
3. **Respons Transien Terbaik:** *Overshoot* dan *settling time* jauh lebih baik dibandingkan PID konvensional pada sistem non-linear

### Kekurangan

1. **Beban Komputasi Tinggi:** Membutuhkan mikrokontroler/DSP yang lebih cepat karena harus menghitung Fuzzifikasi, Inferensi (49 aturan × 3 output), dan Defuzzifikasi di setiap *sampling time*
2. **Tuning yang Kompleks:** Menentukan rentang *scaling factor* (GF, GEC, GKp, GKi, GKd) dan membuat 49 aturan yang optimal membutuhkan waktu dan *trial-error* yang tidak sedikit
3. **Ketergantungan pada Sensor:** Sangat sensitif terhadap *noise* pada sensor, terutama pada turunan error ($ec$) yang memicu aksi $K_d$

---


## 8. Referensi Bacaan Lanjutan

1. Passino, K. M., & Yurkovich, S. (1998). *Fuzzy Control*. Addison-Wesley.
2. Mudi, R. K., & Pal, N. R. (1999). "A self-tuning fuzzy PID controller". *IEEE Transactions on Systems, Man, and Cybernetics*. (Paper klasik yang sangat disarankan untuk dibaca)
3. Ogata, K. (2010). *Modern Control Engineering*. (Sebagai referensi dasar teori PID)

---
