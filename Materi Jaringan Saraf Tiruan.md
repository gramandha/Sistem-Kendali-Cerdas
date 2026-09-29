# Materi Jaringan Saraf Tiruan

Sep 29, 2026 · @gramandha

## 1. Pendahuluan

Jaringan Saraf Tiruan (JST), atau Artificial Neural Network (ANN), adalah model komputasi yang belajar memetakan masukan ke keluaran dari contoh data. JST tersusun dari banyak unit sederhana bernama **neuron** yang saling terhubung dan bekerja secara paralel.

### 1.1 Inspirasi dari neuron biologis

| Neuron biologis | Neuron tiruan |
| --- | --- |
| Dendrit: menerima sinyal dari neuron lain | Masukan x1, x2, …, xn |
| Sinapsis: kekuatan sambungan antarneuron | Bobot w1, w2, …, wn |
| Soma (badan sel): mengakumulasi sinyal | Penjumlahan berbobot Σ wi·xi + b |
| Ambang aktivasi: neuron “menembak” jika sinyal cukup kuat | Fungsi aktivasi f(·) |
| Akson: meneruskan sinyal keluar | Keluaran y |

Kemampuan belajar JST berasal dari penyesuaian **bobot**. Ini mirip dengan sinapsis di otak yang menguat atau melemah karena pengalaman.

### 1.2 Sejarah singkat

| Tahun | Tonggak |
| --- | --- |
| 1943 | McCulloch dan Pitts memperkenalkan model neuron matematis pertama |
| 1958 | Rosenblatt memperkenalkan perceptron yang dapat belajar dari data |
| 1969 | Minsky dan Papert menunjukkan perceptron satu lapis tidak bisa menyelesaikan XOR; riset JST melambat |
| 1986 | Rumelhart, Hinton, dan Williams mempopulerkan algoritma backpropagation untuk jaringan berlapis |
| 1998 | LeCun dkk. memperkenalkan CNN LeNet-5 untuk pengenalan angka tulisan tangan |
| 2012 | AlexNet memenangkan kompetisi ImageNet; era deep learning dimulai |
| 2017 | Arsitektur Transformer diperkenalkan dan menjadi dasar model bahasa besar |

### 1.3 Kapan JST digunakan?

JST cocok untuk masalah yang polanya sulit dirumuskan dengan aturan eksplisit, tetapi contoh datanya banyak. Contohnya pengenalan wajah, pengenalan suara, klasifikasi sinyal EKG, prediksi deret waktu, dan penerjemahan bahasa.

## 2. Model Neuron Tiruan

Satu neuron menghitung jumlahan berbobot dari masukannya, menambahkan bias, lalu melewatkan hasilnya ke fungsi aktivasi.

```math
z = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^{T}\mathbf{x} + b, \qquad y = f(z)
```

| Komponen | Simbol | Peran |
| --- | --- | --- |
| Masukan | x1 … xn | Fitur data, misalnya nilai piksel atau sampel sinyal |
| Bobot | w1 … wn | Menentukan seberapa penting setiap masukan; inilah yang dipelajari |
| Bias | b | Menggeser ambang aktivasi, agar neuron bisa aktif meski semua masukan nol |
| Net input | z | Hasil penjumlahan berbobot ditambah bias |
| Fungsi aktivasi | f(·) | Menambahkan sifat nonlinier dan menentukan keluaran |
| Keluaran | y | Hasil neuron, diteruskan ke neuron berikutnya |

**Catatan**: bentuk Σ wi·xi adalah perkalian titik (dot product). Operasi ini sama dengan satu titik dari konvolusi diskrit tanpa proses flip, sehingga materi konvolusi sebelumnya muncul lagi di JST, terutama pada CNN.

### 2.1 Fungsi aktivasi

Tanpa fungsi aktivasi nonlinier, tumpukan banyak lapisan tetap setara dengan satu fungsi linier. Fungsi aktivasilah yang membuat JST mampu memodelkan pola rumit.

| Fungsi | Rumus f(z) | Rentang keluaran | Turunan f′(z) | Penggunaan umum |
| --- | --- | --- | --- | --- |
| Step (undak biner) | 1 jika z ≥ 0; 0 jika z < 0 | {0, 1} | 0 (tidak bisa dipakai backprop) | Perceptron klasik |
| Linear | z | (−∞, ∞) | 1 | Keluaran regresi |
| Sigmoid | 1 / (1 + e^(−z)) | (0, 1) | f(z)·(1 − f(z)) | Keluaran klasifikasi biner |
| Tanh | (e^z − e^(−z)) / (e^z + e^(−z)) | (−1, 1) | 1 − f(z)² | Lapisan tersembunyi, RNN |
| ReLU | max(0, z) | \[0, ∞) | 1 jika z > 0; 0 jika z < 0 | Lapisan tersembunyi (paling populer) |
| Leaky ReLU | z jika z > 0; 0,01z jika z ≤ 0 | (−∞, ∞) | 1 atau 0,01 | Mengatasi neuron ReLU yang “mati” |
| Softmax | e^(zi) / Σ e^(zj) | (0, 1), total = 1 | — | Keluaran klasifikasi banyak kelas |

**Masalah vanishing gradient**: turunan sigmoid maksimal hanya 0,25. Pada jaringan dalam, gradien dikalikan berulang kali dengan angka kecil sehingga mendekati nol, dan lapisan awal hampir tidak belajar. ReLU mengurangi masalah ini karena turunannya 1 untuk z > 0.

## 3. Arsitektur Jaringan Saraf Tiruan

Neuron disusun dalam lapisan. Pada jaringan feedforward, sinyal hanya mengalir satu arah dari input ke keluaran, dan setiap neuron terhubung ke semua neuron di lapisan berikutnya (fully connected).

![Arsitektur MLP 3–4–2](gambar/arsitektur-mlp.png)

| Lapisan | Fungsi | Jumlah neuron |
| --- | --- | --- |
| Input | Menerima data mentah; tidak melakukan perhitungan | Sama dengan jumlah fitur, misalnya 3 untuk 3 fitur |
| Tersembunyi (hidden) | Mengekstraksi pola dan fitur dari data | Dipilih perancang; boleh lebih dari satu lapisan |
| Keluaran | Menghasilkan prediksi akhir | 1 untuk regresi atau biner; K untuk K kelas |

- Jaringan dengan **dua atau lebih lapisan tersembunyi** disebut deep neural network. Pembelajarannya disebut **deep learning**.
- **Jumlah parameter** satu lapisan fully connected = (neuron masuk × neuron keluar) + neuron keluar. Pada contoh 3–4–2: (3×4 + 4) + (4×2 + 2) = 16 + 10 = 26 parameter.
- **Teorema aproksimasi universal**: MLP dengan satu lapisan tersembunyi dan neuron yang cukup banyak dapat mendekati fungsi kontinu apa pun. Dalam praktik, jaringan yang lebih dalam biasanya lebih efisien.

## 4. Proses Pembelajaran

JST belajar dengan mengulang tiga langkah: menghitung prediksi (forward), mengukur kesalahan (loss), lalu memperbaiki bobot ke arah yang mengurangi kesalahan (backward).

### 4.1 Forward propagation

Data masuk ke lapisan input, lalu dihitung lapis demi lapis sampai keluaran. Untuk lapisan ke-l:

```math
\mathbf{z}^{(l)} = \mathbf{W}^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}, \qquad \mathbf{a}^{(l)} = f\left(\mathbf{z}^{(l)}\right)
```

a⁽⁰⁾ adalah vektor masukan x, dan aktivasi lapisan terakhir adalah prediksi ŷ.

### 4.2 Fungsi loss

Fungsi loss mengukur seberapa jauh prediksi ŷ dari target y. Makin kecil loss, makin baik model.

| Fungsi loss | Rumus (untuk N data) | Dipakai untuk |
| --- | --- | --- |
| Mean Squared Error (MSE) | (1/N) Σ (y − ŷ)² | Regresi (keluaran angka kontinu) |
| Binary Cross-Entropy | −(1/N) Σ \[y ln ŷ + (1 − y) ln(1 − ŷ)\] | Klasifikasi dua kelas, keluaran sigmoid |
| Categorical Cross-Entropy | −(1/N) Σ Σ yk ln ŷk | Klasifikasi banyak kelas, keluaran softmax |

### 4.3 Gradient descent

Bobot diperbarui berlawanan arah dengan gradien loss, seperti menuruni lembah menuju titik terendah:

```math
w \leftarrow w - \eta \frac{\partial L}{\partial w}, \qquad b \leftarrow b - \eta \frac{\partial L}{\partial b}
```

η adalah **learning rate**. Jika terlalu besar, loss bisa melompat-lompat atau membesar. Jika terlalu kecil, pelatihan sangat lambat. Nilai awal yang umum dicoba antara 0,001 dan 0,1.

| Varian | Data per pembaruan bobot | Ciri |
| --- | --- | --- |
| Batch gradient descent | Seluruh data latih | Stabil, tetapi lambat untuk data besar |
| Stochastic gradient descent (SGD) | 1 data | Cepat, tetapi arah pembaruan berisik |
| Mini-batch gradient descent | Sekelompok kecil, misalnya 32 atau 64 data | Kompromi; paling banyak dipakai |

Optimizer modern seperti **Momentum**, **RMSProp**, dan **Adam** memodifikasi aturan di atas agar konvergen lebih cepat dan stabil.

### 4.4 Backpropagation

Backpropagation menghitung ∂L/∂w untuk semua bobot secara efisien dengan **aturan rantai** (chain rule), dimulai dari lapisan keluaran dan bergerak mundur.

Definisikan error lokal (delta) setiap neuron, δ = ∂L/∂z:

```math
\delta^{(L)} = \frac{\partial L}{\partial \hat{y}} \cdot f'\left(z^{(L)}\right), \qquad \delta_j^{(l)} = \left( \sum_{k} w_{kj}^{(l+1)} \delta_k^{(l+1)} \right) f'\left(z_j^{(l)}\right)
```

Gradien setiap bobot adalah delta neuron tujuan dikalikan aktivasi neuron asal:

```math
\frac{\partial L}{\partial w_{ji}^{(l)}} = \delta_j^{(l)}\, a_i^{(l-1)}, \qquad \frac{\partial L}{\partial b_j^{(l)}} = \delta_j^{(l)}
```

### 4.5 Alur pelatihan lengkap

1. Inisialisasi bobot dengan nilai acak kecil dan bias dengan nol.
2. Ambil satu mini-batch data latih.
3. Forward propagation: hitung prediksi ŷ.
4. Hitung loss L antara ŷ dan y.
5. Backpropagation: hitung gradien semua bobot dan bias.
6. Perbarui bobot dan bias dengan gradient descent.
7. Ulangi langkah 2–6 sampai seluruh data terpakai. Satu putaran penuh disebut **satu epoch**.
8. Ulangi untuk banyak epoch sampai loss data validasi berhenti membaik.

## 5. Contoh Perhitungan Manual

### 5.1 Contoh 1 — perceptron untuk gerbang AND

Latih perceptron dengan fungsi step (y = 1 jika z ≥ 0, selainnya 0) agar meniru gerbang AND. Nilai awal w1 = 0, w2 = 0, b = 0, dan learning rate η = 1.

Aturan belajar perceptron, dengan t adalah target:

```math
w_i \leftarrow w_i + \eta\,(t - y)\,x_i, \qquad b \leftarrow b + \eta\,(t - y)
```

**Epoch 1**, langkah demi langkah:

| x1 | x2 | Target t | z = w1x1 + w2x2 + b | y | Error t − y | Bobot baru (w1, w2, b) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0 + 0 + 0 = 0 | 1 | −1 | (0, 0, −1) |
| 0 | 1 | 0 | 0 + 0 − 1 = −1 | 0 | 0 | (0, 0, −1) |
| 1 | 0 | 0 | 0 + 0 − 1 = −1 | 0 | 0 | (0, 0, −1) |
| 1 | 1 | 1 | 0 + 0 − 1 = −1 | 0 | +1 | (1, 1, 0) |

Proses yang sama diulang. Bobot di akhir setiap epoch:

| Epoch | Jumlah error | (w1, w2, b) di akhir epoch |
| --- | --- | --- |
| 1 | 2 | (1, 1, 0) |
| 2 | 3 | (2, 1, −1) |
| 3 | 3 | (2, 1, −2) |
| 4 | 2 | (2, 2, −2) |
| 5 | 1 | (2, 1, −3) |
| 6 | 0 | (2, 1, −3) — berhenti |

**Verifikasi** dengan w1 = 2, w2 = 1, b = −3: (0,0) → z = −3 → 0; (0,1) → z = −2 → 0; (1,0) → z = −1 → 0; (1,1) → z = 0 → 1. Semua sesuai tabel kebenaran AND.

Garis keputusan 2x1 + x2 − 3 = 0 memisahkan titik (1,1) dari tiga titik lainnya. Perceptron satu lapis hanya bisa menyelesaikan masalah yang **terpisah secara linier** seperti AND dan OR, tetapi tidak XOR.

### 5.2 Contoh 2 — satu langkah backpropagation

Jaringan sederhana: satu masukan x, satu neuron tersembunyi h, dan satu neuron keluaran ŷ. Keduanya memakai sigmoid σ, dan loss L = ½ (y − ŷ)².

Data: x = 1, target y = 1. Bobot awal w1 = 0,5, b1 = 0, w2 = 0,5, b2 = 0. Learning rate η = 0,5.

**Langkah 1 — forward propagation**

- z1 = w1·x + b1 = 0,5 → h = σ(0,5) = 0,6225
- z2 = w2·h + b2 = 0,3112 → ŷ = σ(0,3112) = 0,5772
- L = ½ (1 − 0,5772)² = 0,0894

**Langkah 2 — backpropagation** (ingat σ′(z) = σ(z)(1 − σ(z)))

- Delta keluaran: δ2 = (ŷ − y) · ŷ(1 − ŷ) = (−0,4228)(0,2440) = −0,1032
- Gradien w2: ∂L/∂w2 = δ2 · h = −0,1032 × 0,6225 = −0,0642
- Delta tersembunyi: δ1 = w2 · δ2 · h(1 − h) = 0,5 × (−0,1032) × 0,2350 = −0,0121
- Gradien w1: ∂L/∂w1 = δ1 · x = −0,0121

**Langkah 3 — perbarui bobot**

| Parameter | Nilai lama | Gradien | Nilai baru = lama − η × gradien |
| --- | --- | --- | --- |
| w2 | 0,5 | −0,0642 | 0,5321 |
| b2 | 0 | −0,1032 | 0,0516 |
| w1 | 0,5 | −0,0121 | 0,5061 |
| b1 | 0 | −0,0121 | 0,0061 |

**Cek hasil**: dengan bobot baru, forward propagation menghasilkan ŷ = 0,5949 dan L = 0,0820. Loss turun dari 0,0894 ke 0,0820, jadi prediksi bergerak mendekati target 1.

## 6. Jenis-Jenis JST dan Kaitannya dengan Pengolahan Sinyal

Arsitektur JST dipilih sesuai bentuk datanya. Data tabel cocok dengan MLP, gambar dengan CNN, dan data berurutan seperti sinyal atau teks dengan RNN atau Transformer.

| Jenis | Ide utama | Cocok untuk data | Contoh aplikasi |
| --- | --- | --- | --- |
| Perceptron | Satu neuron dengan fungsi step | Masalah terpisah linier | Gerbang logika AND/OR |
| MLP (Multilayer Perceptron) | Beberapa lapisan neuron terhubung penuh | Tabel / vektor fitur | Prediksi harga, klasifikasi fitur sinyal |
| CNN (Convolutional Neural Network) | Filter kecil digeser sepanjang data (konvolusi) | Gambar (2D), sinyal (1D) | Pengenalan wajah, deteksi aritmia EKG |
| RNN (Recurrent Neural Network) | Keluaran langkah sebelumnya diumpan kembali sebagai memori | Deret waktu, teks | Prediksi deret waktu, pengenalan suara |
| LSTM / GRU | RNN dengan “gerbang” yang mengatur apa yang diingat dan dilupakan | Deret panjang | Prediksi beban listrik, terjemahan |
| Autoencoder | Memampatkan data lalu merekonstruksinya | Data tanpa label | Denoising sinyal, deteksi anomali |
| Transformer | Mekanisme attention: setiap bagian data melihat semua bagian lain | Teks, suara, gambar | Model bahasa besar, pengenalan suara |

### 6.1 CNN dan konvolusi

Lapisan konvolusi pada CNN menjalankan operasi yang sama dengan konvolusi diskrit di materi sebelumnya. Kernel (filter) berperan seperti h\[n\], sedangkan sinyal masukan berperan seperti x\[n\]:

```math
y[n] = \sum_{k=0}^{K-1} w[k]\, x[n+k] + b
```

Secara teknis, kebanyakan pustaka deep learning menghitung **korelasi silang** (tanpa flip). Namun karena bobot kernel dipelajari dari data, hasil akhirnya setara. Bedanya dengan filter klasik: pada pengolahan sinyal biasa, koefisien filter dirancang manusia. Pada CNN, koefisien filter **dipelajari otomatis** dari data.

### 6.2 Contoh penerapan JST pada sinyal

- **Klasifikasi sinyal EKG**: CNN 1D mendeteksi detak jantung normal dan aritmia langsung dari sampel sinyal.
- **Pengenalan suara**: sinyal audio diubah menjadi spektrogram dengan FFT, lalu diproses CNN atau Transformer. Di sini materi transformasi Fourier dan JST bertemu.
- **Denoising sinyal**: autoencoder belajar memetakan sinyal berderau ke sinyal bersih.
- **Prediksi deret waktu**: LSTM memprediksi nilai berikutnya dari sinyal sensor, misalnya beban listrik atau getaran mesin.

## 7. Overfitting, Latihan Soal, dan Ringkasan

### 7.1 Overfitting dan underfitting

Data biasanya dibagi menjadi tiga: **data latih** (untuk memperbarui bobot), **data validasi** (untuk memantau dan memilih model), dan **data uji** (untuk evaluasi akhir). Pembagian yang umum adalah 70% : 15% : 15%.

| Kondisi | Gejala | Penyebab umum | Solusi |
| --- | --- | --- | --- |
| Underfitting | Loss latih dan loss validasi sama-sama tinggi | Model terlalu sederhana, pelatihan terlalu singkat | Tambah neuron atau lapisan, latih lebih lama |
| Pas (good fit) | Loss latih dan validasi rendah dan berdekatan | — | Pertahankan |
| Overfitting | Loss latih sangat rendah, loss validasi tinggi atau naik | Model menghafal data latih, data terlalu sedikit | Tambah data, regularisasi L2, dropout, early stopping |

**Tips pelatihan**:

- Normalisasi masukan, misalnya ke rentang 0–1 atau rata-rata 0 dan simpangan baku 1.
- Gunakan ReLU di lapisan tersembunyi, sigmoid untuk keluaran biner, dan softmax untuk banyak kelas.
- Mulai dengan optimizer Adam dan learning rate 0,001.
- Pakai **early stopping**: hentikan pelatihan saat loss validasi tidak membaik selama beberapa epoch.
- **Dropout** mematikan sebagian neuron secara acak saat pelatihan, sehingga jaringan tidak bergantung pada neuron tertentu.

### 7.2 Latihan soal

1. Sebuah neuron menerima x = (2, −1, 3) dengan w = (0,5; 1; −0,2) dan b = 0,4. Hitung z dan keluarannya jika memakai ReLU.
2. Ulangi soal 1 dengan fungsi aktivasi sigmoid.
3. Berapa jumlah parameter (bobot + bias) pada MLP berukuran 4–5–3 (4 masukan, 5 neuron tersembunyi, 3 keluaran)?
4. Hitung keluaran softmax untuk z = (2, 1, 0).
5. Hitung binary cross-entropy untuk satu data dengan target y = 1 dan prediksi ŷ = 0,8.
6. Diketahui w = 1,0, ∂L/∂w = 0,4, dan η = 0,1. Berapa nilai w setelah satu langkah gradient descent?
7. Periksa apakah perceptron dengan w1 = 1, w2 = 1, b = −0,5 (step: 1 jika z ≥ 0) dapat meniru gerbang OR.
8. Lapisan konvolusi 1D menerima sinyal 100 sampel dengan kernel 5, stride 1, tanpa padding. Berapa panjang keluarannya?
9. Mengapa perceptron satu lapis tidak dapat menyelesaikan XOR? Bagaimana cara mengatasinya?
10. Sebuah model punya loss latih 0,02 dan loss validasi 0,85. Apa diagnosisnya dan apa dua solusinya?

### 7.3 Kunci jawaban

| No. | Jawaban |
| --- | --- |
| 1 | z = 1 − 1 − 0,6 + 0,4 = −0,2; ReLU(−0,2) = 0 |
| 2 | σ(−0,2) = 0,4502 |
| 3 | (4×5 + 5) + (5×3 + 3) = 25 + 18 = 43 parameter |
| 4 | (0,665; 0,245; 0,090) |
| 5 | L = −ln(0,8) = 0,2231 |
| 6 | w = 1,0 − 0,1 × 0,4 = 0,96 |
| 7 | Bisa. (0,0) → z = −0,5 → 0; (0,1) dan (1,0) → z = 0,5 → 1; (1,1) → z = 1,5 → 1 |
| 8 | 100 − 5 + 1 = 96 sampel |
| 9 | Titik XOR tidak dapat dipisahkan oleh satu garis lurus. Solusinya MLP dengan minimal satu lapisan tersembunyi dan aktivasi nonlinier |
| 10 | Overfitting. Solusi: tambah data, dropout, regularisasi L2, atau early stopping |

### 7.4 Ringkasan

- Neuron tiruan menghitung z = Σ wi·xi + b lalu y = f(z); bobot dan bias adalah hal yang dipelajari.
- Fungsi aktivasi nonlinier (ReLU, sigmoid, tanh, softmax) membuat JST mampu memodelkan pola rumit.
- Pelatihan = forward propagation → hitung loss → backpropagation → perbarui bobot dengan gradient descent, diulang banyak epoch.
- Perceptron satu lapis hanya untuk masalah terpisah linier; MLP dengan lapisan tersembunyi dapat menyelesaikan XOR.
- CNN memakai konvolusi, sehingga sangat cocok untuk sinyal dan gambar; pantau loss validasi untuk mencegah overfitting.
