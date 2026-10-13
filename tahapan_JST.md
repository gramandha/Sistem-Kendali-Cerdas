# Dokumentasi Proses Jaringan Syaraf Tiruan (ANN)

Dokumen ini menjelaskan alur kerja lengkap pelatihan Jaringan Syaraf Tiruan (Artificial Neural Network) yang terdiri dari dua fase utama: **Forward Propagation (Feedforward)** dan **Backward Propagation (Propagasi Mundur)**.

---

## 📋 Alur Kerja Utama Pelatihan JST (Overall Workflow)

Proses pelatihan JST menggunakan metode iteratif yang disebut **Epoch**. Satu Epoch terdiri dari urutan langkah berikut:

```text
[ 1. Inisialisasi ] ──> [ 2. Forward Propagation ] ──> [ 3. Hitung Loss (Error) ]
                                                                 │
[ 5. Iterasi Berikutnya ] <── [ 4. Update Parameter ] <── [ 4. Backpropagation ]
```

1. **Inisialisasi:** Mengisi bobot (*weights*) dan bias awal dengan nilai acak kecil.
2. **Forward Propagation:** Mengalirkan data input ke depan untuk menghasilkan prediksi.
3. **Hitung Loss:** Mengukur seberapa jauh prediksi model dari target asli menggunakan fungsi eror.
4. **Backpropagation:** Mengalirkan nilai eror ke belakang untuk menghitung gradien (kontribusi kesalahan) setiap bobot dan bias.
5. **Update Parameter:** Memperbarui nilai bobot dan bias berdasarkan gradien yang ditemukan untuk meminimalkan eror pada iterasi berikutnya.

---

## 1. Forward Propagation (Feedforward)

Forward propagation adalah proses di mana fitur masukan (*input*) diproses maju melewati setiap lapisan jaringan hingga menghasilkan sebuah nilai prediksi (*output*).

### Tahapan Proses:
*   **Langkah 1 (Input):** Lapisan input menerima data fitur dari dataset ($x_1, x_2, \dots, x_n$).
*   **Langkah 2 (Kombinasi Linear):** Pada setiap neuron di lapisan tersembunyi (*hidden layer*) dan keluaran (*output layer*), dilakukan operasi perkalian matriks antara input dan bobot (*weight*), kemudian ditambahkan dengan *bias*.
   
    $$\mathbf{z} = \sum (x_i \cdot w_i) + b$$

*   **Langkah 3 (Aktivasi Non-Linear):** Hasil penjumlahan linear ($z$) dimasukkan ke dalam fungsi aktivasi non-linear (seperti *Sigmoid*, *ReLU*, atau *Tanh*) untuk menghasilkan nilai aktivasi ($a$). Langkah ini memberikan kemampuan bagi JST untuk mempelajari pola rumit non-linear.
   
    $$a = \sigma(z)$$

*   **Langkah 4 (Output):** Langkah 2 dan 3 terus berlanjut dari satu lapisan ke lapisan berikutnya hingga menghasilkan nilai prediksi akhir ($\hat{y}$).

---

## 2. Menghitung Evaluasi Kesalahan (Loss Function)

Setelah mendapatkan hasil prediksi ($\hat{y}$), jaringan akan menghitung seberapa besar kesalahan prediksinya dengan membandingkannya terhadap target asli ($y$). Salah satu fungsi kerugian yang umum digunakan adalah **Mean Squared Error (MSE)** untuk satu sampel data:

$$L = \frac{1}{2}(y - \hat{y})^2$$

---

## 3. Backward Propagation (Propagasi Mundur)

Backward propagation adalah proses menghitung gradien kesalahan (*loss gradient*) dari lapisan paling akhir (output) mundur menuju lapisan paling awal (input). Proses ini menggunakan konsep **Aturan Rantai (Chain Rule)** dalam kalkulus untuk mencari turunan parsial dari fungsi *loss*.

### Tahapan Proses:

### 🔹 Langkah A: Hitung Eror di Output Layer ($\delta_{output}$)
Mencari tahu seberapa sensitif fungsi *loss* ($L$) terhadap perubahan nilai linear ($z$) di lapisan output.
$$\delta_{output} = \frac{\partial L}{\partial z_{output}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}} {\partial z_{output}}$$

### 🔹 Langkah B: Hitung Eror di Hidden Layer ($\delta_{hidden}$)
Mengalirkan nilai kesalahan dari lapisan output kembali ke lapisan tersembunyi di belakangnya dengan mengalikan nilai eror output dengan bobot penghubungnya.
$$\delta_{hidden} = \frac{\partial L}{\partial z_{hidden}} = (\delta_{output} \cdot w_{kombinasi}) \cdot \frac{\partial a_{hidden}}{\partial z_{hidden}}$$

### 🔹 Langkah C: Hitung Gradien Bobot dan Bias
Setelah nilai eror ($\delta$) pada lapisan terkait ditemukan, kita hitung turunan parsial akhir untuk mengetahui seberapa besar kontribusi komponen parameter tersebut terhadap error:
*   **Gradien Bobot:** $\frac{\partial L}{\partial w} = a_{sebelumnya} \cdot \delta$
*   **Gradien Bias:** $\frac{\partial L}{\partial b} = \delta$

---

## 4. Pembaruan Parameter (Weight & Bias Update)

Setelah semua nilai gradien ditemukan, nilai bobot dan bias yang lama diperbarui menggunakan algoritma optimasi (misalnya *Gradient Descent*) yang dikalikan dengan konstanta kecepatan belajar bernama **Learning Rate ($\alpha$)**:

*   **Pembaruan Bobot:**  
    $$w_{baru} = w_{lama} - \alpha \cdot \frac{\partial L}{\partial w}$$
*   **Pembaruan Bias:**  
    $$b_{baru} = b_{lama} - \alpha \cdot \frac{\partial L}{\partial b}$$

---
*Proses dari Forward Propagation hingga Pembaruan Parameter di atas akan terus diulang secara konsisten hingga nilai kesalahan (Loss) mencapai titik minimum yang diinginkan atau model telah konvergen.*
