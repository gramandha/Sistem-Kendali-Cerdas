# Studi Kasus Perhitungan Manual JST (Aktivasi ReLU)

Mari kita lakukan simulasi perhitungan manual pada sebuah jaringan saraf tiruan sederhana dengan arsitektur:
*   **1 Input Neuron ($x$)**
*   **1 Hidden Neuron ($h$)**
*   **1 Output Neuron ($\hat{y}$)**

Pada studi kasus ini, kita menggunakan **Fungsi Aktivasi ReLU (Rectified Linear Unit)**:
$$\text{ReLU}(z) = \max(0, z)$$

Serta turunan fungsinya untuk proses backpropagation:
$$\text{ReLU}'(z) = \begin{cases} 1 & \text{jika } z > 0 \\ 0 & \text{jika } z \le 0 \end{cases}$$

Konstanta **Learning Rate ($\alpha$) ditetapkan sebesar 0.1**.

---

## 1. Data Awal & Inisialisasi Parameter

Misalkan kita memiliki satu pasang data pelatihan:
*   **Input ($x$):** $2.0$
*   **Target Aktual ($y$):** $1.5$

### Bobot & Bias Awal (Ditetapkan Acak):
*   **Hidden Layer:** Bobot ($w_1$) = $0.5$, Bias ($b_1$) = $0.2$
*   **Output Layer:** Bobot ($w_2$) = $0.8$, Bias ($b_2$) = $0.1$

---

## 2. Fase Forward Propagation (Feedforward)

### 🔹 Langkah 1: Hitung Output di Hidden Neuron ($h$)
1. **Kombinasi Linear ($z_1$):**
   $$z_1 = (x \cdot w_1) + b_1$$
   $$z_1 = (2.0 \cdot 0.5) + 0.2 = 1.0 + 0.2 = 1.2$$

2. **Aktivasi ReLU ($a_1$ atau nilai $h$):**
   Karena $z_1 = 1.2$ (lebih besar dari 0), maka nilainya tetap.
   $$a_1 = \max(0, 1.2) = 1.2$$

### 🔹 Langkah 2: Hitung Output di Output Neuron ($\hat{y}$)
1. **Kombinasi Linear ($z_2$):**
   Input untuk lapisan ini adalah nilai aktivasi dari hidden layer ($a_1$).
   $$z_2 = (a_1 \cdot w_2) + b_2$$
   $$z_2 = (1.2 \cdot 0.8) + 0.1 = 0.96 + 0.1 = 1.06$$

2. **Aktivasi ReLU ($a_2$ atau nilai prediksi $\hat{y}$):**
   Karena $z_2 = 1.06$ (lebih besar dari 0), maka nilainya tetap.
   $$\hat{y} = a_2 = \max(0, 1.06) = 1.06$$

### 🔹 Langkah 3: Hitung Nilai Loss (Error)
Menggunakan rumus kuadrat kesalahan tunggal (MSE):
$$L = \frac{1}{2}(y - \hat{y})^2 = \frac{1}{2}(1.5 - 1.06)^2 = \frac{1}{2}(0.44)^2 = 0.0968$$

---

## 3. Fase Backward Propagation (Propagasi Mundur)

Kita mencari kontribusi eror ($\delta$) menggunakan *chain rule*. Karena nilai $z_1$ dan $z_2$ keduanya positif ($> 0$), nilai turunan ReLU ($\text{ReLU}'$) untuk kedua lapisan adalah **1**.

### 🔹 Langkah A: Hitung Eror di Output Layer ($\delta_{output}$)
$$\delta_{output} = -(y - \hat{y}) \cdot \text{ReLU}'(z_2)$$
$$\delta_{output} = -(1.5 - 1.06) \cdot 1 = -0.44$$

### 🔹 Langkah B: Hitung Gradien Parameter Output Layer ($w_2$ dan $b_2$)
*   **Gradien Bobot $w_2$:**
    $$\frac{\partial L}{\partial w_2} = a_1 \cdot \delta_{output} = 1.2 \cdot (-0.44) = -0.528$$
*   **Gradien Bias $b_2$:**
    $$\frac{\partial L}{\partial b_2} = \delta_{output} = -0.44$$

### 🔹 Langkah C: Hitung Eror di Hidden Layer ($\delta_{hidden}$)
Mengalirkan eror dari output layer mundur ke hidden layer melalui bobot $w_2$ dan dikalikan turunan aktivasi hidden layer.
$$\delta_{hidden} = (\delta_{output} \cdot w_2) \cdot \text{ReLU}'(z_1)$$
$$\delta_{hidden} = (-0.44 \cdot 0.8) \cdot 1 = -0.352$$

### 🔹 Langkah D: Hitung Gradien Parameter Hidden Layer ($w_1$ dan $b_1$)
*   **Gradien Bobot $w_1$:**
    $$\frac{\partial L}{\partial w_1} = x \cdot \delta_{hidden} = 2.0 \cdot (-0.352) = -0.704$$
*   **Gradien Bias $b_1$:**
    $$\frac{\partial L}{\partial b_1} = \delta_{hidden} = -0.352$$

---

## 4. Pembaruan Parameter (Gradient Descent)

Sekarang kita perbarui semua bobot dan bias awal menggunakan **Learning Rate ($\alpha$) = 0.1**.

###  Perbaikan di Output Layer:
*   $$w_{2\_baru} = w_2 - (\alpha \cdot \frac{\partial L}{\partial w_2}) = 0.8 - (0.1 \cdot -0.528) = 0.8 + 0.0528 = \mathbf{0.8528}$$
*   $$b_{2\_baru} = b_2 - (\alpha \cdot \frac{\partial L}{\partial b_2}) = 0.1 - (0.1 \cdot -0.44) = 0.1 + 0.044 = \mathbf{0.144}$$

###  Perbaikan di Hidden Layer:
*   $$w_{1\_baru} = w_1 - (\alpha \cdot \frac{\partial L}{\partial w_1}) = 0.5 - (0.1 \cdot -0.704) = 0.5 + 0.0704 = \mathbf{0.5704}$$
*   $$b_{1\_baru} = b_1 - (\alpha \cdot \frac{\partial L}{\partial b_1}) = 0.2 - (0.1 \cdot -0.352) = 0.2 + 0.0352 = \mathbf{0.2352}$$

---
**Kesimpulan:** Menggunakan aktivasi ReLU membuat perhitungan turunan jauh lebih sederhana (bernilai 1 saat positif). Setelah 1 epoch ini, bobot baru yang disesuaikan akan membuat tebakan model berikutnya menghasilkan nilai $\hat{y}$ yang lebih besar dari $1.06$, bergerak mendekati target asli kita yaitu $1.5$.
