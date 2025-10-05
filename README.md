# **Tugas Individu: Implementasi Arsitektur Transformer dari Nol dengan NumPy** </br>
Mengimplementasikan arsitektur Transformer from scratch menggunakan NumPy, tanpa bantuan library deep learning seperti PyTorch atau TensorFlow untuk membangun alur forward pass dari sebuah decoder-only Transformer (GPT-style), mulai dari embedding hingga menghasilkan distribusi probabilitas untuk prediksi token berikutnya.

## Overview
### Desain Arsitektur </br>
<img width="860" height="74" alt="Screenshot 2025-10-04 160700" src="https://github.com/user-attachments/assets/45046db2-4f62-4ead-957a-53090f26fd43" />

- **Token Embedding** </br>
Mengonversi token ID ke vektor dense menggunakan matriks dan diinisialisasi dengan distribusi normal (scale 0.02) untuk stabilitas. </br>
- **Sinusoidal Positional Encoding** </br>
Ditambahkan ke embedding untuk menyuntikkan informasi posisi. </br>
- **Decoder Blocks** </br>
Setiap block menggunakan pre-norm architecture: LayerNorm → Multi-Head Attention (n_heads=8, head_dim=8) dengan causal mask → residual add → LayerNorm → Feed-Forward Network (d_ff=64, aktivasi GELU) → residual add. </br>
- **Multi-Head Attention** </br>
Proyeksi linear Q/K/V, split ke heads, scaled dot-product attention, concat, dan proyeksi. </br>
- **Feed-Forward Network** </br>
Dua lapisan linear dengan bias dan GELU untuk non-linearitas. </br> 
- **Layer Normalization** </br>
Pre-norm dengan γ= 1, β = 0, dan ε = 1e-5 dan untuk normalisasi per token. . 
- **Output Layer** </br>
Proyeksi akhir ke vocab_size (100) menghasilkan logits [batch, seqlen, vocab] , diikuti softmax hanya pada posisi terakhir untuk probs_next. </br>


### Rumus Implementasi Positional Encoding Sinusoidal
<img width="953" height="392" alt="image" src="https://github.com/user-attachments/assets/4df3fa73-37d6-4631-ab51-d4ec21b7ad14" />
**Keterangan:** 
- *pos* = posisi token dalam sequence (0, 1, 2, …).
- *i* = indeks dimensi (0, 1, 2, …).
- *dmodel*​ = ukuran dimensi embedding (misalnya 64).
- Untuk dimensi genap (*2i*) digunakan sinus, sedangkan untuk dimensi ganjil (*2i+1*) digunakan cosinus.

### Output Program</br>
<img width="566" height="184" alt="Screenshot 2025-10-04 163937" src="https://github.com/user-attachments/assets/04238ec4-2ec8-45f5-be21-4822eb4fe4a0" />

### Kesimpulan
Implementasi arsitektur Transformer decoder-only sederhana dengan komponen utama, yaitu token embedding, positional encoding sinusoidal, scaled dot-product attention, causal masking, multi-head self-attention, feed-forward network, residual connection + layer normalization, dan output layer. Model menerima input token [1, 3, 5, 7, 9, 11] dengan seq_len=6 dan menghasilkan output berupa logits berukuran [1, 6, vocab_size] serta distribusi probabilitas token berikutnya [1, vocab_size]. Hasil pengujian menunjukkan dimensi konsisten, distribusi softmax valid (jumlah = 1.0), dan causal mask bekerja benar untuk menjaga sifat autoregressive.


