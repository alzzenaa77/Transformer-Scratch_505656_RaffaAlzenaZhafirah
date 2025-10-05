# **Tugas Individu: Implementasi Arsitektur Transformer dari Nol dengan NumPy** </br>
Mengimplementasikan arsitektur Transformer from scratch menggunakan NumPy, tanpa bantuan library deep learning seperti PyTorch atau TensorFlow untuk membangun alur forward pass dari sebuah decoder-only Transformer (GPT-style), mulai dari embedding hingga menghasilkan distribusi probabilitas untuk prediksi token berikutnya.

## Overview
a. Token Embedding: Mengonversi token ID ke vektor dense menggunakan matriks dan diinisialisasi dengan distribusi normal (scale 0.02) untuk stabilitas. </br>
b. Sinusoidal Positional Encoding: Ditambahkan ke embedding untuk menyuntikkan informasi posisi. </br>
c. Decoder Blocks (n_layers=4): Setiap block menggunakan pre-norm architecture: LayerNorm → Multi-Head Attention (n_heads=8, head_dim=8) dengan causal mask → residual add → LayerNorm → Feed-Forward Network (d_ff=64, aktivasi GELU) → residual add. </br>
c. Multi-Head Attention: Proyeksi linear Q/K/V, split ke heads, scaled dot-product attention, concat, dan proyeksi. </br>
d. Feed-Forward Network: Dua lapisan linear dengan bias dan GELU untuk non-linearitas. </br> 
e. Layer Normalization: Pre-norm dengan γ= 1, β = 0, dan ε = 1e-5 dan untuk normalisasi per token. . 
f. Output Layer: Proyeksi akhir ke vocab_size (100) menghasilkan logits [batch, seqlen, vocab] , diikuti softmax hanya pada posisi terakhir untuk probs_next. </br>

## Desain Arsitektur </br>
<img width="860" height="74" alt="Screenshot 2025-10-04 160700" src="https://github.com/user-attachments/assets/45046db2-4f62-4ead-957a-53090f26fd43" />

## Output Program</br>
<img width="566" height="184" alt="Screenshot 2025-10-04 163937" src="https://github.com/user-attachments/assets/04238ec4-2ec8-45f5-be21-4822eb4fe4a0" />

