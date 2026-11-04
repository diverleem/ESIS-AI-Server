# Panduan Deployment Face Recognition Model ke RunPod (E.S.I.S)

Dokumen ini menjelaskan arsitektur dan langkah-langkah lengkap untuk melakukan uji coba dan deployment program **Face Recognition Multi-Kamera** (`main.py`) ke cloud GPU **RunPod**.

---

## 1. Struktur File di Folder Model Output

Semua berkas yang dibutuhkan untuk deployment sudah disiapkan di direktori `AI Server/Face Model/Model Output/`:

| Nama File | Deskripsi |
| :--- | :--- |
| **`main.py`** | Script utama inferensi AI multi-camera (Lab Riset Lt 5 & Lab Sisdig) dilengkapi web streaming server (MJPEG) bawaan untuk menampilkan video di RunPod. |
| **`facenet.onnx`** | Model ekstraksi embedding wajah 512 dimensi FaceNet (Inception-ResNet-v1) yang telah dioptimalkan untuk inferensi cepat dengan ONNX Runtime GPU. |
| **`data_wajah_facenet_mtcnn.npz`** | Basis data vektor embedding seluruh mahasiswa/volunteer (397 vektor, 512 dimensi L2-normalized). |
| **`requirements.txt`** | Daftar dependency Python (OpenCV headless, ONNX Runtime GPU, MTCNN, TensorFlow, NumPy). |
| **`Dockerfile`** | Berkas kontainerisasi Docker jika ingin deploy menggunakan Custom Container image di RunPod. |

---

## 2. Catatan Penting: Koneksi Jaringan CCTV (Lokal vs Cloud)

Kamera CCTV Anda menggunakan alamat IP privat lokal kampus (`172.20.32.50` dan `172.20.32.43`):
- **Lab Riset Lt 5**: `rtsp://admin:EYQMDK@172.20.32.50:554/Streaming/Channels/101`
- **Lab Sisdig**: `rtsp://admin:JOHVAF@172.20.32.43:554/Streaming/Channels/101`

Karena RunPod berada di cloud publik (di luar jaringan lokal kampus PLT), koneksi RTSP langsung ke IP privat `172.20.32.x` tidak akan tembus kecuali menggunakan salah satu solusi berikut:

### Pilihan Solusi Akses Kamera di RunPod:
1. **Mode Uji Coba Cepat (Mock/Simulasi):**
   Gunakan flag `--mock` saat menjalankan `main.py`. Program akan membuat feed video simulasi CCTV sintetis secara real-time untuk memastikan model FaceNet, detektor, GPU, dan web streaming di RunPod berfungsi 100%.
2. **Menggunakan Tailscale / WireGuard (Direkomendasikan untuk CCTV Riil):**
   Pasang [Tailscale](https://tailscale.com/) di server lokal kampus dan pasang juga di pod RunPod Anda. Keduanya akan berada dalam satu virtual LAN terenkripsi, sehingga RunPod dapat mengakses RTSP kampus secara langsung.
3. **MediaMTX Relay / Reverse Tunnel (Ngrok/Cloudflare):**
   Forward port RTSP MediaMTX dari kampus ke URL publik menggunakan tunnel, lalu masukkan URL tersebut ke parameter `--cam1` dan `--cam2`.

---

## 3. Langkah Demi Langkah Deploy ke RunPod

### Metode 1: Menggunakan Standard GPU Pod (Paling Cepat & Praktis)

1. **Masuk ke RunPod & Sewa Pod:**
   - Login ke akun [RunPod.io](https://www.runpod.io/).
   - Buka menu **Pods** -> klik **+ Deploy**.
   - Pilih GPU hemat (misal: **RTX 3070**, **RTX 4000 Ada**, **RTX 3080**, atau **RTX 3090**).
   - Pada bagian **Template**, pilih template resmi: `RunPod PyTorch 2.1` atau `RunPod PyTorch 2.2`.
   - Klik **Edit Template / Port Configuration**:
     - Tambahkan port HTTP: `8000` (atau port yang Anda inginkan) agar dapat diakses dari browser luar.
   - Klik **Deploy On-Demand** atau **Deploy Spot**.

2. **Upload File Model ke RunPod:**
   Setelah Pod berstatus `Running`, Anda memiliki beberapa cara untuk mengunggah file:
   - **Cara A (Web Terminal / Jupyter Lab):**
     Klik tombol **Connect** -> buka **JupyterLab** -> Upload berkas:
     - `main.py`
     - `facenet.onnx`
     - `data_wajah_facenet_mtcnn.npz`
     - `requirements.txt`
   - **Cara B (Zip & Download langsung di terminal Pod):**
     Bisa kompres folder `Model Output` menjadi zip, upload ke Google Drive / GitHub release, lalu unduh dengan `wget` di terminal RunPod.

3. **Install Dependencies di Terminal Pod:**
   Buka **Web Terminal** di RunPod dan jalankan perintah:
   ```bash
   cd /workspace
   pip install -r requirements.txt
   ```

4. **Jalankan Program:**

   * **Untuk Uji Coba Pertama (Simulasi / Mock Mode):**
     ```bash
     python main.py --mock --port 8000
     ```

   * **Untuk Menghubungkan ke CCTV Asli (Jika Jaringan Sudah Terhubung / Tunneling Aktif):**
     ```bash
     python main.py \
       --cam1 "rtsp://admin:EYQMDK@172.20.32.50:554/Streaming/Channels/101" \
       --cam2 "rtsp://admin:JOHVAF@172.20.32.43:554/Streaming/Channels/101" \
       --port 8000 \
       --threshold 0.75
     ```

5. **Melihat Output Video di Browser:**
   - Di dashboard RunPod, klik tombol **Connect** pada Pod Anda.
   - Pada bagian **HTTP Service (Port 8000)**, klik **Connect to HTTP Port [8000]** atau buka URL:
     `https://<POD_ID>-8000.proxy.runpod.net/`
   - Tampilan antarmuka dashboard modern E.S.I.S akan terbuka secara live di browser Anda!

---

## 4. Endpoint Streaming Video yang Tersedia

Server web bawaan di `main.py` menyediakan beberapa rute yang dapat dibuka langsung di browser maupun diintegrasikan ke frontend Next.js:

| Endpoint | Fungsi |
| :--- | :--- |
| **`/`** | Dashboard interaktif monitoring dual kamera, metrik FPS, latensi AI, dan riwayat wajah teridentifikasi. |
| **`/video_feed/1`** | Stream video MJPEG murni Kamera 1 (Lab Riset Lantai 5). |
| **`/video_feed/2`** | Stream video MJPEG murni Kamera 2 (Lab Sistem Digital). |
| **`/video_feed/mosaic`** | Stream video MJPEG gabungan kedua kamera berdampingan (*Side-by-Side*). |
| **`/api/status`** | API JSON real-time berisi status koneksi kamera, FPS, inferensi AI, dan daftar mahasiswa yang terdeteksi saat ini. |

---

## 5. Parameter CLI `main.py` Lengkap

| Argumen | Default | Keterangan |
| :--- | :--- | :--- |
| `--cam1` | RTSP Riset Lt 5 | RTSP URL atau device index untuk Lab Riset Lt 5 |
| `--cam2` | RTSP Sisdig | RTSP URL atau device index untuk Lab Sisdig |
| `--port` | `8000` | Port HTTP Web Server (sesuaikan dengan port RunPod) |
| `--threshold` | `0.75` | Batas jarak Euclidean FaceNet (0.70 - 0.75 optimal) |
| `--detector` | `auto` | Pilihan detektor wajah: `auto`, `mtcnn`, atau `haar` |
| `--device` | `auto` | Pilihan komputasi: `auto` (GPU CUDA jika ada), `cpu` |
| `--mock` | `False` | Jalankan simulasi CCTV tanpa perlu RTSP fisik |
| `--display` | `False` | Tampilkan jendela OpenCV GUI lokal (hanya jika desktop monitor ada) |
