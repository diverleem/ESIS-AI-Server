# Panduan Deployment Face Recognition Model ke RunPod (E.S.I.S)

Dokumen ini menjelaskan arsitektur dan langkah-langkah lengkap untuk melakukan uji coba dan deployment program **Face Recognition Multi-Kamera** (`main.py`) ke cloud GPU **RunPod** menggunakan protokol **WebRTC (WHEP)** berlatensi sangat rendah (sub-second latency).

---

## Update Utama: WebRTC (WHEP) Ingestion & Pergantian Kamera Lab OS

1. **Streaming Langsung via WebRTC (WHEP)**:
   - Input video di-ingest dan di-decode langsung oleh Python (`aiortc` & `PyAV`).
   - Latensi **sub-second (< 500ms)**, jauh lebih cepat dibandingkan HLS (delay 2–6 detik).
   - Endpoint WHEP MediaMTX dapat diakses langsung oleh RunPod melalui internet publik tanpa perlu port forwarding atau tunnel Pinggy:
     * **Lab Sistem Digital**: `https://stream.ucim.my.id/ecam_lab_sistem_digital/whep`
     * **Lab OS**: `https://stream.ucim.my.id/ecam_lab_os/whep`
2. **Pergantian Kamera Lab Riset ke Lab OS**:
   - Kamera Lab Riset Lantai 5 menggunakan codec video H.265 (HEVC) yang tidak didukung oleh WebRTC standar.
   - Kamera kedua telah dialihkan ke **Lab OS** (`ecam_lab_os`) yang menggunakan H.264 sehingga **100% kompatibel dan lancar** dengan WebRTC.

---

## 1. Struktur Berkas di Direktori Deployment

Semua berkas yang dibutuhkan untuk deployment ke RunPod berada di direktori `AI Server/Face Model/Model Output/`:

| Nama File | Deskripsi |
| :--- | :--- |
| **`main.py`** | Script inferensi AI multi-kamera (Lab Sisdig & Lab OS) via WebRTC WHEP + Web Streaming Server (MJPEG) bawaan untuk RunPod. |
| **`facenet.onnx`** | Model ekstraksi embedding wajah 512 dimensi FaceNet (Inception-ResNet-v1) yang telah dioptimalkan dengan ONNX Runtime GPU. |
| **`data_wajah_facenet_mtcnn.npz`** | Basis data vektor embedding wajah mahasiswa/volunteer (397 vektor, 512 dimensi L2-normalized). |
| **`requirements.txt`** | Daftar dependency Python (OpenCV headless, ONNX Runtime GPU, MTCNN, TensorFlow, NumPy, aiortc, aiohttp, av). |
| **`Dockerfile`** | Berkas kontainerisasi Docker jika ingin deploy menggunakan Custom Container image di RunPod. |

---

## 2. Langkah Demi Langkah Deploy ke RunPod

### Langkah 1: Sewa Pod di RunPod
1. Login ke akun [RunPod.io](https://www.runpod.io/) -> buka menu **Pods** -> klik **+ Deploy**.
2. Pilih GPU (misal: **RTX 3070**, **RTX 4000 Ada**, **RTX 3080**, atau **RTX 3090**).
3. Template: pilih **RunPod PyTorch 2.1**.
4. **Port Configuration**: Tambahkan port **`8000`** di bagian HTTP Ports agar dashboard streaming bisa diakses langsung melalui browser.
5. Klik **Deploy On-Demand**.

### Langkah 2: Upload File ke Pod
1. Pada Pod yang sedang berjalan, klik **Connect** -> pilih **JupyterLab**.
2. Upload berkas berikut ke folder `/workspace`:
   - `main.py`
   - `facenet.onnx`
   - `data_wajah_facenet_mtcnn.npz`
   - `requirements.txt`

### Langkah 3: Install Dependencies di Terminal Pod
Buka **Web Terminal** di RunPod dan jalankan:
```bash
cd /workspace
pip install -r requirements.txt
```

### Langkah 4: Jalankan Program AI
* **Jalankan Default (WebRTC WHEP Lab Sisdig & Lab OS):**
  ```bash
  python main.py --port 8000
  ```
* **Jalankan dengan Mode Fallback Pinggy HLS (Opsional):**
  ```bash
  python main.py --pinggy "https://subdomain-anda.a.pinggy.link" --port 8000
  ```
* **Jalankan Mode Simulasi (Mock Test tanpa CCTV fisik):**
  ```bash
  python main.py --mock --port 8000
  ```

### Langkah 5: Buka Video Stream di Browser
1. Di dashboard RunPod, klik tombol **Connect** pada Pod Anda.
2. Di bagian **HTTP Service (Port 8000)**, klik link atau buka di browser:
   `https://<POD_ID>-8000.proxy.runpod.net/`
3. Video live kedua kamera beserta kotak deteksi hijau, nama mahasiswa, skor kemiripan (%), dan FPS akan tampil secara real-time!

---

## 3. Daftar Endpoint Streaming HTTP MJPEG

Server streaming bawaan menyediakan beberapa endpoint video yang dapat langsung disematkan ke dashboard atau frontend web:

- **Dashboard Lengkap**: `http://<IP_OR_PROXY>:8000/`
- **Stream Kamera 1 (Lab Sisdig)**: `http://<IP_OR_PROXY>:8000/video_feed/1`
- **Stream Kamera 2 (Lab OS)**: `http://<IP_OR_PROXY>:8000/video_feed/2`
- **Stream Mosaic (Side-by-Side Dual Camera)**: `http://<IP_OR_PROXY>:8000/video_feed/mosaic`
- **JSON API Status & Riwayat Deteksi**: `http://<IP_OR_PROXY>:8000/api/status`
