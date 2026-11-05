# Panduan Deployment 4-Camera Face Recognition Node ke RunPod (E.S.I.S)

Dokumen ini menjelaskan arsitektur dan langkah-langkah lengkap untuk melakukan uji coba dan deployment program **Face Recognition 4-Kamera** ([`main.py`](file:///D:/web/esis/AI%20Server/Face%20Model/Model%20Output/main.py)) ke cloud GPU **RunPod** menggunakan protokol **HLS (FFmpeg Native Decoding)** dan **WebRTC (WHEP)**.

---

## Fitur Utama

1. **Dukungan 4 Kamera Sekaligus**:
   - **Kamera 1**: Lab Riset Lantai 5 (`ecam_lab_riset_lt_5` - Codec **H.265 / HEVC**)
   - **Kamera 2**: Lab Sistem Digital (`ecam_lab_sistem_digital` - Codec **H.264 / AVC**)
   - **Kamera 3**: Lab OS (`ecam_lab_os` - Codec **H.264 / AVC**)
   - **Kamera 4**: Lab Programming (`ecam_lab_programming` - Codec **H.264 / AVC**)

2. **Dukungan Penuh Codec H.265 & H.264 via HLS**:
   - Ingestion video menggunakan backend OpenCV FFmpeg yang secara bawaan mendukung decoding video **H.265 (HEVC)** tanpa kendala lisensi WebRTC.
   - Menggunakan tunnel HLS MediaMTX via Pinggy:
     * `https://fdxuqgcmfo.a.pinggy.link/<ecam_path>/index.m3u8`

3. **Mode Simulasi Mandiri (Mock Face Recognition)**:
   - Dilengkapi flag `--mock` yang secara otomatis menanamkan foto volunteer mahasiswa nyata dengan animasi gerakan halus.
   - Memungkinkan pengujian menyeluruh (face detection, face matching, bounding box nama, skor akurasi, dan logging riwayat) langsung di browser tanpa perlu menunggu orang lewat di depan CCTV.

4. **Tampilan Dashboard & Mosaic 2x2 HD**:
   - Grid 2x2 yang menampilkan ke-4 kamera secara bersamaan.
   - Endpoint streaming MJPEG individual (`/video_feed/1`, `/2`, `/3`, `/4`) dan mosaic (`/video_feed/mosaic`).
   - Snapshot API (`/snapshot/1`, `/2`, `/3`, `/4`, `/mosaic`) dan REST status JSON (`/api/status`).

---

## 1. Struktur Berkas Direktori Deployment

Semua berkas yang dibutuhkan berada di direktori `AI Server/Face Model/Model Output/`:

| Nama File | Deskripsi |
| :--- | :--- |
| **`main.py`** | Script inferensi AI 4-kamera dengan HTTP streaming server (MJPEG & 2x2 Mosaic). |
| **`facenet.onnx`** | Model ekstraksi embedding wajah 512 dimensi FaceNet (Inception-ResNet-v1). |
| **`data_wajah_facenet_mtcnn.npz`** | Basis data vektor embedding wajah mahasiswa (397 foto dari 38 identitas). |
| **`requirements.txt`** | Daftar dependency Python (OpenCV headless, ONNX Runtime GPU, MTCNN, TensorFlow, NumPy, aiortc, aiohttp, av). |
| **`Dockerfile`** | Berkas kontainerisasi Docker untuk deployment custom image di RunPod. |

---

## 2. Cara Menjalankan Program

### A. Menjalankan di Komputer Lokal (Laptop / PC)

1. **Jalankan Mode 4 Kamera HLS (Default CCTV Live):**
   ```bash
   cd "AI Server/Face Model/Model Output"
   python main.py --port 8000
   ```

2. **Jalankan Mode Simulasi Penuh (Mock Test Tanpa CCTV):**
   ```bash
   python main.py --mock --port 8000
   ```
   *(Mode ini langsung memunculkan volunteer wajah di ke-4 kamera dan menampilkan kotak deteksi hijau beserta nama mahasiswa).*

3. **Jalankan dengan Pop-up GUI Desktop Lokal:**
   ```bash
   python main.py --display --port 8000
   ```

---

### B. Menjalankan di Cloud GPU RunPod

1. **Sewa Pod**:
   - Pilih GPU hemat: **RTX 3070**, **RTX 3080**, atau **RTX 4000 Ada**.
   - Template: **RunPod PyTorch 2.1**.
   - Tambahkan port **`8000`** pada kolom HTTP Ports.
2. **Upload Berkas**:
   - Buka JupyterLab / Terminal di Pod `/workspace`.
   - Upload berkas `main.py`, `facenet.onnx`, `data_wajah_facenet_mtcnn.npz`, dan `requirements.txt`.
3. **Instal Dependency**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Jalankan Program**:
   ```bash
   python main.py --port 8000
   ```
5. **Buka di Browser**:
   Klik tombol **Connect** di dashboard RunPod -> klik link port **8000**:
   `https://<POD_ID>-8000.proxy.runpod.net/`

---

## 3. Daftar Endpoint Streaming HTTP MJPEG

| Endpoint | Fungsi |
| :--- | :--- |
| **`/`** | Dashboard Web Lengkap (Grid 4 Kamera, Kontrol Tampilan, dan Riwayat Deteksi) |
| **`/video_feed/1`** | Live Stream Kamera 1 (Lab Riset Lt 5) |
| **`/video_feed/2`** | Live Stream Kamera 2 (Lab Sistem Digital) |
| **`/video_feed/3`** | Live Stream Kamera 3 (Lab OS) |
| **`/video_feed/4`** | Live Stream Kamera 4 (Lab Programming) |
| **`/video_feed/mosaic`** | Live Stream 2x2 Grid Mosaic Gabungan 4 Kamera (1280x720) |
| **`/snapshot/<id>`** | Ambil 1 frame gambar JPEG statis (`1`, `2`, `3`, `4`, `mosaic`) |
| **`/api/status`** | Endpoint JSON status kamera, FPS, latensi inferensi, dan daftar nama terdeteksi |
