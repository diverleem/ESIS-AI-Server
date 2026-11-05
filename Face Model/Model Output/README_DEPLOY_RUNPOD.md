# Panduan E.S.I.S Smart Patrol Tour (Main Stream HD Face Recognition)

Program [`main.py`](file:///D:/web/esis/AI%20Server/Face%20Model/Model%20Output/main.py) mengimplementasikan fitur **Smart Patrol Tour (Patroli Bergilir 30 Detik)** menggunakan **Main Stream (1080p / 2K)** untuk memaksimalkan akurasi deteksi dan pengenalan wajah mahasiswa tanpa membebani bandwidth jaringan.

---

## 1. Konsep & Arsitektur Sistem

```
┌────────────────────────────────────────────────────────────────────────┐
│                   SMART PATROL TOUR CONTROLLER                         │
│                                                                        │
│   [Lab Riset Lt 5] ──(30s)──> [Lab Sisdig] ──(30s)──> [Lab OS]...      │
│     (2K QHD H.265)             (720p H.264)             (Main/Sub)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ 1 Stream HD Aktif (~3-4 Mbps)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                ZERO-LATENCY BUFFER-DROP GRABBER                        │
│   * Membuang frame antrian lama secara instan (selalu real-time)       │
│   * Dual-Scale Inference: Deteksi cepat, crop pada resolusi penuh      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             WEB DASHBOARD & MJPEG STREAM (PORT 8000)                   │
│   * Tampilan Live Feed Full HD dengan HUD CCTV & Countdown Timer       │
│   * Tombol Kunci Kamera Manual vs Patroli Otomatis                     │
│   * Pilihan interval patroli: 15s / 30s / 60s                          │
│   * Tabel log riwayat deteksi mahasiswa real-time                      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Cara Menjalankan Program

### A. Mode Simulasi Mandiri (`--mock`) — *Direkomendasikan untuk Uji Coba Cepat*
Menjalankan simulasi patroli lab dengan foto volunteer mahasiswa nyata tanpa membutuhkan koneksi fisik CCTV:
```bash
cd "AI Server/Face Model/Model Output"
python main.py --mock --port 8000
```
> [!TIP]
> Setiap 30 detik (atau interval yang dipilih), sistem akan otomatis berpindah lab dan mengenali wajah volunteer mahasiswa di lab tersebut secara *real-time*.

### B. Mode CCTV Live (Main Stream HD)
Membuka aliran video Mainstream CCTV asli dari lab PLT:
```bash
cd "AI Server/Face Model/Model Output"
python main.py --port 8000
```

*Argumen Tambahan (Opsional):*
* Ubah durasi patroli: `python main.py --interval 45 --port 8000`
* Tampilkan pop-up monitor desktop: `python main.py --display --port 8000`

---

## 3. Fitur Interaktif di Web Dashboard (`http://localhost:8000/`)

1. **Live Patrol Feed**:
   - Menampilkan feed video resolusi tinggi dengan overlay HUD waktu, FPS, latensi, dan sisa waktu patroli (*countdown*).
2. **Kontrol Patroli Otomatis**:
   - Tombol **"🔄 Mulai Patroli Otomatis"**: Berputar antar-lab setiap 30 detik.
   - Pilihan Durasi: **15 Detik**, **30 Detik**, atau **60 Detik**.
3. **Kunci Kamera Manual (*Manual Lock*)**:
   - Klik salah satu tombol lab: **Lab Riset Lantai 5**, **Lab Sisdig**, **Lab OS**, atau **Lab Programming**.
   - Sistem akan berhenti berputar dan mengunci pengawasan pada lab tersebut secara permanen hingga tombol patroli otomatis ditekan kembali.
4. **Tabel Riwayat Deteksi Mahasiswa**:
   - Mencatat mahasiswa yang teridentifikasi di setiap lab secara berurutan lengkap dengan persentase kemiripan (*similarity*).

---

## 4. Daftar Endpoint API

| Endpoint | Metode | Fungsi |
| :--- | :---: | :--- |
| **`/`** | GET | Dashboard Pengawasan SOC E.S.I.S Lengkap |
| **`/video_feed/live`** | GET | MJPEG Live Stream Kamera Patroli Aktif |
| **`/snapshot/live`** | GET | Snapshot Frame Tunggal JPEG Kamera Aktif |
| **`/api/status`** | GET | JSON Status Kamera Aktif, Sisa Waktu, FPS, & Riwayat Deteksi |
| **`/api/patrol/auto`** | POST | Mengaktifkan Mode Patroli Otomatis |
| **`/api/patrol/lock?id=<1-4>`** | POST | Mengunci Pengawasan ke Kamera Lab Tertentu |
| **`/api/patrol/interval?sec=<detik>`** | POST | Mengubah Durasi Patroli per Lab |
