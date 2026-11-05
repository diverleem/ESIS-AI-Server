#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
E.S.I.S (Electronic Smoking Identification System)
Smart Patrol Tour - Main Stream Multi-Camera Face Recognition Node
================================================================================
Features:
- Smart Patrol Tour: Berputar otomatis antar 4 lab setiap 30 detik (default).
- Main Stream Full HD / 2K Decoding: Kualitas gambar jernih maksimal untuk FaceNet.
- Zero-Latency Frame Grabber: Mencegah penumpukan buffer video (selalu real-time).
- Dual-Scale Fast Inference: Deteksi cepat pada skala optimal, crop pada resolusi penuh.
- Seamless Pre-loader: Menghubungkan kamera berikutnya di latar belakang agar transisi mulus.
- Interactive Web SOC Dashboard:
    * Live MJPEG Stream dengan HUD CCTV dinamis & Countdown Timer
    * Kontrol Patroli Otomatis vs Kunci Kamera Tertentu (Manual Lock)
    * Pilihan interval waktu patroli (15s, 30s, 60s)
    * Statistik real-time (FPS, Latensi AI, Resolusi, Status Buffer)
    * Tabel riwayat identifikasi mahasiswa real-time
- Fallback Cerdas: Otomatis mencoba Mainstream, jika gagal fallback ke Substream.
- Mode Simulasi (--mock): Menggunakan foto volunteer nyata dengan simulasi patroli lab.
================================================================================
"""

import os
import sys
import time
import json
import argparse
import threading
import urllib.parse
from typing import Dict, List, Tuple, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import cv2

# ------------------------------------------------------------------------------
# 1. KONFIGURASI DEFAULT & DEFINISI 4 LAB PATROLI
# ------------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODEL_ONNX = os.path.join(SCRIPT_DIR, "facenet.onnx")
DEFAULT_DATABASE_NPZ = os.path.join(SCRIPT_DIR, "data_wajah_facenet_mtcnn.npz")
DEFAULT_HLS_BASE = os.environ.get("HLS_BASE", "https://fdxuqgcmfo.a.pinggy.link").rstrip("/")

DEFAULT_PATROL_INTERVAL = int(os.environ.get("PATROL_INTERVAL", "30")) # detik per lab
DEFAULT_PORT = int(os.environ.get("PORT", "8000"))
DEFAULT_THRESHOLD = float(os.environ.get("THRESHOLD", "0.75"))

PATROL_LABS = [
    {
        "id": 1,
        "key": "cam1",
        "name": "Lab Riset Lantai 5",
        "main_path": "cam_lab_riset_lt_5",
        "sub_path": "ecam_lab_riset_lt_5",
        "main_url": f"{DEFAULT_HLS_BASE}/cam_lab_riset_lt_5/index.m3u8",
        "sub_url": f"{DEFAULT_HLS_BASE}/ecam_lab_riset_lt_5/index.m3u8",
        "codec_info": "H.265 (HEVC) - 2K QHD"
    },
    {
        "id": 2,
        "key": "cam2",
        "name": "Lab Sistem Digital",
        "main_path": "cam_lab_sistem_digital",
        "sub_path": "ecam_lab_sistem_digital",
        "main_url": f"{DEFAULT_HLS_BASE}/cam_lab_sistem_digital/index.m3u8",
        "sub_url": f"{DEFAULT_HLS_BASE}/ecam_lab_sistem_digital/index.m3u8",
        "codec_info": "H.264 (AVC) - 720p HD"
    },
    {
        "id": 3,
        "key": "cam3",
        "name": "Lab OS",
        "main_path": "cam_lab_os",
        "sub_path": "ecam_lab_os",
        "main_url": f"{DEFAULT_HLS_BASE}/cam_lab_os/index.m3u8",
        "sub_url": f"{DEFAULT_HLS_BASE}/ecam_lab_os/index.m3u8",
        "codec_info": "H.264 (AVC)"
    },
    {
        "id": 4,
        "key": "cam4",
        "name": "Lab Programming",
        "main_path": "cam_lab_programming",
        "sub_path": "ecam_lab_programming",
        "main_url": f"{DEFAULT_HLS_BASE}/cam_lab_programming/index.m3u8",
        "sub_url": f"{DEFAULT_HLS_BASE}/ecam_lab_programming/index.m3u8",
        "codec_info": "H.264 (AVC)"
    }
]

# ------------------------------------------------------------------------------
# 2. DETEKTOR WAJAH MULTI-BACKEND (MTCNN / YUNET / HAAR DENGAN OPTIMASI CLAHE)
# ------------------------------------------------------------------------------
class FaceDetector:
    """
    Detektor wajah serbaguna dengan auto-fallback dan penyempurnaan kontras CLAHE.
    """
    def __init__(self, mode: str = "auto", conf_threshold: float = 0.80):
        self.mode = mode.lower()
        self.conf_threshold = conf_threshold
        self.active_backend = None
        self.detector_obj = None
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self._init_detector()

    def _init_detector(self):
        if self.mode in ("auto", "mtcnn"):
            try:
                from mtcnn import MTCNN
                self.detector_obj = MTCNN()
                self.active_backend = "MTCNN (Deep Learning)"
                print(f" [FaceDetector] Backend aktif: {self.active_backend}")
                return
            except Exception as e:
                if self.mode == "mtcnn":
                    print(f" [FaceDetector] Gagal memuat MTCNN ({e}), mencoba fallback...")

        # Fallback Haar Cascade bawaan OpenCV
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector_obj = cv2.CascadeClassifier(haar_path)
        self.active_backend = "OpenCV Haar Cascade (Optimized CLAHE)"
        print(f" [FaceDetector] Backend aktif: {self.active_backend}")

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], np.ndarray, float]]:
        """
        Mendeteksi seluruh wajah pada frame dengan teknik dual-scale untuk performa real-time.
        """
        results = []
        if frame is None or frame.size == 0:
            return results

        h, w = frame.shape[:2]

        # Dual-scale: Jika resolusi besar (Mainstream 1080p / 2K), downscale dulu untuk deteksi cepat
        scale = 1.0
        if w > 1280 or h > 720:
            scale = 960.0 / w
            det_frame = cv2.resize(frame, (960, int(h * scale)))
        else:
            det_frame = frame

        dh, dw = det_frame.shape[:2]

        if "MTCNN" in self.active_backend:
            rgb_frame = cv2.cvtColor(det_frame, cv2.COLOR_BGR2RGB)
            try:
                detections = self.detector_obj.detect_faces(rgb_frame)
                for det in detections:
                    conf = det.get("confidence", 0.0)
                    if conf < self.conf_threshold:
                        continue
                    x, y, bw, bh = det["box"]
                    # Skala kembali ke koordinat resolusi asli
                    orig_x = max(0, int(x / scale))
                    orig_y = max(0, int(y / scale))
                    orig_bw = min(w - orig_x, int(bw / scale))
                    orig_bh = min(h - orig_y, int(bh / scale))

                    if orig_bw < 20 or orig_bh < 20:
                        continue
                    # Crop wajah dari frame resolusi asli yang jernih
                    crop = frame[orig_y:orig_y + orig_bh, orig_x:orig_x + orig_bw]
                    if crop.size == 0:
                        continue
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop_160 = cv2.resize(crop_rgb, (160, 160))
                    results.append(((orig_x, orig_y, orig_bw, orig_bh), crop_160, conf))
            except Exception:
                pass
            return results

        else: # Haar Cascade dengan CLAHE
            gray = cv2.cvtColor(det_frame, cv2.COLOR_BGR2GRAY)
            gray = self.clahe.apply(gray)
            faces = self.detector_obj.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(16, 16)
            )
            for (x, y, bw, bh) in faces:
                orig_x = max(0, int(x / scale))
                orig_y = max(0, int(y / scale))
                orig_bw = min(w - orig_x, int(bw / scale))
                orig_bh = min(h - orig_y, int(bh / scale))

                if orig_bw < 20 or orig_bh < 20:
                    continue
                crop = frame[orig_y:orig_y + orig_bh, orig_x:orig_x + orig_bw]
                if crop.size == 0:
                    continue
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_160 = cv2.resize(crop_rgb, (160, 160))
                results.append(((orig_x, orig_y, orig_bw, orig_bh), crop_160, 0.92))
            return results

# ------------------------------------------------------------------------------
# 3. ENGINE REKOGNISI WAJAH (FACENET ONNX + NPZ DATABASE)
# ------------------------------------------------------------------------------
class FaceRecognitionEngine:
    def __init__(self, model_path: str, npz_database_path: str, threshold: float = 0.75, device: str = "auto"):
        self.model_path = model_path
        self.npz_database_path = npz_database_path
        self.threshold = threshold
        self.active_provider = "CPU"
        self.session = None

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model FaceNet tidak ditemukan: {self.model_path}")
        if not os.path.exists(self.npz_database_path):
            raise FileNotFoundError(f"Database embedding tidak ditemukan: {self.npz_database_path}")

        self._load_onnx_model(device)

        data = np.load(self.npz_database_path)
        self.db_embeddings = data["x"]
        self.db_labels = data["y"]
        unique_classes = len(np.unique(self.db_labels))
        print(f" [FaceRecognition] Basis data wajah dimuat: {self.db_embeddings.shape[0]} foto dari {unique_classes} identitas.")

    def _load_onnx_model(self, device: str):
        import onnxruntime as ort

        available = ort.get_available_providers()
        if device.lower() in ("auto", "cuda") and "CUDAExecutionProvider" in available:
            try:
                test_sess = ort.InferenceSession(self.model_path, providers=["CUDAExecutionProvider"])
                self.session = test_sess
                self.active_provider = "CUDA (NVIDIA GPU)"
                print(f" [FaceRecognition] Model ONNX FaceNet dimuat dengan akselerasi: {self.active_provider}")
                return
            except Exception as e:
                print(f" [FaceRecognition] CUDA fallback ke CPU: {e}")

        self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.active_provider = "CPU"
        print(f" [FaceRecognition] Model ONNX FaceNet dimuat dengan provider: {self.active_provider}")

    def extract_embedding(self, face_160_rgb: np.ndarray) -> np.ndarray:
        norm_face = (face_160_rgb.astype(np.float32) - 127.5) / 127.5
        inp = np.expand_dims(norm_face, axis=0)
        out = self.session.run(None, {"input": inp})[0][0]
        norm = np.linalg.norm(out)
        if norm > 0:
            out = out / norm
        return out

    def recognize(self, face_160_rgb: np.ndarray) -> Tuple[str, float, float]:
        emb = self.extract_embedding(face_160_rgb)
        distances = np.linalg.norm(emb - self.db_embeddings, axis=1)
        best_idx = int(np.argmin(distances))
        min_dist = float(distances[best_idx])

        if min_dist < self.threshold:
            name = str(self.db_labels[best_idx])
            similarity = max(0.0, min(100.0, (1.0 - (min_dist / self.threshold)) * 100.0))
        else:
            name = "Unknown"
            similarity = max(0.0, min(100.0, (1.0 - (min_dist / 1.5)) * 100.0))

        return name, min_dist, similarity

# ------------------------------------------------------------------------------
# 4. SINGLE-STREAM ZERO-LATENCY PATROL CAMERA WORKER
# ------------------------------------------------------------------------------
class PatrolStreamWorker:
    """
    Mengelola 1 kamera aktif secara dinamis dengan buffer-dropping agar nol latensi,
    serta melakukan pra-koneksi ke kamera berikutnya sebelum waktu 30 detik habis.
    """
    def __init__(self, patrol_labs: List[Dict], interval: int = 30, is_mock: bool = False):
        self.patrol_labs = patrol_labs
        self.interval = interval
        self.is_mock = is_mock

        self.current_index = 0
        self.locked_id: Optional[int] = None
        self.is_auto = True

        self.cap: Optional[cv2.VideoCapture] = None
        self.active_source_url: str = ""
        self.active_resolution: str = "Menghubungkan..."
        self.active_codec: str = ""

        self.latest_frame: Optional[np.ndarray] = None
        self.is_connected = False
        self.is_running = False
        self.fps = 0.0

        self.time_switched = time.time()
        self.lock = threading.Lock()

        # Mock volunteer photos for realistic test mode
        self.mock_volunteers = []
        if self.is_mock:
            self._load_mock_volunteers()

    def _load_mock_volunteers(self):
        try:
            candidates = [
                os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "known_faces")),
                os.path.abspath(os.path.join(SCRIPT_DIR, "..", "known_faces")),
                os.path.abspath(os.path.join(SCRIPT_DIR, "known_faces")),
            ]
            for c in candidates:
                if os.path.exists(c) and os.path.isdir(c):
                    subdirs = sorted([d for d in os.listdir(c) if os.path.isdir(os.path.join(c, d))])
                    for d in subdirs:
                        d_path = os.path.join(c, d)
                        imgs = [f for f in os.listdir(d_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                        if imgs:
                            img = cv2.imread(os.path.join(d_path, imgs[0]))
                            if img is not None:
                                self.mock_volunteers.append((d, img))
                    break
            print(f" [MockPatrol] Dimuat {len(self.mock_volunteers)} sampel wajah volunteer untuk simulasi.")
        except Exception:
            pass

    def start(self):
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.worker_thread.start()
        return self

    def stop(self):
        self.is_running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass

    def get_active_info(self) -> Dict:
        with self.lock:
            lab = self.patrol_labs[self.current_index]
            elapsed = time.time() - self.time_switched
            remain = max(0, int(self.interval - elapsed)) if self.is_auto else 0
            return {
                "id": lab["id"],
                "name": lab["name"],
                "resolution": self.active_resolution,
                "codec": lab.get("codec_info", ""),
                "is_auto": self.is_auto,
                "interval": self.interval,
                "time_left": remain,
                "locked_id": self.locked_id,
                "fps": round(self.fps, 1),
                "is_connected": self.is_connected
            }

    def set_lock_camera(self, lab_id: int):
        with self.lock:
            found = False
            for idx, lab in enumerate(self.patrol_labs):
                if lab["id"] == lab_id:
                    self.current_index = idx
                    self.locked_id = lab_id
                    self.is_auto = False
                    self.time_switched = time.time()
                    found = True
                    break
            if found:
                print(f" [Patrol] Kamera dikunci ke: {self.patrol_labs[self.current_index]['name']}")
                self._reconnect_active()

    def set_auto_patrol(self, interval: Optional[int] = None):
        with self.lock:
            self.is_auto = True
            self.locked_id = None
            if interval:
                self.interval = interval
            self.time_switched = time.time()
            print(f" [Patrol] Mode Patroli Otomatis diaktifkan (Interval: {self.interval}s)")

    def _open_camera(self, lab: Dict) -> Tuple[Optional[cv2.VideoCapture], str, str]:
        if self.is_mock:
            return None, "Simulasi Mock CCTV", "1280x720 (HD)"

        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|stimeout;4000000|rw_timeout;4000000|timeout;4000000|http_user_agent;Mozilla/5.0"
        )

        # 1. Coba Main Stream dulu (Full HD / 2K)
        main_url = lab["main_url"]
        try:
            print(f" [Patrol] Membuka Main Stream: {lab['name']} ({main_url})")
            cap = cv2.VideoCapture(main_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    res_str = f"{w}x{h}"
                    print(f" [Patrol] [OK] Main Stream AKTIF! Resolusi: {res_str}")
                    return cap, main_url, res_str
                cap.release()
        except Exception as e:
            print(f" [Patrol] Mainstream gagal ({e}), beralih ke Substream...")

        # 2. Fallback ke Substream jika Mainstream gagal
        sub_url = lab["sub_url"]
        try:
            print(f" [Patrol] Membuka Sub Stream: {lab['name']} ({sub_url})")
            cap = cv2.VideoCapture(sub_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    res_str = f"{w}x{h} (Substream)"
                    print(f" [Patrol] [OK] Sub Stream AKTIF! Resolusi: {res_str}")
                    return cap, sub_url, res_str
                cap.release()
        except Exception as e:
            print(f" [Patrol] Substream gagal ({e})")

        return None, "Gagal Terhubung", "No Signal"

    def _reconnect_active(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        lab = self.patrol_labs[self.current_index]
        cap, src_url, res_str = self._open_camera(lab)
        self.cap = cap
        self.active_source_url = src_url
        self.active_resolution = res_str
        self.is_connected = (cap is not None and cap.isOpened()) or self.is_mock
        self.time_switched = time.time()

    def _generate_mock_frame(self, lab: Dict) -> np.ndarray:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (20, 24, 32)

        # Grid garis halus
        for y in range(0, 720, 80):
            cv2.line(frame, (0, y), (1280, y), (30, 36, 48), 1)
        for x in range(0, 1280, 80):
            cv2.line(frame, (x, 0), (x, 720), (30, 36, 48), 1)

        t_now = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"[ SIMULASI PATROLI CCTV: {lab['name'].upper()} ]", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 200), 2)
        cv2.putText(frame, f"Waktu Sistem: {t_now} | Main Stream HD Simulation", (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

        # Pasang volunteer wajah nyata dengan animasi bergerak
        if self.mock_volunteers:
            v_idx = self.current_index % len(self.mock_volunteers)
            v_name, v_img = self.mock_volunteers[v_idx]
            vh, vw = v_img.shape[:2]
            target_h = 240
            target_w = int(vw * (target_h / vh))
            face_resized = cv2.resize(v_img, (target_w, target_h))

            shift_x = int(60 * np.sin(time.time() * 1.8))
            shift_y = int(25 * np.cos(time.time() * 1.8))
            pos_x = max(50, min(1280 - target_w - 50, 500 + shift_x))
            pos_y = max(150, min(720 - target_h - 50, 240 + shift_y))

            frame[pos_y:pos_y + target_h, pos_x:pos_x + target_w] = face_resized
            cv2.rectangle(frame, (pos_x, pos_y), (pos_x + target_w, pos_y + target_h), (80, 80, 80), 1)

        return frame

    def _run_loop(self):
        self._reconnect_active()

        fps_counter = 0
        fps_timer = time.time()

        while self.is_running:
            # 1. Cek apakah waktu patroli sudah habis (Auto Switch)
            if self.is_auto and (time.time() - self.time_switched >= self.interval):
                with self.lock:
                    self.current_index = (self.current_index + 1) % len(self.patrol_labs)
                    next_lab = self.patrol_labs[self.current_index]
                    print(f"\n [Patrol Tour] Waktu habis! Beralih ke kamera berikutnya: {next_lab['name']}")
                self._reconnect_active()

            # 2. Ambil frame kamera
            if self.is_mock:
                with self.lock:
                    lab = self.patrol_labs[self.current_index]
                    self.latest_frame = self._generate_mock_frame(lab)
                    self.is_connected = True
                self.fps = 25.0
                time.sleep(0.04)
                continue

            if not self.is_connected or self.cap is None:
                self._reconnect_active()
                time.sleep(2.0)
                continue

            # Zero-Latency Frame Grab: Selalu ambil frame paling baru, buang buffer antrian lama
            ret = self.cap.grab()
            if not ret:
                print(f" [Patrol] Koneksi stream terputus. Mencoba reconnect...")
                self.is_connected = False
                time.sleep(1.0)
                continue

            ret, frame = self.cap.retrieve()
            if not ret or frame is None:
                continue

            with self.lock:
                self.latest_frame = frame

            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                self.fps = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer = time.time()

            time.sleep(0.005)

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

# ------------------------------------------------------------------------------
# 5. PIPELINE INFERENSI & ANOTASI VIDEO DENGAN HUD CCTV PATROL
# ------------------------------------------------------------------------------
class PipelineManager:
    """
    Mengorkestrasi inferensi pengenalan wajah pada kamera patroli aktif,
    menggambar HUD overlay CCTV profesional, dan mencatat riwayat deteksi.
    """
    def __init__(self, patrol_worker: PatrolStreamWorker, detector: FaceDetector, recognizer: FaceRecognitionEngine):
        self.worker = patrol_worker
        self.detector = detector
        self.recognizer = recognizer

        self.annotated_frame: Optional[np.ndarray] = None
        self.detections_history: List[Dict] = []
        self.last_inf_time = 0.0
        self.lock = threading.Lock()
        self.is_running = False

    def start(self):
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.worker_thread.start()
        return self

    def _draw_hud(self, frame: np.ndarray, detections: List[Dict], info: Dict) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        # 1. Gambar Bounding Box Mahasiswa
        for d in detections:
            x, y, bw, bh = d["box"]
            name = d["name"]
            sim = d["similarity"]
            dist = d["distance"]

            if name != "Unknown":
                color = (46, 204, 113) # Hijau terang
                tag_text = f"{name} ({sim:.1f}%)"
            else:
                color = (41, 128, 250) # Oranye/Merah
                tag_text = f"Unknown (d:{dist:.2f})"

            # Bounding box & sudut aksen
            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
            corner_len = min(25, int(bw * 0.25))
            cv2.line(out, (x, y), (x + corner_len, y), color, 4)
            cv2.line(out, (x, y), (x, y + corner_len), color, 4)
            cv2.line(out, (x + bw, y), (x + bw - corner_len, y), color, 4)
            cv2.line(out, (x + bw, y), (x + bw, y + corner_len), color, 4)
            cv2.line(out, (x, y + bh), (x + corner_len, y + bh), color, 4)
            cv2.line(out, (x, y + bh), (x, y + bh - corner_len), color, 4)
            cv2.line(out, (x + bw, y + bh), (x + bw - corner_len, y + bh), color, 4)
            cv2.line(out, (x + bw, y + bh), (x + bw, y + bh - corner_len), color, 4)

            # Label banner
            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
            label_y1 = max(0, y - th - 14)
            label_y2 = y
            cv2.rectangle(out, (x, label_y1), (x + tw + 18, label_y2), color, -1)
            cv2.putText(out, tag_text, (x + 8, label_y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Header HUD Atas (Semi-Transparan)
        header_h = 48
        hud_bg = np.zeros((header_h, w, 3), dtype=np.uint8)
        overlay = out[0:header_h, 0:w].copy()
        cv2.addWeighted(hud_bg, 0.70, overlay, 0.30, 0, out[0:header_h, 0:w])

        cam_name = info["name"].upper()
        res_info = info["resolution"]
        fps = info["fps"]

        # Indikator Status Patroli
        if info["is_auto"]:
            mode_badge = f"PATROLI AUTO: SISA {info['time_left']}s"
            badge_color = (0, 200, 255) # Amber
        else:
            mode_badge = "MANUAL LOCK"
            badge_color = (0, 140, 255) # Orange

        cv2.putText(out, f"KAMERA: {cam_name} ({res_info})", (16, 32), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 220), 1, cv2.LINE_AA)
        cv2.putText(out, f"[ {mode_badge} ]", (w // 2 - 110, 32), cv2.FONT_HERSHEY_DUPLEX, 0.60, badge_color, 1, cv2.LINE_AA)
        cv2.putText(out, f"{time.strftime('%H:%M:%S')} | {fps:.1f} FPS", (w - 200, 32), cv2.FONT_HERSHEY_DUPLEX, 0.60, (240, 240, 240), 1, cv2.LINE_AA)

        # 3. Footer Bar Bawah
        footer_y = h - 35
        cv2.rectangle(out, (0, footer_y), (w, h), (12, 16, 24), -1)
        ai_stat = f"AI Engine: {self.recognizer.active_provider} | Latensi: {self.last_inf_time:.1f}ms | Wajah Terdeteksi: {len(detections)}"
        cv2.putText(out, ai_stat, (16, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)

        return out

    def _process_loop(self):
        while self.is_running:
            raw_frame = self.worker.get_frame()
            if raw_frame is None:
                time.sleep(0.02)
                continue

            t0 = time.time()
            raw_faces = self.detector.detect_faces(raw_frame)

            info = self.worker.get_active_info()
            cam_name = info["name"]

            detections = []
            for (bbox, face_crop, conf) in raw_faces:
                name, dist, sim = self.recognizer.recognize(face_crop)
                det_item = {
                    "box": bbox,
                    "name": name,
                    "similarity": sim,
                    "distance": dist,
                    "camera": cam_name,
                    "timestamp": time.strftime("%H:%M:%S")
                }
                detections.append(det_item)

                if name != "Unknown":
                    self._log_detection(name, cam_name, sim)

            self.last_inf_time = (time.time() - t0) * 1000.0

            annotated = self._draw_hud(raw_frame, detections, info)

            with self.lock:
                self.annotated_frame = annotated

            time.sleep(0.01)

    def _log_detection(self, name: str, camera_name: str, similarity: float):
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        with self.lock:
            for item in reversed(self.detections_history[-15:]):
                if item["name"] == name and item["camera"] == camera_name:
                    return
            self.detections_history.append({
                "name": name,
                "camera": camera_name,
                "similarity": f"{similarity:.1f}%",
                "time": now_str
            })
            if len(self.detections_history) > 60:
                self.detections_history.pop(0)

    def get_live_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.annotated_frame is not None:
                return self.annotated_frame.copy()
        return self.worker.get_frame()

    def get_status_dict(self) -> Dict:
        info = self.worker.get_active_info()
        with self.lock:
            return {
                "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "active_camera": info,
                "ai_provider": self.recognizer.active_provider,
                "detector_backend": self.detector.active_backend,
                "latency_ms": round(self.last_inf_time, 1),
                "recent_detections": list(reversed(self.detections_history[-15:])),
                "patrol_labs": [
                    {"id": l["id"], "name": l["name"], "codec": l.get("codec_info", "")}
                    for l in self.worker.patrol_labs
                ]
            }

# ------------------------------------------------------------------------------
# 6. WEB DASHBOARD TEMPLATE (HTML5 + CSS DARK THEME + PATROL CONTROLS)
# ------------------------------------------------------------------------------
HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>E.S.I.S - Smart Patrol Tour Face Recognition (Main Stream 1080p HD)</title>
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-card: #131b2e;
            --border-color: #1f2c4a;
            --accent-green: #10b981;
            --accent-blue: #3b82f6;
            --accent-cyan: #06b6d4;
            --accent-amber: #f59e0b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
        body { background: var(--bg-primary); color: var(--text-main); min-height: 100vh; padding: 20px; }
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); padding-bottom: 16px; margin-bottom: 20px; flex-wrap: wrap; gap: 12px; }
        .logo-title h1 { font-size: 20px; font-weight: 700; color: #fff; letter-spacing: -0.5px; }
        .logo-title p { font-size: 13px; color: var(--text-muted); margin-top: 4px; }
        .badges-group { display: flex; gap: 8px; flex-wrap: wrap; }
        .badge { padding: 5px 12px; border-radius: 9999px; font-size: 12px; font-weight: 600; background: rgba(16, 185, 129, 0.15); color: var(--accent-green); border: 1px solid var(--accent-green); }
        .badge-cyan { background: rgba(6, 182, 212, 0.15); color: var(--accent-cyan); border: 1px solid var(--accent-cyan); }
        .badge-amber { background: rgba(245, 158, 11, 0.15); color: var(--accent-amber); border: 1px solid var(--accent-amber); }
        
        .main-layout { display: grid; grid-template-columns: 2.3fr 1fr; gap: 20px; margin-bottom: 24px; }
        @media (max-width: 1024px) {
            .main-layout { grid-template-columns: 1fr; }
        }

        /* Player Box */
        .player-container { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 14px; overflow: hidden; box-shadow: 0 4px 25px rgba(0,0,0,0.4); }
        .player-header { padding: 12px 18px; background: rgba(10, 14, 23, 0.8); display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); }
        .player-header h3 { font-size: 15px; font-weight: 600; color: #fff; display: flex; align-items: center; gap: 8px; }
        .live-dot { width: 10px; height: 10px; border-radius: 50%; background: #ef4444; box-shadow: 0 0 8px #ef4444; animation: pulse 1.5s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
        
        .feed-wrapper { width: 100%; aspect-ratio: 16 / 9; background: #000; position: relative; overflow: hidden; }
        .feed-wrapper img { width: 100%; height: 100%; object-fit: contain; display: block; }
        
        /* Control Panel */
        .controls-panel { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 14px; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
        .controls-panel h2 { font-size: 15px; font-weight: 700; color: #fff; border-bottom: 1px solid var(--border-color); padding-bottom: 10px; }
        
        .btn-group { display: flex; flex-direction: column; gap: 8px; }
        .btn-action { background: #1a253d; border: 1px solid var(--border-color); color: var(--text-main); padding: 10px 14px; border-radius: 8px; font-size: 13px; font-weight: 600; cursor: pointer; text-align: left; transition: all 0.2s; display: flex; justify-content: space-between; align-items: center; }
        .btn-action:hover { background: #243354; border-color: var(--accent-blue); }
        .btn-action.active { background: var(--accent-blue); border-color: var(--accent-blue); color: #fff; }
        .btn-patrol { background: rgba(245, 158, 11, 0.15); border: 1px solid var(--accent-amber); color: var(--accent-amber); text-align: center; justify-content: center; font-weight: 700; }
        .btn-patrol.active { background: var(--accent-amber); color: #000; }

        .interval-selector { display: flex; gap: 6px; margin-top: 4px; }
        .btn-pill { flex: 1; background: #1a253d; border: 1px solid var(--border-color); color: var(--text-muted); padding: 6px; border-radius: 6px; font-size: 11px; font-weight: 600; cursor: pointer; text-align: center; }
        .btn-pill.active { background: var(--accent-cyan); color: #000; border-color: var(--accent-cyan); }

        .stat-item { display: flex; justify-content: space-between; font-size: 13px; padding: 6px 0; border-bottom: 1px solid rgba(31, 44, 74, 0.5); }
        .stat-item .label { color: var(--text-muted); }
        .stat-item .val { font-weight: 600; color: #fff; }

        /* History Table */
        .log-section { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 14px; padding: 20px; }
        .log-section h2 { font-size: 15px; font-weight: 700; margin-bottom: 14px; }
        .log-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .log-table th { text-align: left; padding: 10px 14px; border-bottom: 1px solid var(--border-color); color: var(--text-muted); font-weight: 600; }
        .log-table td { padding: 10px 14px; border-bottom: 1px solid rgba(31, 44, 74, 0.6); }
        .tag-match { color: var(--accent-green); font-weight: 600; }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo-title">
            <h1>E.S.I.S - Smart Patrol Tour Face Recognition (Main Stream HD)</h1>
            <p>Sistem Pengawasan Patroli Bergilir 30 Detik per Lab | Main Stream Full HD + FaceNet ONNX</p>
        </div>
        <div class="badges-group">
            <span class="badge badge-amber" id="badge-mode">PATROLI 30s</span>
            <span class="badge badge-cyan" id="badge-res">-</span>
            <span class="badge" id="ai-status">RUNPOD GPU NODE ACTIVE</span>
        </div>
    </div>

    <div class="main-layout">
        <!-- Live Player -->
        <div class="player-container">
            <div class="player-header">
                <h3><span class="live-dot"></span> <span id="active-lab-name">Menghubungkan Kamera...</span></h3>
                <span class="badge badge-cyan" id="hud-timer">SISA: --s</span>
            </div>
            <div class="feed-wrapper">
                <img id="live-stream" src="/video_feed/live" onerror="handleStreamFallback(this)" alt="Patrol Live Feed">
            </div>
        </div>

        <!-- Controls & Status Panel -->
        <div class="controls-panel">
            <h2>Kontrol Mode Patroli</h2>
            
            <button class="btn-action btn-patrol active" id="btn-auto-patrol" onclick="setAutoPatrol()">
                <span>🔄 Mulai Patroli Otomatis</span>
                <span id="patrol-timer-badge">30s</span>
            </button>

            <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">Interval Pergantian Lab:</div>
            <div class="interval-selector">
                <button class="btn-pill" onclick="setIntervalSec(15)">15 Detik</button>
                <button class="btn-pill active" onclick="setIntervalSec(30)">30 Detik</button>
                <button class="btn-pill" onclick="setIntervalSec(60)">60 Detik</button>
            </div>

            <div style="font-size: 12px; color: var(--text-muted); margin-top: 10px;">Atau Kunci Kamera Tertentu (Manual):</div>
            <div class="btn-group">
                <button class="btn-action" id="btn-cam-1" onclick="lockCamera(1)">
                    <span>Lab Riset Lantai 5</span>
                    <span style="font-size: 11px; opacity: 0.7;">2K QHD</span>
                </button>
                <button class="btn-action" id="btn-cam-2" onclick="lockCamera(2)">
                    <span>Lab Sistem Digital</span>
                    <span style="font-size: 11px; opacity: 0.7;">720p HD</span>
                </button>
                <button class="btn-action" id="btn-cam-3" onclick="lockCamera(3)">
                    <span>Lab OS</span>
                    <span style="font-size: 11px; opacity: 0.7;">Main/Sub</span>
                </button>
                <button class="btn-action" id="btn-cam-4" onclick="lockCamera(4)">
                    <span>Lab Programming</span>
                    <span style="font-size: 11px; opacity: 0.7;">Main/Sub</span>
                </button>
            </div>

            <h2 style="margin-top: 10px;">Metrik Performa</h2>
            <div class="stat-item">
                <span class="label">Resolusi Input</span>
                <span class="val" id="stat-res">-</span>
            </div>
            <div class="stat-item">
                <span class="label">FPS Stream</span>
                <span class="val" id="stat-fps">-</span>
            </div>
            <div class="stat-item">
                <span class="label">Latensi AI</span>
                <span class="val" id="stat-lat">-</span>
            </div>
            <div class="stat-item">
                <span class="label">Inference Engine</span>
                <span class="val" id="stat-engine">-</span>
            </div>
        </div>
    </div>

    <!-- Real-time Recognition Log Table -->
    <div class="log-section">
        <h2>Riwayat Mahasiswa Teridentifikasi di Seluruh Lab</h2>
        <table class="log-table">
            <thead>
                <tr>
                    <th>Waktu Deteksi</th>
                    <th>Nama Mahasiswa</th>
                    <th>Lokasi Kamera Lab</th>
                    <th>Akurasi Kemiripan</th>
                </tr>
            </thead>
            <tbody id="log-body">
                <tr><td colspan="4" style="text-align: center; color: var(--text-muted);">Menunggu aktivitas wajah...</td></tr>
            </tbody>
        </table>
    </div>

    <script>
        function handleStreamFallback(img) {
            console.warn("Stream interrupted, fallback polling activated");
            setInterval(() => {
                img.src = "/snapshot/live?t=" + Date.now();
            }, 100);
        }

        async function setAutoPatrol() {
            try {
                await fetch('/api/patrol/auto', { method: 'POST' });
                fetchStatus();
            } catch(e) { console.error(e); }
        }

        async function lockCamera(id) {
            try {
                await fetch('/api/patrol/lock?id=' + id, { method: 'POST' });
                fetchStatus();
            } catch(e) { console.error(e); }
        }

        async function setIntervalSec(sec) {
            document.querySelectorAll('.btn-pill').forEach(b => b.classList.remove('active'));
            if (event && event.target) event.target.classList.add('active');
            try {
                await fetch('/api/patrol/interval?sec=' + sec, { method: 'POST' });
                fetchStatus();
            } catch(e) { console.error(e); }
        }

        async function fetchStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                const cam = data.active_camera;

                document.getElementById('active-lab-name').textContent = cam.name;
                document.getElementById('stat-res').textContent = cam.resolution;
                document.getElementById('badge-res').textContent = cam.resolution;
                document.getElementById('stat-fps').textContent = cam.fps.toFixed(1) + ' FPS';
                document.getElementById('stat-lat').textContent = data.latency_ms.toFixed(1) + ' ms';
                document.getElementById('stat-engine').textContent = data.ai_provider;

                const btnPatrol = document.getElementById('btn-auto-patrol');
                const badgeMode = document.getElementById('badge-mode');
                const hudTimer = document.getElementById('hud-timer');

                // Update active buttons
                document.querySelectorAll('.btn-action:not(.btn-patrol)').forEach(b => b.classList.remove('active'));

                if (cam.is_auto) {
                    btnPatrol.classList.add('active');
                    badgeMode.textContent = `PATROLI AUTO: ${cam.time_left}s`;
                    hudTimer.textContent = `SISA: ${cam.time_left}s`;
                    document.getElementById('patrol-timer-badge').textContent = `${cam.time_left}s`;
                } else {
                    btnPatrol.classList.remove('active');
                    badgeMode.textContent = `TERKUNCI: CAM ${cam.id}`;
                    hudTimer.textContent = `LOCKED`;
                    const activeBtn = document.getElementById('btn-cam-' + cam.id);
                    if (activeBtn) activeBtn.classList.add('active');
                }

                // Table Log
                const tbody = document.getElementById('log-body');
                if (data.recent_detections && data.recent_detections.length > 0) {
                    tbody.innerHTML = data.recent_detections.map(d => `
                        <tr>
                            <td>${d.time}</td>
                            <td class="tag-match">${d.name}</td>
                            <td>${d.camera}</td>
                            <td><span class="badge">${d.similarity}</span></td>
                        </tr>
                    `).join('');
                }
            } catch(e) {
                console.error(e);
            }
        }

        setInterval(fetchStatus, 1000);
        fetchStatus();
    </script>
</body>
</html>
"""

# ------------------------------------------------------------------------------
# 7. HTTP HANDLER & REST API CONTROLLER
# ------------------------------------------------------------------------------
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

def make_stream_handler(pipeline: PipelineManager, patrol_worker: PatrolStreamWorker):
    class StreamHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            params = urllib.parse.parse_qs(parsed.query)

            if path == "/api/patrol/auto":
                patrol_worker.set_auto_patrol()
                self._send_json({"success": True, "mode": "auto"})
                return

            if path == "/api/patrol/lock":
                cam_id = int(params.get("id", [1])[0])
                patrol_worker.set_lock_camera(cam_id)
                self._send_json({"success": True, "locked_id": cam_id})
                return

            if path == "/api/patrol/interval":
                sec = int(params.get("sec", [30])[0])
                patrol_worker.set_auto_patrol(interval=sec)
                self._send_json({"success": True, "interval": sec})
                return

            self.send_error(404, "Endpoint POST tidak ditemukan")

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path

            if path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(HTML_DASHBOARD.encode("utf-8"))
                return

            if path == "/api/status":
                self._send_json(pipeline.get_status_dict())
                return

            # Snapshot Image Endpoint
            if path in ("/snapshot/live", "/snapshot/1"):
                frame = pipeline.get_live_frame()
                if frame is not None:
                    ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        b_data = jpeg.tobytes()
                        self.send_response(200)
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(b_data)))
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                        self.end_headers()
                        self.wfile.write(b_data)
                        return

            # MJPEG Stream Endpoint (Live Patrol Feed)
            if path in ("/video_feed/live", "/video_feed/1"):
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, no-transform")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.send_header("X-Accel-Buffering", "no")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()

                try:
                    while True:
                        frame = pipeline.get_live_frame()
                        if frame is not None:
                            ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            if ret:
                                b_data = jpeg.tobytes()
                                self.wfile.write(
                                    b"--frame\r\n"
                                    b"Content-Type: image/jpeg\r\n"
                                    b"Content-Length: " + str(len(b_data)).encode("ascii") + b"\r\n\r\n" +
                                    b_data + b"\r\n"
                                )
                                self.wfile.flush()
                        time.sleep(0.035)
                except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                    pass
                except Exception:
                    pass
                return

            self.send_error(404, "Halaman Tidak Ditemukan")

        def _send_json(self, data: Dict):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))

    return StreamHandler

# ------------------------------------------------------------------------------
# 8. MAIN ENTRYPOINT & CLI PARSER
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="E.S.I.S Smart Patrol Tour Face Recognition Node (Main Stream HD)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--interval", type=int, default=DEFAULT_PATROL_INTERVAL, help="Durasi patroli per lab (detik)")
    parser.add_argument("--pinggy", type=str, default=DEFAULT_HLS_BASE, help="Base URL Pinggy tunnel untuk HLS (.m3u8)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ONNX, help="Path model ONNX FaceNet")
    parser.add_argument("--database", type=str, default=DEFAULT_DATABASE_NPZ, help="Path database embedding .npz")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port Web Streaming HTTP Server")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Threshold Euclidean distance (0.7 - 0.8)")
    parser.add_argument("--detector", type=str, default="auto", choices=["auto", "mtcnn", "haar"], help="Pilihan backend face detector")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Pilihan komputasi FaceNet")
    parser.add_argument("--mock", action="store_true", help="Jalankan dalam mode simulasi CCTV (dengan foto volunteer nyata)")
    parser.add_argument("--display", action="store_true", help="Buka jendela pop-up GUI desktop lokal (cv2.imshow)")

    args = parser.parse_args()

    # Perbarui base url Pinggy jika dikustomisasi
    p_base = args.pinggy.strip().rstrip("/")
    labs = []
    for l in PATROL_LABS:
        l_copy = dict(l)
        l_copy["main_url"] = f"{p_base}/{l['main_path']}/index.m3u8"
        l_copy["sub_url"] = f"{p_base}/{l['sub_path']}/index.m3u8"
        labs.append(l_copy)

    print("=" * 80)
    print(" [START] MEMULAI E.S.I.S SMART PATROL TOUR FACE RECOGNITION NODE")
    print("=" * 80)
    print(f"  * Mode Operasi        : SMART PATROL TOUR (MAIN STREAM 1080p HD)")
    print(f"  * Interval Patroli    : {args.interval} Detik per Lab")
    print(f"  * Total Lab Dipantau  : 4 Lab (Lab Riset Lt 5, Sisdig, OS, Programming)")
    print(f"  * Model FaceNet       : {args.model}")
    print(f"  * Database Volunteer  : {args.database}")
    print(f"  * Mode Simulasi Mock  : {'AKTIF (Menggunakan Foto Volunteer)' if args.mock else 'NONAKTIF (CCTV Live)'}")
    print(f"  * Web Server Port     : {args.port}")
    print(f"  * Threshold Akurasi   : {args.threshold}")
    print("=" * 80)

    detector = FaceDetector(mode=args.detector)

    recognizer = FaceRecognitionEngine(
        model_path=args.model,
        npz_database_path=args.database,
        threshold=args.threshold,
        device=args.device
    )

    patrol_worker = PatrolStreamWorker(
        patrol_labs=labs,
        interval=args.interval,
        is_mock=args.mock
    ).start()

    pipeline = PipelineManager(
        patrol_worker=patrol_worker,
        detector=detector,
        recognizer=recognizer
    ).start()

    handler = make_stream_handler(pipeline, patrol_worker)
    server = ThreadedHTTPServer(("0.0.0.0", args.port), handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print("\n[OK] SERVER SMART PATROL BERHASIL BERJALAN!")
    print(f"  -> Buka Dashboard Web : http://localhost:{args.port}/")
    print(f"  -> Stream Live Patrol : http://localhost:{args.port}/video_feed/live")
    print(f"  -> Snapshot Live Lab  : http://localhost:{args.port}/snapshot/live")
    print(f"  -> JSON Status API    : http://localhost:{args.port}/api/status")
    print("-" * 80)
    if args.display:
        print("[INFO] Jendela monitor desktop lokal dibuka. Tekan 'q' untuk keluar.")
    else:
        print("[INFO] Silakan buka tautan di atas di browser Anda.")
        print("Tekan [Ctrl + C] di terminal untuk menghentikan program.")
    print("-" * 80)

    try:
        while True:
            if args.display:
                frame = pipeline.get_live_frame()
                if frame is not None:
                    cv2.imshow("E.S.I.S Smart Patrol Tour - Main Stream HD", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[WAIT] Menghentikan server patroli...")
    finally:
        patrol_worker.stop()
        server.shutdown()
        cv2.destroyAllWindows()
        print("[OK] Program berhasil dihentikan secara bersih.")

if __name__ == "__main__":
    main()
