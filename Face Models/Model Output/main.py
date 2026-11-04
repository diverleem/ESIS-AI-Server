#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
E.S.I.S (Electronic Surveillance & Intelligent System)
Multi-Camera Face Recognition Node for RunPod & Local Deployment
================================================================================
Features:
- Dual Camera Support: Lab Riset Lantai 5 & Lab Sistem Digital (RTSP / MediaMTX)
- High-Performance Face Recognition:
    * Detection: MTCNN (primary) / YuNet (fast ONNX) / Haar Cascade (fallback)
    * Feature Extraction: FaceNet (ONNX Runtime with GPU CUDA acceleration)
    * Matching: L2-Normalized Euclidean Distance against data_wajah_facenet_mtcnn.npz
- Built-in Web Streaming Server (HTTP MJPEG):
    * Zero-dependency dashboard accessible via browser or RunPod HTTP proxy port
    * Individual camera feeds (/video_feed/1, /video_feed/2)
    * Combined mosaic view (/video_feed/mosaic)
    * Live status JSON API (/api/status)
- Threaded non-blocking RTSP capture with auto-reconnection & mock fallback
================================================================================
"""

import os
import sys
import time
import json
import argparse
import threading
from typing import Dict, List, Tuple, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import urllib.request

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
from PIL import Image

# ------------------------------------------------------------------------------
# 1. KONFIGURASI DEFAULT & PATH MODEL
# ------------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODEL_ONNX = os.path.join(SCRIPT_DIR, "facenet.onnx")
DEFAULT_DATABASE_NPZ = os.path.join(SCRIPT_DIR, "data_wajah_facenet_mtcnn.npz")

# Default RTSP URL dari Database ESIS (Main Stream 101 (HD 1080p) untuk akurasi pengenalan wajah optimal)
DEFAULT_RTSP_CAM1 = os.environ.get(
    "RTSP_RISET_LT5",
    "rtsp://admin:EYQMDK@172.20.32.50:554/Streaming/Channels/101"
)
DEFAULT_RTSP_CAM2 = os.environ.get(
    "RTSP_SISDIG",
    "rtsp://admin:JOHVAF@172.20.32.43:554/Streaming/Channels/101"
)

DEFAULT_PORT = int(os.environ.get("PORT", "8000"))
DEFAULT_THRESHOLD = float(os.environ.get("THRESHOLD", "0.75"))

# ------------------------------------------------------------------------------
# 2. DETEKTOR WAJAH MULTI-BACKEND (MTCNN / YUNET / HAAR)
# ------------------------------------------------------------------------------
class FaceDetector:
    """
    Detektor wajah serbaguna dengan auto-fallback:
    1. MTCNN (jika paket 'mtcnn' terpasang - paling presisi sesuai data latih)
    2. OpenCV YuNet (jika bobot ONNX tersedia atau berhasil diunduh)
    3. OpenCV Haar Cascade (bawaan OpenCV, 100% selalu tersedia tanpa internet)
    """
    def __init__(self, mode: str = "auto", conf_threshold: float = 0.80):
        self.mode = mode.lower()
        self.conf_threshold = conf_threshold
        self.active_backend = None
        self.detector = None
        self._init_backend()

    def _init_backend(self):
        # 1. Coba MTCNN jika mode auto atau mtcnn
        if self.mode in ("auto", "mtcnn"):
            try:
                from mtcnn import MTCNN
                det = MTCNN()
                _ = det.detect_faces(np.zeros((100, 100, 3), dtype=np.uint8))
                self.detector = det
                self.active_backend = "MTCNN"
                print(" [FaceDetector] Backend aktif: MTCNN (Deep Learning)")
                return
            except Exception as e:
                if self.mode == "mtcnn":
                    print(f" [FaceDetector] MTCNN tidak tersedia ({e}), fallback ke alternatif.")

        # 2. Coba YuNet ONNX jika mode auto atau yunet
        if self.mode in ("auto", "yunet"):
            yunet_file = os.path.join(SCRIPT_DIR, "face_detection_yunet_2023mar.onnx")
            if os.path.exists(yunet_file) and hasattr(cv2, "FaceDetectorYN"):
                try:
                    det = cv2.FaceDetectorYN.create(
                        model=yunet_file,
                        config="",
                        input_size=(100, 100),
                        score_threshold=self.conf_threshold,
                        nms_threshold=0.3,
                        top_k=5000
                    )
                    det.setInputSize((100, 100))
                    _ = det.detect(np.zeros((100, 100, 3), dtype=np.uint8))
                    self.detector = det
                    self.active_backend = "YuNet"
                    print(" [FaceDetector] Backend aktif: OpenCV YuNet (ONNX)")
                    return
                except Exception:
                    pass

        # 3. Fallback utama: OpenCV Haar Cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        self.active_backend = "HaarCascade"
        print(" [FaceDetector] Backend aktif: OpenCV Haar Cascade (Fast Built-in)")

    def detect_faces(self, frame_bgr: np.ndarray, margin_percent: float = 0.10) -> List[Tuple[Tuple[int, int, int, int], np.ndarray, float]]:
        """
        Mendeteksi wajah pada frame BGR.
        Mengembalikan daftar tuple: [((x, y, w, h), face_160_rgb, confidence), ...]
        """
        results = []
        h, w = frame_bgr.shape[:2]

        if self.active_backend == "MTCNN":
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            raw_faces = self.detector.detect_faces(img_rgb)
            for f in raw_faces:
                conf = f.get("confidence", 0.0)
                if conf < self.conf_threshold:
                    continue
                x, y, fw, fh = f["box"]
                x, y = max(0, abs(x)), max(0, abs(y))
                # Padding margin 10%
                pad_x = int(fw * margin_percent)
                pad_y = int(fh * margin_percent)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w, x + fw + pad_x)
                y2 = min(h, y + fh + pad_y)

                crop = img_rgb[y1:y2, x1:x2]
                if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                    continue
                crop_160 = cv2.resize(crop, (160, 160))
                results.append(((x1, y1, x2 - x1, y2 - y1), crop_160, conf))

        elif self.active_backend == "YuNet":
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(frame_bgr)
            if faces is not None:
                for f in faces:
                    conf = float(f[-1])
                    if conf < self.conf_threshold:
                        continue
                    x, y, fw, fh = map(int, f[:4])
                    pad_x = int(fw * margin_percent)
                    pad_y = int(fh * margin_percent)
                    x1 = max(0, x - pad_x)
                    y1 = max(0, y - pad_y)
                    x2 = min(w, x + fw + pad_x)
                    y2 = min(h, y + fh + pad_y)

                    crop_bgr = frame_bgr[y1:y2, x1:x2]
                    if crop_bgr.size == 0 or crop_bgr.shape[0] < 10 or crop_bgr.shape[1] < 10:
                        continue
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    crop_160 = cv2.resize(crop_rgb, (160, 160))
                    results.append(((x1, y1, x2 - x1, y2 - y1), crop_160, conf))

        else: # Haar Cascade
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            for (x, y, fw, fh) in faces:
                pad_x = int(fw * margin_percent)
                pad_y = int(fh * margin_percent)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w, x + fw + pad_x)
                y2 = min(h, y + fh + pad_y)

                crop_bgr = frame_bgr[y1:y2, x1:x2]
                if crop_bgr.size == 0:
                    continue
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                crop_160 = cv2.resize(crop_rgb, (160, 160))
                results.append(((x1, y1, x2 - x1, y2 - y1), crop_160, 0.90))

        return results

# ------------------------------------------------------------------------------
# 3. ENGINE PENGENALAN WAJAH FACENET DENGAN AKSELERASI ONNX RUNTIME
# ------------------------------------------------------------------------------
class FaceRecognitionEngine:
    """
    Ekstraksi Fitur FaceNet 512D & Pencocokan Euclidean Terhadap Basis Data NPZ.
    Mendukung akselerasi CUDA GPU di RunPod dan CPU fallback otomatis.
    """
    def __init__(self, model_path: str, npz_database_path: str, threshold: float = 0.75, device: str = "auto"):
        self.threshold = threshold
        self.model_path = model_path
        self.npz_database_path = npz_database_path
        self.session = None
        self.db_embeddings: Optional[np.ndarray] = None
        self.db_labels: Optional[np.ndarray] = None
        self.active_provider = "CPU"

        self._load_database()
        self._load_onnx_model(device)

    def _load_database(self):
        if not os.path.exists(self.npz_database_path):
            raise FileNotFoundError(f"File database embedding NPZ tidak ditemukan di: {self.npz_database_path}")

        data = np.load(self.npz_database_path)
        self.db_embeddings = data["x"] # Shape: (N, 512)
        self.db_labels = data["y"]     # Shape: (N,)
        unique_classes = len(np.unique(self.db_labels))
        print(f" [FaceRecognition] Basis data wajah dimuat: {self.db_embeddings.shape[0]} foto dari {unique_classes} identitas.")

    def _load_onnx_model(self, device: str):
        import onnxruntime as ort

        available = ort.get_available_providers()
        providers = []

        if device.lower() in ("auto", "cuda") and "CUDAExecutionProvider" in available:
            try:
                # Tes inisialisasi CUDA Execution Provider
                test_sess = ort.InferenceSession(self.model_path, providers=["CUDAExecutionProvider"])
                self.session = test_sess
                self.active_provider = "CUDA (NVIDIA GPU)"
                print(f" [FaceRecognition] Model ONNX FaceNet dimuat dengan akselerasi: {self.active_provider}")
                return
            except Exception as e:
                print(f" [FaceRecognition] CUDA tidak dapat diaktifkan ({e}). Menggunakan CPUExecutionProvider.")

        # Fallback CPU
        self.session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.active_provider = "CPU"
        print(f" [FaceRecognition] Model ONNX FaceNet dimuat dengan provider: {self.active_provider}")

    def extract_embedding(self, face_160_rgb: np.ndarray) -> np.ndarray:
        """
        Menghasilkan 512D L2-normalized embedding dari potongan wajah 160x160 RGB.
        Formula standarisasi FaceNet: (pixel - 127.5) / 127.5
        """
        # 1. Normalisasi Fixed Image Standardization
        norm_face = (face_160_rgb.astype(np.float32) - 127.5) / 127.5
        inp = np.expand_dims(norm_face, axis=0) # [1, 160, 160, 3]

        # 2. Inferensi ONNX
        out = self.session.run(None, {"input": inp})[0][0] # 512 features

        # 3. L2 Normalization
        norm = np.linalg.norm(out)
        if norm > 0:
            out = out / norm
        return out

    def recognize(self, face_160_rgb: np.ndarray) -> Tuple[str, float, float]:
        """
        Mencocokkan embedding wajah dengan basis data NPZ.
        Mengembalikan: (nama_prediksi, jarak_euclidean, similarity_percent)
        """
        emb = self.extract_embedding(face_160_rgb)

        # Hitung Euclidean distance ke seluruh vektor database
        distances = np.linalg.norm(emb - self.db_embeddings, axis=1)
        best_idx = int(np.argmin(distances))
        min_dist = float(distances[best_idx])

        if min_dist < self.threshold:
            name = str(self.db_labels[best_idx])
            similarity = max(0.0, (1.0 - (min_dist / self.threshold))) * 100.0
        else:
            name = "Unknown"
            similarity = max(0.0, (1.0 - (min_dist / 2.0))) * 100.0

        return name, min_dist, similarity

# ------------------------------------------------------------------------------
# 4. WORKER THREAD KAMERA CCTV (NON-BLOCKING DENGAN AUTO-RECONNECT & MOCK)
# ------------------------------------------------------------------------------
class ThreadedCamera:
    """
    Menangani streaming kamera RTSP di thread terpisah.
    Menghindari frame buffer latency dan reconnect otomatis jika RTSP terputus.
    """
    def __init__(self, name: str, source: str, is_mock: bool = False):
        self.name = name
        self.source = source
        self.is_mock = is_mock

        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.is_running = False
        self.is_connected = False
        self.last_frame_time = 0.0
        self.fps = 0.0

        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.is_running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass

    def _open_stream(self) -> bool:
        if self.is_mock:
            self.is_connected = True
            return True

        # Jika input berupa integer (misal: '0' untuk webcam lokal)
        src = int(self.source) if self.source.isdigit() else self.source

        # Set environment flag untuk timeout RTSP
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"

        try:
            self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG if isinstance(src, str) else cv2.CAP_ANY)
            # Minimalkan internal buffer OpenCV
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.is_connected = True
                    return True
        except Exception as e:
            print(f" [{self.name}] Error menghubungkan ke {self.source}: {e}")

        self.is_connected = False
        return False

    def _generate_mock_frame(self) -> np.ndarray:
        """Membuat frame simulasi CCTV jika kamera offline atau mode mock diaktifkan"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Gradient background
        frame[:] = (30, 30, 35)

        # Garis grid CCTV
        cv2.line(frame, (0, 240), (640, 240), (45, 45, 55), 1)
        cv2.line(frame, (320, 0), (320, 480), (45, 45, 55), 1)

        t_now = time.strftime("%Y-%m-%d %H:%M:%S")
        status_txt = "SIMULASI CCTV (TEST MODE)" if self.is_mock else "MENGHUBUNGKAN ULANG KE CCTV..."

        cv2.putText(frame, f"[ {self.name.upper()} ]", (25, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 200), 1)
        cv2.putText(frame, f"Waktu: {t_now}", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(frame, f"Source: {self.source[:45]}...", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

        # Kotak status tengah
        box_color = (0, 165, 255) if self.is_mock else (0, 0, 255)
        cv2.rectangle(frame, (120, 200), (520, 280), (20, 20, 25), -1)
        cv2.rectangle(frame, (120, 200), (520, 280), box_color, 2)
        cv2.putText(frame, status_txt, (140, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        # Animasi lingkaran detak
        pulse_r = int(10 + 6 * np.sin(time.time() * 4))
        cv2.circle(frame, (600, 35), pulse_r, (0, 255, 0) if self.is_mock else (0, 0, 255), -1)

        return frame

    def _capture_loop(self):
        fps_counter = 0
        fps_timer = time.time()

        while self.is_running:
            if not self.is_connected:
                success = self._open_stream()
                if not success:
                    # Tampilkan frame standby sembari mencoba menghubungkan ulang
                    with self.lock:
                        self.latest_frame = self._generate_mock_frame()
                    time.sleep(3.0)
                    continue

            if self.is_mock:
                with self.lock:
                    self.latest_frame = self._generate_mock_frame()
                time.sleep(0.04) # ~25 fps
            else:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print(f" [{self.name}] Koneksi RTSP terputus. Mencoba reconnect...")
                    self.is_connected = False
                    if self.cap:
                        self.cap.release()
                    time.sleep(1.0)
                    continue

                with self.lock:
                    self.latest_frame = frame

            # Hitung real-time capture FPS
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
# 5. PIPELINE INFERENSI & ANOTASI VIDEO (HUD OVERLAY)
# ------------------------------------------------------------------------------
class PipelineManager:
    """
    Mengorkestrasi inferensi pengenalan wajah pada multi-kamera
    serta menghasilkan output frame beranotasi untuk video streaming.
    """
    def __init__(self, cameras: Dict[str, ThreadedCamera], detector: FaceDetector, recognizer: FaceRecognitionEngine):
        self.cameras = cameras
        self.detector = detector
        self.recognizer = recognizer

        self.annotated_frames: Dict[str, np.ndarray] = {}
        self.detections_history: List[Dict] = []
        self.camera_stats: Dict[str, Dict] = {}
        self.lock = threading.Lock()
        self.is_running = False

    def start(self):
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.worker_thread.start()
        return self

    def _draw_annotation(self, frame: np.ndarray, detections: List[Dict], cam_name: str, inf_time_ms: float) -> np.ndarray:
        """Menggambar bounding box, nama mahasiswa, kemiripan, dan HUD canggih"""
        out = frame.copy()
        h, w = out.shape[:2]

        # 1. Gambar Bounding Box Wajah
        for d in detections:
            x, y, bw, bh = d["box"]
            name = d["name"]
            sim = d["similarity"]
            dist = d["distance"]

            if name != "Unknown":
                color = (46, 204, 113) # Hijau zamrud (Match)
                tag_text = f"{name} ({sim:.1f}%)"
            else:
                color = (41, 128, 250) # Oranye/Merah (Unknown)
                tag_text = f"Unknown (d:{dist:.2f})"

            # Bounding box dengan sudut elegan
            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
            corner_len = min(20, int(bw * 0.25))
            # Corner accents
            cv2.line(out, (x, y), (x + corner_len, y), color, 4)
            cv2.line(out, (x, y), (x, y + corner_len), color, 4)
            cv2.line(out, (x + bw, y), (x + bw - corner_len, y), color, 4)
            cv2.line(out, (x + bw, y), (x + bw, y + corner_len), color, 4)
            cv2.line(out, (x, y + bh), (x + corner_len, y + bh), color, 4)
            cv2.line(out, (x, y + bh), (x, y + bh - corner_len), color, 4)
            cv2.line(out, (x + bw, y + bh), (x + bw - corner_len, y + bh), color, 4)
            cv2.line(out, (x + bw, y + bh), (x + bw, y + bh - corner_len), color, 4)

            # Banner Label Nama
            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
            label_y1 = max(0, y - th - 12)
            label_y2 = y
            cv2.rectangle(out, (x, label_y1), (x + tw + 16, label_y2), color, -1)
            cv2.putText(out, tag_text, (x + 8, label_y2 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Top Header HUD (Nama Lab, Waktu, FPS, Latensi)
        hud_bg = np.zeros((40, w, 3), dtype=np.uint8)
        overlay = out[0:40, 0:w].copy()
        cv2.addWeighted(hud_bg, 0.65, overlay, 0.35, 0, out[0:40, 0:w])

        t_stamp = time.strftime("%H:%M:%S")
        fps = self.camera_stats.get(cam_name, {}).get("fps", 0.0)

        cv2.putText(out, f"CAM: {cam_name.upper()}", (12, 26), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 220), 1, cv2.LINE_AA)
        cv2.putText(out, f"FPS: {fps:.1f} | Latensi AI: {inf_time_ms:.1f}ms", (w // 2 - 120, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(out, f"{t_stamp}", (w - 100, 26), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        return out

    def _process_loop(self):
        while self.is_running:
            for cam_key, cam in self.cameras.items():
                frame = cam.get_frame()
                if frame is None:
                    continue

                t0 = time.time()
                # 1. Deteksi Wajah
                raw_faces = self.detector.detect_faces(frame)

                detections = []
                for (bbox, face_crop, conf) in raw_faces:
                    # 2. Pengenalan Wajah dengan FaceNet ONNX
                    name, dist, sim = self.recognizer.recognize(face_crop)
                    det_item = {
                        "box": bbox,
                        "name": name,
                        "similarity": sim,
                        "distance": dist,
                        "camera": cam.name,
                        "timestamp": time.strftime("%H:%M:%S")
                    }
                    detections.append(det_item)

                    # Simpan ke riwayat jika nama valid
                    if name != "Unknown":
                        self._log_detection(name, cam.name, sim)

                t_inf = (time.time() - t0) * 1000.0

                # 3. Anotasi Frame
                annotated = self._draw_annotation(frame, detections, cam.name, t_inf)

                with self.lock:
                    self.annotated_frames[cam_key] = annotated
                    self.camera_stats[cam.name] = {
                        "fps": cam.fps,
                        "faces_detected": len(detections),
                        "latency_ms": round(t_inf, 1),
                        "active_people": [d["name"] for d in detections if d["name"] != "Unknown"]
                    }

            time.sleep(0.01)

    def _log_detection(self, name: str, camera_name: str, similarity: float):
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        # Hindari spam jika terdeteksi dalam 5 detik terakhir di kamera yang sama
        with self.lock:
            for item in reversed(self.detections_history[-10:]):
                if item["name"] == name and item["camera"] == camera_name:
                    return # Skip duplikasi log instan

            self.detections_history.append({
                "name": name,
                "camera": camera_name,
                "similarity": f"{similarity:.1f}%",
                "time": now_str
            })
            if len(self.detections_history) > 50:
                self.detections_history.pop(0)

    def get_annotated_frame(self, cam_key: str) -> Optional[np.ndarray]:
        with self.lock:
            return self.annotated_frames.get(cam_key)

    def get_mosaic_frame(self) -> Optional[np.ndarray]:
        """Menggabungkan video kedua kamera menjadi satu frame berdampingan (Side-by-Side)"""
        with self.lock:
            f1 = self.annotated_frames.get("cam1")
            f2 = self.annotated_frames.get("cam2")

        if f1 is None and f2 is None:
            return None

        target_h, target_w = 360, 640
        def resize_std(img):
            if img is None:
                blank = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                cv2.putText(blank, "NO SIGNAL", (240, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                return blank
            return cv2.resize(img, (target_w, target_h))

        s1 = resize_std(f1)
        s2 = resize_std(f2)
        return np.hstack([s1, s2])

    def get_status_dict(self) -> Dict:
        with self.lock:
            return {
                "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "cameras": self.camera_stats,
                "recent_detections": list(reversed(self.detections_history[-15:])),
                "ai_provider": self.recognizer.active_provider,
                "detector_backend": self.detector.active_backend
            }

# ------------------------------------------------------------------------------
# 6. SERVER WEB STREAMING HTTP (MJPEG + DASHBOARD HTML MODERN)
# ------------------------------------------------------------------------------
HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>E.S.I.S - Multi-Camera Face Recognition (RunPod Node)</title>
    <style>
        :root {
            --bg-primary: #0f172a;
            --bg-card: #1e293b;
            --border-color: #334155;
            --accent-green: #10b981;
            --accent-blue: #3b82f6;
            --accent-cyan: #06b6d4;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
        body { background: var(--bg-primary); color: var(--text-main); min-height: 100vh; padding: 20px; }
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); padding-bottom: 16px; margin-bottom: 24px; }
        .logo-title h1 { font-size: 20px; font-weight: 700; color: #fff; letter-spacing: -0.5px; }
        .logo-title p { font-size: 13px; color: var(--text-muted); margin-top: 4px; }
        .badge { padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 600; background: rgba(16, 185, 129, 0.15); color: var(--accent-green); border: 1px solid var(--accent-green); }
        
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .stat-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 16px; }
        .stat-card .label { font-size: 12px; color: var(--text-muted); text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }
        .stat-card .val { font-size: 22px; font-weight: 700; color: #fff; margin-top: 6px; }

        .video-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 20px; margin-bottom: 24px; }
        .cam-box { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; overflow: hidden; }
        .cam-header { padding: 12px 16px; background: rgba(15, 23, 42, 0.6); display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); }
        .cam-header h3 { font-size: 14px; font-weight: 600; color: var(--text-main); }
        .feed-wrapper { width: 100%; aspect-ratio: 16 / 9; background: #000; position: relative; }
        .feed-wrapper img { width: 100%; height: 100%; object-fit: contain; }
        
        .log-section { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; }
        .log-section h2 { font-size: 16px; font-weight: 600; margin-bottom: 12px; }
        .log-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .log-table th { text-align: left; padding: 10px; border-bottom: 1px solid var(--border-color); color: var(--text-muted); font-weight: 600; }
        .log-table td { padding: 10px; border-bottom: 1px solid rgba(51, 65, 85, 0.4); }
        .tag-match { color: var(--accent-green); font-weight: 600; }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo-title">
            <h1>E.S.I.S - Multi-Camera Face Recognition Node</h1>
            <p>Laboratorium Riset Lt 5 & Laboratorium Sistem Digital | Powered by FaceNet ONNX</p>
        </div>
        <div>
            <span class="badge" id="ai-status">RUNPOD GPU NODE ACTIVE</span>
        </div>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="label">Inference Engine</div>
            <div class="val" id="engine-val">-</div>
        </div>
        <div class="stat-card">
            <div class="label">Detector Backend</div>
            <div class="val" id="detector-val">-</div>
        </div>
        <div class="stat-card">
            <div class="label">Cam 1 (Lab Riset Lt 5)</div>
            <div class="val" id="cam1-fps">-</div>
        </div>
        <div class="stat-card">
            <div class="label">Cam 2 (Lab Sisdig)</div>
            <div class="val" id="cam2-fps">-</div>
        </div>
    </div>

    <div class="video-container">
        <div class="cam-box">
            <div class="cam-header">
                <h3>  * Kamera 1: Lab Riset Lantai 5</h3>
                <span class="badge">LIVE FEED</span>
            </div>
            <div class="feed-wrapper">
                <img src="/video_feed/1" alt="Feed Lab Riset Lt 5">
            </div>
        </div>
        <div class="cam-box">
            <div class="cam-header">
                <h3>  * Kamera 2: Lab Sistem Digital</h3>
                <span class="badge">LIVE FEED</span>
            </div>
            <div class="feed-wrapper">
                <img src="/video_feed/2" alt="Feed Lab Sisdig">
            </div>
        </div>
    </div>

    <div class="log-section">
        <h2>📋 Riwayat Wajah Teridentifikasi Secara Real-Time</h2>
        <table class="log-table">
            <thead>
                <tr>
                    <th>Waktu</th>
                    <th>Nama Mahasiswa</th>
                    <th>Lokasi Kamera</th>
                    <th>Tingkat Kemiripan</th>
                </tr>
            </thead>
            <tbody id="log-body">
                <tr><td colspan="4" style="text-align: center; color: var(--text-muted);">Menunggu aktivitas wajah...</td></tr>
            </tbody>
        </table>
    </div>

    <script>
        async function fetchStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                document.getElementById('engine-val').textContent = data.ai_provider;
                document.getElementById('detector-val').textContent = data.detector_backend;

                const c1 = data.cameras['Lab Riset Lt 5'];
                if (c1) {
                    document.getElementById('cam1-fps').textContent = `${c1.fps.toFixed(1)} FPS (${c1.latency_ms}ms)`;
                }
                const c2 = data.cameras['Lab Sisdig'];
                if (c2) {
                    document.getElementById('cam2-fps').textContent = `${c2.fps.toFixed(1)} FPS (${c2.latency_ms}ms)`;
                }

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
            } catch (e) {
                console.error(e);
            }
        }
        setInterval(fetchStatus, 1500);
        fetchStatus();
    </script>
</body>
</html>
"""

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

def make_stream_handler(pipeline: PipelineManager):
    class StreamHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass # Matikan logging per request agar terminal bersih

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(HTML_DASHBOARD.encode("utf-8"))
                return

            if self.path == "/api/status":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(pipeline.get_status_dict()).encode("utf-8"))
                return

            # MJPEG Stream Endpoint
            target_feed = None
            if self.path == "/video_feed/1":
                target_feed = "cam1"
            elif self.path == "/video_feed/2":
                target_feed = "cam2"
            elif self.path == "/video_feed/mosaic":
                target_feed = "mosaic"

            if target_feed:
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
                self.end_headers()

                try:
                    while True:
                        if target_feed == "mosaic":
                            frame = pipeline.get_mosaic_frame()
                        else:
                            frame = pipeline.get_annotated_frame(target_feed)

                        if frame is not None:
                            ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                            if ret:
                                b_data = jpeg.tobytes()
                                self.wfile.write(b"--FRAME\r\n")
                                self.send_header("Content-Type", "image/jpeg")
                                self.send_header("Content-Length", str(len(b_data)))
                                self.end_headers()
                                self.wfile.write(b_data)
                                self.wfile.write(b"\r\n")

                        time.sleep(0.035) # ~28 fps cap
                except Exception:
                    pass
                return

            self.send_error(404, "Halaman Tidak Ditemukan")

    return StreamHandler

# ------------------------------------------------------------------------------
# 7. MAIN RUNNER & CLI ARGUMENTS
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="E.S.I.S Multi-Camera Face Recognition Node for RunPod / Local Deployment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--cam1", type=str, default=DEFAULT_RTSP_CAM1, help="RTSP URL / index kamera Lab Riset Lt 5")
    parser.add_argument("--cam2", type=str, default=DEFAULT_RTSP_CAM2, help="RTSP URL / index kamera Lab Sisdig")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ONNX, help="Path model ONNX FaceNet")
    parser.add_argument("--database", type=str, default=DEFAULT_DATABASE_NPZ, help="Path database embedding .npz")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port Web Streaming HTTP Server")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Threshold Euclidean distance (0.7 - 0.8)")
    parser.add_argument("--detector", type=str, default="auto", choices=["auto", "mtcnn", "yunet", "haar"], help="Pilihan backend face detector")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Pilihan komputasi FaceNet")
    parser.add_argument("--mock", action="store_true", help="Jalankan dalam mode simulasi CCTV tanpa RTSP fisik")
    parser.add_argument("--display", action="store_true", help="Buka jendela lokal OpenCV GUI (jika ada monitor desktop)")

    args = parser.parse_args()

    print("=" * 75)
    print("[START] MEMULAI E.S.I.S MULTI-CAMERA FACE RECOGNITION NODE")
    print("=" * 75)
    print(f"  * Model FaceNet   : {args.model}")
    print(f"  * Database NPZ    : {args.database}")
    print(f"  * Kamera 1 (Riset): {args.cam1}")
    print(f"  * Kamera 2 (Sisdig: {args.cam2}")
    print(f"  * HTTP Web Port   : {args.port}")
    print(f"  * Threshold Jarak : {args.threshold}")
    print("=" * 75)

    # 1. Inisialisasi Detektor Wajah
    detector = FaceDetector(mode=args.detector)

    # 2. Inisialisasi Engine Pengenalan Wajah
    recognizer = FaceRecognitionEngine(
        model_path=args.model,
        npz_database_path=args.database,
        threshold=args.threshold,
        device=args.device
    )

    # 3. Inisialisasi Kamera CCTV
    cam1 = ThreadedCamera(name="Lab Riset Lt 5", source=args.cam1, is_mock=args.mock).start()
    cam2 = ThreadedCamera(name="Lab Sisdig", source=args.cam2, is_mock=args.mock).start()

    cameras = {"cam1": cam1, "cam2": cam2}

    # 4. Inisialisasi Pipeline AI
    pipeline = PipelineManager(cameras=cameras, detector=detector, recognizer=recognizer).start()

    # 5. Jalankan Web Streaming Server di Background
    handler = make_stream_handler(pipeline)
    server = ThreadedHTTPServer(("0.0.0.0", args.port), handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print("\n[OK] SERVER STREAMING HTTP BERHASIL BERJALAN!")
    print(f"  -> Akses Dashboard Web   : http://localhost:{args.port}/")
    print(f"  -> Stream Kamera 1 (Riset): http://localhost:{args.port}/video_feed/1")
    print(f"  -> Stream Kamera 2 (Sisdig: http://localhost:{args.port}/video_feed/2")
    print(f"  -> Stream Mosaic (Split)  : http://localhost:{args.port}/video_feed/mosaic")
    print(f"  -> Endpoint API Status    : http://localhost:{args.port}/api/status")
    print("-" * 75)
    print("Tekan [Ctrl + C] untuk menghentikan program.")
    print("-" * 75)

    try:
        while True:
            if args.display:
                m_frame = pipeline.get_mosaic_frame()
                if m_frame is not None:
                    cv2.imshow("E.S.I.S Face Recognition - Dual Camera Live", m_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[WAIT] Menghentikan seluruh worker kamera & server...")
    finally:
        cam1.stop()
        cam2.stop()
        server.shutdown()
        cv2.destroyAllWindows()
        print("[OK] Program berhasil dihentikan secara bersih.")

if __name__ == "__main__":
    main()
