#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
E.S.I.S (Electronic Smoking Identification System)
Multi-Camera Face Recognition Node for RunPod & Local Deployment
================================================================================
Features:
- 4-Camera Support (Lab Riset Lt 5, Lab Sisdig, Lab OS, Lab Programming)
- Native HLS Ingestion via OpenCV VideoCapture (FFmpeg Decoder - H.264 & H.265)
- Native WebRTC (WHEP) Support via aiortc & PyAV for ultra low-latency
- High-Performance Face Recognition:
    * Detection: MTCNN (primary) / YuNet (fast ONNX) / Haar Cascade (fallback)
    * Feature Extraction: FaceNet (ONNX Runtime with GPU CUDA acceleration)
    * Matching: L2-Normalized Euclidean Distance against data_wajah_facenet_mtcnn.npz
- Built-in Web Streaming Server (HTTP MJPEG & Snapshot API):
    * Modern dark responsive dashboard accessible via browser or RunPod HTTP proxy port (8000)
    * Flush & X-Accel-Buffering optimized for Nginx/RunPod proxy (zero-stall streaming)
    * Individual camera feeds (/video_feed/1, /2, /3, /4)
    * Combined 2x2 mosaic view (/video_feed/mosaic)
    * Snapshot fallback API (/snapshot/1, /2, /3, /4, /mosaic)
    * Live status JSON API (/api/status)
- Desktop GUI support via OpenCV window (--display)
- Interactive Simulation Mode (--mock) with real volunteer face recognition
- Threaded non-blocking capture with auto-reconnection
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
# 1. KONFIGURASI DEFAULT & PATH MODEL
# ------------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODEL_ONNX = os.path.join(SCRIPT_DIR, "facenet.onnx")
DEFAULT_DATABASE_NPZ = os.path.join(SCRIPT_DIR, "data_wajah_facenet_mtcnn.npz")

# Base URLs
DEFAULT_HLS_BASE = os.environ.get("HLS_BASE", "https://fdxuqgcmfo.a.pinggy.link").rstrip("/")
DEFAULT_WHEP_BASE = os.environ.get("STREAM_BASE", "https://stream.ucim.my.id").rstrip("/")

# 4 Kamera Utama (Default: HLS via Pinggy Tunnel agar H.264 & H.265 terbaca lancar)
DEFAULT_CAM1_PATH = "ecam_lab_riset_lt_5"      # Lab Riset Lt 5 (H.265)
DEFAULT_CAM2_PATH = "ecam_lab_sistem_digital"   # Lab Sistem Digital (H.264)
DEFAULT_CAM3_PATH = "ecam_lab_os"               # Lab OS (H.264)
DEFAULT_CAM4_PATH = "ecam_lab_programming"      # Lab Programming (H.264)

DEFAULT_PORT = int(os.environ.get("PORT", "8000"))
DEFAULT_THRESHOLD = float(os.environ.get("THRESHOLD", "0.75"))

def normalize_camera_source(url: str, protocol: str = "auto") -> Tuple[str, bool]:
    """
    Menormalkan URL stream kamera dan menentukan apakah menggunakan WebRTC (WHEP)
    atau OpenCV VideoCapture (HLS/RTSP/Webcam).
    Mengembalikan: (normalized_url, is_webrtc)
    """
    if not isinstance(url, str):
        return url, False

    u = url.strip()
    if u.isdigit():
        return int(u), False

    # Jika RTSP
    if u.startswith("rtsp://"):
        return u, False

    proto = protocol.lower()

    # WebRTC eksplisit
    if proto == "webrtc" or u.endswith("/whep") or "/whep" in u or u.startswith("webrtc://"):
        if u.startswith("webrtc://"):
            u = u.replace("webrtc://", "https://")
        clean = u.rstrip("/")
        if not clean.endswith("/whep"):
            clean = clean.replace("/index.m3u8", "").rstrip("/")
            clean = f"{clean}/whep"
        return clean, True

    # HLS eksplisit
    if proto == "hls" or u.endswith(".m3u8"):
        clean = u.rstrip("/")
        if not clean.endswith(".m3u8") and not clean.endswith(".mp4"):
            clean = f"{clean}/index.m3u8"
        return clean, False

    # Auto mode untuk HTTP/HTTPS
    if u.startswith("http://") or u.startswith("https://"):
        if "/whep" in u:
            return u, True
        if ".m3u8" in u:
            return u, False
        if "pinggy.link" in u:
            clean = u.rstrip("/")
            if not clean.endswith(".m3u8"):
                clean = f"{clean}/index.m3u8"
            return clean, False
        if "stream.ucim.my.id" in u:
            clean = u.rstrip("/").replace("/index.m3u8", "")
            return f"{clean}/whep", True
        else:
            clean = u.rstrip("/")
            return f"{clean}/index.m3u8", False

    return u, False

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
        self.detector_obj = None

        self._init_detector()

    def _init_detector(self):
        # 1. Coba MTCNN
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

        # 2. Coba YuNet
        if self.mode in ("auto", "yunet"):
            yunet_weights = os.path.join(SCRIPT_DIR, "face_detection_yunet_2023mar.onnx")
            if os.path.exists(yunet_weights):
                try:
                    self.detector_obj = cv2.FaceDetectorYN.create(
                        yunet_weights, "", (320, 320), self.conf_threshold, 0.3, 5000
                    )
                    self.active_backend = "OpenCV YuNet (ONNX)"
                    print(f" [FaceDetector] Backend aktif: {self.active_backend}")
                    return
                except Exception as e:
                    if self.mode == "yunet":
                        print(f" [FaceDetector] Gagal memuat YuNet ({e}), mencoba fallback...")

        # 3. Fallback Haar Cascade (Bawaan OpenCV)
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector_obj = cv2.CascadeClassifier(haar_path)
        self.active_backend = "OpenCV Haar Cascade (Fast Built-in)"
        print(f" [FaceDetector] Backend aktif: {self.active_backend}")

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], np.ndarray, float]]:
        """
        Mendeteksi seluruh wajah pada frame.
        Mengembalikan list tuple: ((x, y, w, h), cropped_face_160x160_rgb, confidence)
        """
        results = []
        if frame is None or frame.size == 0:
            return results

        h, w = frame.shape[:2]

        if "MTCNN" in self.active_backend:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                detections = self.detector_obj.detect_faces(rgb_frame)
                for det in detections:
                    conf = det.get("confidence", 0.0)
                    if conf < self.conf_threshold:
                        continue
                    x, y, bw, bh = det["box"]
                    x, y = max(0, x), max(0, y)
                    bw, bh = min(w - x, bw), min(h - y, bh)
                    if bw < 25 or bh < 25:
                        continue
                    crop = rgb_frame[y:y + bh, x:x + bw]
                    if crop.size == 0:
                        continue
                    crop_160 = cv2.resize(crop, (160, 160))
                    results.append(((x, y, bw, bh), crop_160, conf))
            except Exception:
                pass
            return results

        elif "YuNet" in self.active_backend:
            self.detector_obj.setInputSize((w, h))
            try:
                _, faces = self.detector_obj.detect(frame)
                if faces is not None:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    for face in faces:
                        conf = float(face[-1])
                        if conf < self.conf_threshold:
                            continue
                        x, y, bw, bh = map(int, face[0:4])
                        x, y = max(0, x), max(0, y)
                        bw, bh = min(w - x, bw), min(h - y, bh)
                        if bw < 25 or bh < 25:
                            continue
                        crop = rgb_frame[y:y + bh, x:x + bw]
                        if crop.size == 0:
                            continue
                        crop_160 = cv2.resize(crop, (160, 160))
                        results.append(((x, y, bw, bh), crop_160, conf))
            except Exception:
                pass
            return results

        else: # Haar Cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector_obj.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=5, minSize=(30, 30)
            )
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for (x, y, bw, bh) in faces:
                x, y = max(0, x), max(0, y)
                bw, bh = min(w - x, bw), min(h - y, bh)
                crop = rgb_frame[y:y + bh, x:x + bw]
                if crop.size == 0:
                    continue
                crop_160 = cv2.resize(crop, (160, 160))
                results.append(((x, y, bw, bh), crop_160, 0.90))
            return results

# ------------------------------------------------------------------------------
# 3. ENGINE REKOGNISI WAJAH (FACENET ONNX + NPZ DATABASE)
# ------------------------------------------------------------------------------
class FaceRecognitionEngine:
    """
    Ekstraksi embedding 512 dimensi via FaceNet ONNX dan pencocokan Euclidean Distance.
    Mendukung akselerasi CUDA GPU di RunPod dan CPU fallback otomatis.
    """
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
                print(f" [FaceRecognition] CUDA tidak dapat diaktifkan ({e}). Menggunakan CPUExecutionProvider.")

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
# 4. WORKER THREAD KAMERA CCTV (HLS / WEBRTC / OPENCV DENGAN SIMULASI MOCK)
# ------------------------------------------------------------------------------
class ThreadedCamera:
    """
    Menangani streaming kamera video di thread terpisah.
    Mendukung HLS (FFmpeg Native OpenCV) untuk stabilitas H.264 & H.265,
    WebRTC (WHEP) sub-second latency via aiortc & PyAV,
    serta mode simulasi mock dengan penanaman wajah volunteer nyata.
    """
    def __init__(self, name: str, source: str, protocol: str = "auto", is_mock: bool = False):
        self.name = name
        self.raw_source = source
        self.is_mock = is_mock
        self.protocol = protocol

        self.source, self.is_webrtc = normalize_camera_source(source, protocol)

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        self.fps = 0.0
        self.is_running = False

        self.mock_face_img: Optional[np.ndarray] = None
        self.mock_face_name: str = ""
        if self.is_mock:
            self._load_mock_volunteer_face()

        self.latest_frame: Optional[np.ndarray] = self._generate_mock_frame()

        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def _load_mock_volunteer_face(self):
        """Memuat sampel foto wajah volunteer agar simulasi deteksi AI berfungsi 100% nyata."""
        try:
            candidates = [
                os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "known_faces")),
                os.path.abspath(os.path.join(SCRIPT_DIR, "..", "known_faces")),
                os.path.abspath(os.path.join(SCRIPT_DIR, "known_faces")),
            ]
            base_known = None
            for c in candidates:
                if os.path.exists(c) and os.path.isdir(c):
                    base_known = c
                    break

            if base_known:
                subdirs = sorted([d for d in os.listdir(base_known) if os.path.isdir(os.path.join(base_known, d))])
                if subdirs:
                    idx = abs(hash(self.name)) % len(subdirs)
                    chosen_dir = os.path.join(base_known, subdirs[idx])
                    photos = [f for f in os.listdir(chosen_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                    if photos:
                        p_path = os.path.join(chosen_dir, photos[0])
                        img = cv2.imread(p_path)
                        if img is not None:
                            self.mock_face_img = img
                            self.mock_face_name = subdirs[idx]
                            print(f" [{self.name}] [Simulasi] Volunteer diaktifkan: {self.mock_face_name}")
        except Exception:
            pass

    def start(self):
        self.is_running = True
        if self.is_mock:
            self.thread = threading.Thread(target=self._mock_loop, daemon=True)
        elif self.is_webrtc:
            print(f" [{self.name}] Mode input: WebRTC (WHEP) Low-Latency")
            self.thread = threading.Thread(target=self._webrtc_thread_entry, daemon=True)
        else:
            print(f" [{self.name}] Mode input: OpenCV VideoCapture (HLS / RTSP)")
            self.thread = threading.Thread(target=self._opencv_loop, daemon=True)

        self.thread.start()
        return self

    def stop(self):
        self.is_running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass

    def _generate_mock_frame(self) -> np.ndarray:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (25, 28, 35)

        cv2.line(frame, (0, 240), (640, 240), (45, 48, 58), 1)
        cv2.line(frame, (320, 0), (320, 480), (45, 48, 58), 1)

        t_now = time.strftime("%Y-%m-%d %H:%M:%S")
        if self.is_mock:
            status_txt = "SIMULASI CCTV (TEST MODE)"
            box_color = (0, 165, 255)
        elif self.is_webrtc:
            status_txt = "MENGHUBUNGKAN KE WEBRTC..."
            box_color = (0, 200, 255)
        else:
            status_txt = "MENGHUBUNGKAN KE CCTV (HLS/RTSP)..."
            box_color = (0, 0, 255)

        cv2.putText(frame, f"[ {self.name.upper()} ]", (25, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 200), 1)
        cv2.putText(frame, f"Waktu: {t_now}", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        src_str = str(self.source)
        if len(src_str) > 45:
            src_str = src_str[:42] + "..."
        cv2.putText(frame, f"Source: {src_str}", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

        cv2.rectangle(frame, (90, 200), (550, 280), (18, 20, 26), -1)
        cv2.rectangle(frame, (90, 200), (550, 280), box_color, 2)
        cv2.putText(frame, status_txt, (110, 248), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

        pulse_r = int(10 + 6 * np.sin(time.time() * 4))
        dot_color = (0, 255, 0) if self.is_connected else (0, 0, 255)
        cv2.circle(frame, (600, 35), pulse_r, dot_color, -1)

        # Jika ada foto volunteer untuk mock, tempelkan ke frame dengan animasi gerakan
        if self.is_mock and self.mock_face_img is not None:
            hf, wf = self.mock_face_img.shape[:2]
            target_h = 190
            target_w = int(wf * (target_h / hf))
            face_resized = cv2.resize(self.mock_face_img, (target_w, target_h))

            shift_x = int(35 * np.sin(time.time() * 1.5))
            shift_y = int(12 * np.cos(time.time() * 1.5))
            pos_x = max(20, min(640 - target_w - 20, 240 + shift_x))
            pos_y = max(110, min(480 - target_h - 30, 150 + shift_y))

            frame[pos_y:pos_y + target_h, pos_x:pos_x + target_w] = face_resized
            cv2.rectangle(frame, (pos_x, pos_y), (pos_x + target_w, pos_y + target_h), (80, 80, 80), 1)

        return frame

    # ------------------ WebRTC (WHEP) Ingestion Loop ------------------
    def _webrtc_thread_entry(self):
        import asyncio
        while self.is_running:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._webrtc_session())
                loop.close()
            except Exception as e:
                print(f" [{self.name}] [WebRTC] Sesi terputus ({e}). Reconnecting dalam 3s...")
                self.is_connected = False
                with self.lock:
                    self.latest_frame = self._generate_mock_frame()
                time.sleep(3.0)

    async def _webrtc_session(self):
        import asyncio
        import aiohttp
        from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer

        print(f" [{self.name}] Menghubungkan ke WHEP: {self.source}")
        config = RTCConfiguration(
            iceServers=[
                RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
                RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
            ]
        )
        pc = RTCPeerConnection(configuration=config)
        pc.addTransceiver("video", direction="recvonly")
        track_ended = asyncio.Event()

        @pc.on("track")
        def on_track(track):
            if track.kind == "video":
                print(f" [{self.name}] [WebRTC] Track video diterima! Memulai decode frame...")
                asyncio.create_task(self._read_webrtc_track(track, track_ended))

        @pc.on("connectionstatechange")
        def on_connectionstatechange():
            if pc.connectionState in ("failed", "closed"):
                print(f" [{self.name}] [WebRTC] Sesi terputus (state: {pc.connectionState})")
                track_ended.set()

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        headers = {"Content-Type": "application/sdp"}
        location = None
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.source,
                data=pc.localDescription.sdp,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=8)
            ) as resp:
                if resp.status not in (200, 201):
                    body = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status} dari WHEP endpoint: {body}")
                location = resp.headers.get("Location")
                answer_sdp = await resp.text()

        answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
        await pc.setRemoteDescription(answer)

        await track_ended.wait()

        self.is_connected = False
        try:
            await pc.close()
        except Exception:
            pass

        if location:
            try:
                del_url = urllib.parse.urljoin(self.source, location)
                async with aiohttp.ClientSession() as s:
                    await s.delete(del_url, timeout=aiohttp.ClientTimeout(total=2))
            except Exception:
                pass

    async def _read_webrtc_track(self, track, track_ended):
        import asyncio
        fps_counter = 0
        fps_timer = time.time()
        first_frame = True

        while self.is_running and not track_ended.is_set():
            try:
                frame = await asyncio.wait_for(track.recv(), timeout=6.0)
            except asyncio.TimeoutError:
                print(f" [{self.name}] [WebRTC] Timeout menerima frame (6s). Reconnecting...")
                break
            except Exception as e:
                print(f" [{self.name}] [WebRTC] Frame error: {e}")
                break

            img_bgr = frame.to_ndarray(format="bgr24")
            if first_frame:
                print(f" [{self.name}] [OK] Frame WebRTC pertama DITERIMA! Resolusi: {img_bgr.shape[1]}x{img_bgr.shape[0]}")
                first_frame = False

            with self.lock:
                self.latest_frame = img_bgr
                self.is_connected = True

            fps_counter += 1
            t_now = time.time()
            if t_now - fps_timer >= 1.0:
                self.fps = fps_counter / (t_now - fps_timer)
                fps_counter = 0
                fps_timer = t_now

        track_ended.set()

    # ------------------ OpenCV VideoCapture Fallback Loop ------------------
    def _open_opencv_stream(self) -> bool:
        src = self.source
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|stimeout;3000000|rw_timeout;3000000|timeout;3000000|http_user_agent;Mozilla/5.0"
        )
        try:
            print(f" [{self.name}] Menghubungkan via OpenCV: {src}")
            self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG if isinstance(src, str) else cv2.CAP_ANY)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    print(f" [{self.name}] [OK] OpenCV Stream TERHUBUNG! Resolusi: {frame.shape[1]}x{frame.shape[0]}")
                    self.is_connected = True
                    return True
        except Exception as e:
            print(f" [{self.name}] Gagal membuka {src}: {e}")

        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self.is_connected = False
        return False

    def _opencv_loop(self):
        fps_counter = 0
        fps_timer = time.time()

        while self.is_running:
            if not self.is_connected:
                success = self._open_opencv_stream()
                if not success:
                    with self.lock:
                        self.latest_frame = self._generate_mock_frame()
                    time.sleep(3.0)
                    continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                print(f" [{self.name}] Koneksi terputus. Mencoba reconnect...")
                self.is_connected = False
                if self.cap:
                    self.cap.release()
                time.sleep(1.0)
                continue

            with self.lock:
                self.latest_frame = frame

            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                self.fps = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer = time.time()

            time.sleep(0.005)

    def _mock_loop(self):
        self.is_connected = True
        self.fps = 25.0
        while self.is_running:
            with self.lock:
                self.latest_frame = self._generate_mock_frame()
            time.sleep(0.04)

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

# ------------------------------------------------------------------------------
# 5. PIPELINE INFERENSI & ANOTASI VIDEO (HUD OVERLAY)
# ------------------------------------------------------------------------------
class PipelineManager:
    """
    Mengorkestrasi inferensi pengenalan wajah pada multi-kamera (4 kamera)
    serta menghasilkan output frame beranotasi untuk video streaming.
    """
    def __init__(self, cameras: Dict[str, ThreadedCamera], detector: FaceDetector, recognizer: FaceRecognitionEngine):
        self.cameras = cameras
        self.detector = detector
        self.recognizer = recognizer

        self.annotated_frames: Dict[str, np.ndarray] = {
            cam_key: cam.get_frame() for cam_key, cam in cameras.items()
        }
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
        out = frame.copy()
        h, w = out.shape[:2]

        for d in detections:
            x, y, bw, bh = d["box"]
            name = d["name"]
            sim = d["similarity"]
            dist = d["distance"]

            if name != "Unknown":
                color = (46, 204, 113) # Hijau
                tag_text = f"{name} ({sim:.1f}%)"
            else:
                color = (41, 128, 250) # Oranye/Merah
                tag_text = f"Unknown (d:{dist:.2f})"

            # Bounding box & sudut aksen
            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
            corner_len = min(20, int(bw * 0.25))
            cv2.line(out, (x, y), (x + corner_len, y), color, 4)
            cv2.line(out, (x, y), (x, y + corner_len), color, 4)
            cv2.line(out, (x + bw, y), (x + bw - corner_len, y), color, 4)
            cv2.line(out, (x + bw, y), (x + bw, y + corner_len), color, 4)
            cv2.line(out, (x, y + bh), (x + corner_len, y + bh), color, 4)
            cv2.line(out, (x, y + bh), (x, y + bh - corner_len), color, 4)
            cv2.line(out, (x + bw, y + bh), (x + bw - corner_len, y + bh), color, 4)
            cv2.line(out, (x + bw, y + bh), (x + bw, y + bh - corner_len), color, 4)

            # Label banner
            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
            label_y1 = max(0, y - th - 12)
            label_y2 = y
            cv2.rectangle(out, (x, label_y1), (x + tw + 16, label_y2), color, -1)
            cv2.putText(out, tag_text, (x + 8, label_y2 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Header HUD
        hud_bg = np.zeros((40, w, 3), dtype=np.uint8)
        overlay = out[0:40, 0:w].copy()
        cv2.addWeighted(hud_bg, 0.65, overlay, 0.35, 0, out[0:40, 0:w])

        t_stamp = time.strftime("%H:%M:%S")
        fps = self.camera_stats.get(cam_name, {}).get("fps", 0.0)

        cv2.putText(out, f"CAM: {cam_name.upper()}", (12, 26), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 220), 1, cv2.LINE_AA)
        cv2.putText(out, f"FPS: {fps:.1f} | Latensi AI: {inf_time_ms:.1f}ms", (w // 2 - 130, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(out, f"{t_stamp}", (w - 95, 26), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        return out

    def _process_loop(self):
        while self.is_running:
            for cam_key, cam in self.cameras.items():
                frame = cam.get_frame()
                if frame is None:
                    continue

                t0 = time.time()
                raw_faces = self.detector.detect_faces(frame)

                detections = []
                for (bbox, face_crop, conf) in raw_faces:
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

                    if name != "Unknown":
                        self._log_detection(name, cam.name, sim)

                t_inf = (time.time() - t0) * 1000.0

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
            if len(self.detections_history) > 50:
                self.detections_history.pop(0)

    def get_annotated_frame(self, cam_key: str) -> Optional[np.ndarray]:
        with self.lock:
            frame = self.annotated_frames.get(cam_key)
            if frame is not None:
                return frame
        cam = self.cameras.get(cam_key)
        return cam.get_frame() if cam else None

    def get_mosaic_frame(self) -> Optional[np.ndarray]:
        """Menghasilkan frame 2x2 grid resolusi HD (1280x720) menggabungkan ke-4 kamera."""
        with self.lock:
            f1 = self.annotated_frames.get("cam1")
            f2 = self.annotated_frames.get("cam2")
            f3 = self.annotated_frames.get("cam3")
            f4 = self.annotated_frames.get("cam4")

        target_h, target_w = 360, 640
        def resize_std(img, label):
            if img is None:
                blank = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                cv2.putText(blank, f"NO SIGNAL ({label})", (160, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 2)
                return blank
            return cv2.resize(img, (target_w, target_h))

        s1 = resize_std(f1, "CAM 1 - LAB RISET")
        s2 = resize_std(f2, "CAM 2 - LAB SISDIG")
        s3 = resize_std(f3, "CAM 3 - LAB OS")
        s4 = resize_std(f4, "CAM 4 - LAB PROG")

        row1 = np.hstack([s1, s2])
        row2 = np.hstack([s3, s4])
        return np.vstack([row1, row2])

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
# 6. WEB DASHBOARD TEMPLATE (HTML5 + CSS DARK THEME + MJPEG STREAMING)
# ------------------------------------------------------------------------------
HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>E.S.I.S - Multi-Camera Face Recognition (RunPod 4-Cam Node)</title>
    <style>
        :root {
            --bg-primary: #0b0f19;
            --bg-card: #151d30;
            --border-color: #24304f;
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
        
        .view-controls { display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }
        .btn-view { background: var(--bg-card); border: 1px solid var(--border-color); color: var(--text-muted); padding: 8px 16px; border-radius: 8px; font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
        .btn-view.active, .btn-view:hover { background: var(--accent-blue); color: #fff; border-color: var(--accent-blue); }

        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin-bottom: 20px; }
        .stat-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 14px; }
        .stat-card .label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; }
        .stat-card .val { font-size: 18px; font-weight: 700; color: #fff; margin-top: 6px; }

        .video-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin-bottom: 24px; }
        .video-grid.single { grid-template-columns: 1fr; }
        @media (max-width: 900px) {
            .video-grid { grid-template-columns: 1fr; }
        }
        .cam-box { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
        .cam-header { padding: 10px 14px; background: rgba(11, 15, 25, 0.7); display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); }
        .cam-header h3 { font-size: 13px; font-weight: 600; color: var(--text-main); }
        .feed-wrapper { width: 100%; aspect-ratio: 16 / 9; background: #000; position: relative; overflow: hidden; }
        .feed-wrapper img { width: 100%; height: 100%; object-fit: contain; display: block; }
        
        .log-section { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 18px; }
        .log-section h2 { font-size: 16px; font-weight: 600; margin-bottom: 12px; }
        .log-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .log-table th { text-align: left; padding: 10px; border-bottom: 1px solid var(--border-color); color: var(--text-muted); font-weight: 600; }
        .log-table td { padding: 10px; border-bottom: 1px solid rgba(36, 48, 79, 0.6); }
        .tag-match { color: var(--accent-green); font-weight: 600; }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo-title">
            <h1>E.S.I.S - 4-Camera Face Recognition Live Dashboard</h1>
            <p>Laboratorium Riset Lt 5, Sisdig, OS, & Programming | HLS Native Decoding + FaceNet ONNX</p>
        </div>
        <div class="badges-group">
            <span class="badge badge-amber" id="protocol-badge">4-CAM HLS ACTIVE</span>
            <span class="badge" id="ai-status">RUNPOD GPU NODE ACTIVE</span>
        </div>
    </div>

    <div class="view-controls">
        <button class="btn-view active" onclick="switchView('quad')">Grid 4 Kamera (2x2)</button>
        <button class="btn-view" onclick="switchView('mosaic')">Mosaic Split-Screen</button>
        <button class="btn-view" onclick="switchView('cam1')">Lab Riset Lt 5</button>
        <button class="btn-view" onclick="switchView('cam2')">Lab Sisdig</button>
        <button class="btn-view" onclick="switchView('cam3')">Lab OS</button>
        <button class="btn-view" onclick="switchView('cam4')">Lab Programming</button>
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
            <div class="label">Cam 1 (Lab Riset)</div>
            <div class="val" id="cam1-fps">-</div>
        </div>
        <div class="stat-card">
            <div class="label">Cam 2 (Lab Sisdig)</div>
            <div class="val" id="cam2-fps">-</div>
        </div>
        <div class="stat-card">
            <div class="label">Cam 3 (Lab OS)</div>
            <div class="val" id="cam3-fps">-</div>
        </div>
        <div class="stat-card">
            <div class="label">Cam 4 (Lab Programming)</div>
            <div class="val" id="cam4-fps">-</div>
        </div>
    </div>

    <div class="video-grid" id="main-video-grid">
        <div class="cam-box" id="box-cam1">
            <div class="cam-header">
                <h3>Kamera 1: Lab Riset Lantai 5 (H.265)</h3>
                <span class="badge">LIVE FEED</span>
            </div>
            <div class="feed-wrapper">
                <img id="img-cam1" src="/video_feed/1" onerror="handleStreamFallback(this, '/snapshot/1')" alt="Feed Lab Riset">
            </div>
        </div>
        <div class="cam-box" id="box-cam2">
            <div class="cam-header">
                <h3>Kamera 2: Lab Sistem Digital</h3>
                <span class="badge">LIVE FEED</span>
            </div>
            <div class="feed-wrapper">
                <img id="img-cam2" src="/video_feed/2" onerror="handleStreamFallback(this, '/snapshot/2')" alt="Feed Lab Sisdig">
            </div>
        </div>
        <div class="cam-box" id="box-cam3">
            <div class="cam-header">
                <h3>Kamera 3: Lab OS</h3>
                <span class="badge">LIVE FEED</span>
            </div>
            <div class="feed-wrapper">
                <img id="img-cam3" src="/video_feed/3" onerror="handleStreamFallback(this, '/snapshot/3')" alt="Feed Lab OS">
            </div>
        </div>
        <div class="cam-box" id="box-cam4">
            <div class="cam-header">
                <h3>Kamera 4: Lab Programming</h3>
                <span class="badge">LIVE FEED</span>
            </div>
            <div class="feed-wrapper">
                <img id="img-cam4" src="/video_feed/4" onerror="handleStreamFallback(this, '/snapshot/4')" alt="Feed Lab Programming">
            </div>
        </div>
        <div class="cam-box" id="box-mosaic" style="display: none;">
            <div class="cam-header">
                <h3>Mosaic 2x2 Multi-Camera Live</h3>
                <span class="badge">MOSAIC VIEW</span>
            </div>
            <div class="feed-wrapper">
                <img id="img-mosaic" src="/video_feed/mosaic" onerror="handleStreamFallback(this, '/snapshot/mosaic')" alt="Mosaic Feed">
            </div>
        </div>
    </div>

    <div class="log-section">
        <h2>Riwayat Wajah Teridentifikasi Real-Time</h2>
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
        function switchView(mode) {
            document.querySelectorAll('.btn-view').forEach(b => b.classList.remove('active'));
            if (event && event.target) { event.target.classList.add('active'); }

            const grid = document.getElementById('main-video-grid');
            const b1 = document.getElementById('box-cam1');
            const b2 = document.getElementById('box-cam2');
            const b3 = document.getElementById('box-cam3');
            const b4 = document.getElementById('box-cam4');
            const bm = document.getElementById('box-mosaic');

            const boxes = [b1, b2, b3, b4, bm];
            boxes.forEach(b => { if (b) b.style.display = 'none'; });

            if (mode === 'quad') {
                grid.className = 'video-grid';
                b1.style.display = 'block';
                b2.style.display = 'block';
                b3.style.display = 'block';
                b4.style.display = 'block';
            } else if (mode === 'mosaic') {
                grid.className = 'video-grid single';
                bm.style.display = 'block';
            } else if (mode === 'cam1') {
                grid.className = 'video-grid single';
                b1.style.display = 'block';
            } else if (mode === 'cam2') {
                grid.className = 'video-grid single';
                b2.style.display = 'block';
            } else if (mode === 'cam3') {
                grid.className = 'video-grid single';
                b3.style.display = 'block';
            } else if (mode === 'cam4') {
                grid.className = 'video-grid single';
                b4.style.display = 'block';
            }
        }

        function handleStreamFallback(imgElement, snapshotUrl) {
            console.warn("Stream fallback polling activated for", snapshotUrl);
            setInterval(() => {
                imgElement.src = snapshotUrl + "?t=" + Date.now();
            }, 100);
        }

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
                const c3 = data.cameras['Lab OS'];
                if (c3) {
                    document.getElementById('cam3-fps').textContent = `${c3.fps.toFixed(1)} FPS (${c3.latency_ms}ms)`;
                }
                const c4 = data.cameras['Lab Programming'];
                if (c4) {
                    document.getElementById('cam4-fps').textContent = `${c4.fps.toFixed(1)} FPS (${c4.latency_ms}ms)`;
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
            pass

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
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(json.dumps(pipeline.get_status_dict()).encode("utf-8"))
                return

            # Snapshot Endpoint
            if path in ("/snapshot/1", "/snapshot/2", "/snapshot/3", "/snapshot/4", "/snapshot/mosaic"):
                key = path.replace("/snapshot/", "")
                if key == "mosaic":
                    frame = pipeline.get_mosaic_frame()
                else:
                    frame = pipeline.get_annotated_frame(f"cam{key}")

                if frame is not None:
                    ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
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

            # MJPEG Stream Endpoint
            target_feed = None
            if path == "/video_feed/1":
                target_feed = "cam1"
            elif path == "/video_feed/2":
                target_feed = "cam2"
            elif path == "/video_feed/3":
                target_feed = "cam3"
            elif path == "/video_feed/4":
                target_feed = "cam4"
            elif path == "/video_feed/mosaic":
                target_feed = "mosaic"

            if target_feed:
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
                        if target_feed == "mosaic":
                            frame = pipeline.get_mosaic_frame()
                        else:
                            frame = pipeline.get_annotated_frame(target_feed)

                        if frame is not None:
                            ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
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

    return StreamHandler

# ------------------------------------------------------------------------------
# 7. MAIN RUNNER & CLI ARGUMENTS
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="E.S.I.S 4-Camera Face Recognition Node for RunPod / Local Deployment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--cam1", type=str, default="", help="URL stream Cam 1 (Lab Riset Lt 5)")
    parser.add_argument("--cam2", type=str, default="", help="URL stream Cam 2 (Lab Sisdig)")
    parser.add_argument("--cam3", type=str, default="", help="URL stream Cam 3 (Lab OS)")
    parser.add_argument("--cam4", type=str, default="", help="URL stream Cam 4 (Lab Programming)")
    parser.add_argument("--protocol", type=str, default="hls", choices=["auto", "webrtc", "hls", "rtsp"], help="Protokol video stream (default: hls)")
    parser.add_argument("--webrtc", action="store_true", help="Gunakan WebRTC WHEP direct dari stream.ucim.my.id")
    parser.add_argument("--pinggy", type=str, default=DEFAULT_HLS_BASE, help="Base URL Pinggy tunnel untuk HLS (.m3u8)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ONNX, help="Path model ONNX FaceNet")
    parser.add_argument("--database", type=str, default=DEFAULT_DATABASE_NPZ, help="Path database embedding .npz")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port Web Streaming HTTP Server")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Threshold Euclidean distance (0.7 - 0.8)")
    parser.add_argument("--detector", type=str, default="auto", choices=["auto", "mtcnn", "yunet", "haar"], help="Pilihan backend face detector")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Pilihan komputasi FaceNet")
    parser.add_argument("--mock", action="store_true", help="Jalankan dalam mode simulasi CCTV (dengan volunteer wajah) tanpa koneksi kamera fisik")
    parser.add_argument("--display", action="store_true", help="Buka jendela pop-up GUI monitor desktop lokal (cv2.imshow)")

    args = parser.parse_args()

    # Tentukan mode protokol
    active_proto = "hls"
    if args.webrtc:
        active_proto = "webrtc"
    elif args.protocol:
        active_proto = args.protocol.lower()

    # Tentukan default URL tiap kamera berdasarkan mode
    hls_base = args.pinggy.strip().rstrip("/")
    if active_proto == "webrtc":
        cam1_url = args.cam1 or f"{DEFAULT_WHEP_BASE}/{DEFAULT_CAM1_PATH}/whep"
        cam2_url = args.cam2 or f"{DEFAULT_WHEP_BASE}/{DEFAULT_CAM2_PATH}/whep"
        cam3_url = args.cam3 or f"{DEFAULT_WHEP_BASE}/{DEFAULT_CAM3_PATH}/whep"
        cam4_url = args.cam4 or f"{DEFAULT_WHEP_BASE}/{DEFAULT_CAM4_PATH}/whep"
    else:
        cam1_url = args.cam1 or f"{hls_base}/{DEFAULT_CAM1_PATH}/index.m3u8"
        cam2_url = args.cam2 or f"{hls_base}/{DEFAULT_CAM2_PATH}/index.m3u8"
        cam3_url = args.cam3 or f"{hls_base}/{DEFAULT_CAM3_PATH}/index.m3u8"
        cam4_url = args.cam4 or f"{hls_base}/{DEFAULT_CAM4_PATH}/index.m3u8"

    print("=" * 80)
    print(" [START] MEMULAI E.S.I.S 4-CAMERA FACE RECOGNITION NODE")
    print("=" * 80)
    print(f"  * Model FaceNet       : {args.model}")
    print(f"  * Database Mahasiswa  : {args.database}")
    print(f"  * Kamera 1 (Lab Riset): {cam1_url}")
    print(f"  * Kamera 2 (Lab Sisdig: {cam2_url}")
    print(f"  * Kamera 3 (Lab OS)   : {cam3_url}")
    print(f"  * Kamera 4 (Lab Prog) : {cam4_url}")
    print(f"  * Mode Protokol       : {active_proto.upper()}")
    print(f"  * Mode Simulasi Mock  : {'AKTIF (Menggunakan Foto Volunteer)' if args.mock else 'NONAKTIF (Streaming Langsung)'}")
    print(f"  * HTTP Web Dashboard  : Port {args.port}")
    print(f"  * Threshold Kemiripan : {args.threshold}")
    print(f"  * Local Display GUI   : {'AKTIF (cv2.imshow)' if args.display else 'NONAKTIF'}")
    print("=" * 80)

    detector = FaceDetector(mode=args.detector)

    recognizer = FaceRecognitionEngine(
        model_path=args.model,
        npz_database_path=args.database,
        threshold=args.threshold,
        device=args.device
    )

    cam1 = ThreadedCamera(name="Lab Riset Lt 5", source=cam1_url, protocol=active_proto, is_mock=args.mock).start()
    cam2 = ThreadedCamera(name="Lab Sisdig", source=cam2_url, protocol=active_proto, is_mock=args.mock).start()
    cam3 = ThreadedCamera(name="Lab OS", source=cam3_url, protocol=active_proto, is_mock=args.mock).start()
    cam4 = ThreadedCamera(name="Lab Programming", source=cam4_url, protocol=active_proto, is_mock=args.mock).start()

    cameras = {"cam1": cam1, "cam2": cam2, "cam3": cam3, "cam4": cam4}

    pipeline = PipelineManager(cameras=cameras, detector=detector, recognizer=recognizer).start()

    handler = make_stream_handler(pipeline)
    server = ThreadedHTTPServer(("0.0.0.0", args.port), handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print("\n[OK] SERVER STREAMING HTTP BERHASIL BERJALAN!")
    print(f"  -> Akses Dashboard Web : http://localhost:{args.port}/")
    print(f"  -> Stream Cam 1 (Riset): http://localhost:{args.port}/video_feed/1")
    print(f"  -> Stream Cam 2 (Sisdig): http://localhost:{args.port}/video_feed/2")
    print(f"  -> Stream Cam 3 (Lab OS): http://localhost:{args.port}/video_feed/3")
    print(f"  -> Stream Cam 4 (Prog)  : http://localhost:{args.port}/video_feed/4")
    print(f"  -> Stream Mosaic 2x2    : http://localhost:{args.port}/video_feed/mosaic")
    print(f"  -> Endpoint API Status  : http://localhost:{args.port}/api/status")
    print("-" * 80)
    if args.display:
        print("[INFO] Jendela monitor desktop lokal dibuka. Tekan 'q' pada jendela video untuk keluar.")
    else:
        print("[INFO] Buka tautan dashboard di atas di browser Anda.")
        print("Tekan [Ctrl + C] di terminal untuk menghentikan program.")
    print("-" * 80)

    try:
        while True:
            if args.display:
                m_frame = pipeline.get_mosaic_frame()
                if m_frame is not None:
                    cv2.imshow("E.S.I.S Face Recognition - 4 Camera Live Mosaic", m_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[WAIT] Menghentikan seluruh worker kamera & server...")
    finally:
        for c in cameras.values():
            c.stop()
        server.shutdown()
        cv2.destroyAllWindows()
        print("[OK] Program berhasil dihentikan secara bersih.")

if __name__ == "__main__":
    main()
