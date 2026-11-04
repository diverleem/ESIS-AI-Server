#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
E.S.I.S (Electronic Surveillance & Intelligent System)
Multi-Camera Face Recognition Node for RunPod & Local Deployment
================================================================================
Features:
- Dual Camera Support: Lab Sistem Digital & Lab OS
- Real-Time WebRTC (WHEP) Native Ingestion via aiortc & PyAV (Sub-second Latency!)
- Automatic Fallback to HLS / RTSP via OpenCV VideoCapture
- High-Performance Face Recognition:
    * Detection: MTCNN (primary) / YuNet (fast ONNX) / Haar Cascade (fallback)
    * Feature Extraction: FaceNet (ONNX Runtime with GPU CUDA acceleration)
    * Matching: L2-Normalized Euclidean Distance against data_wajah_facenet_mtcnn.npz
- Built-in Web Streaming Server (HTTP MJPEG & Snapshot API):
    * Modern dark dashboard accessible via browser or RunPod HTTP proxy port (8000)
    * Flush & X-Accel-Buffering optimized for Nginx/RunPod proxy (zero-stall streaming)
    * Individual camera feeds (/video_feed/1, /video_feed/2)
    * Combined mosaic view (/video_feed/mosaic)
    * Snapshot fallback API (/snapshot/1, /snapshot/2, /snapshot/mosaic)
    * Live status JSON API (/api/status)
- Desktop GUI support via OpenCV window (--display)
- Threaded non-blocking capture with auto-reconnection & mock fallback
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

# Default MediaMTX Streaming Server (Direct WebRTC WHEP / HLS)
DEFAULT_STREAM_BASE = os.environ.get("STREAM_BASE", "https://stream.ucim.my.id").rstrip("/")

# Kamera 1: Lab Sistem Digital (WHEP WebRTC)
DEFAULT_CAM1 = os.environ.get(
    "CAM_SISDIG",
    f"{DEFAULT_STREAM_BASE}/ecam_lab_sistem_digital/whep"
)

# Kamera 2: Lab OS (WHEP WebRTC)
DEFAULT_CAM2 = os.environ.get(
    "CAM_LAB_OS",
    f"{DEFAULT_STREAM_BASE}/ecam_lab_os/whep"
)

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
        self.detector = None
        self._init_backend()

    def _init_backend(self):
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

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        self.active_backend = "HaarCascade"
        print(" [FaceDetector] Backend aktif: OpenCV Haar Cascade (Fast Built-in)")

    def detect_faces(self, frame_bgr: np.ndarray, margin_percent: float = 0.10) -> List[Tuple[Tuple[int, int, int, int], np.ndarray, float]]:
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
            similarity = max(0.0, (1.0 - (min_dist / self.threshold))) * 100.0
        else:
            name = "Unknown"
            similarity = max(0.0, (1.0 - (min_dist / 2.0))) * 100.0

        return name, min_dist, similarity

# ------------------------------------------------------------------------------
# 4. WORKER THREAD KAMERA CCTV (WEBRTC WHEP / OPENCV DENGAN AUTO-RECONNECT & MOCK)
# ------------------------------------------------------------------------------
class ThreadedCamera:
    """
    Menangani streaming kamera video di thread terpisah.
    Mendukung WebRTC (WHEP) sub-second latency via aiortc & PyAV,
    serta OpenCV VideoCapture untuk RTSP / HLS dengan auto-reconnect.
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
        self.latest_frame: Optional[np.ndarray] = self._generate_mock_frame()
        self.is_running = False

        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

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

        return frame

    # ------------------ WebRTC (WHEP) Ingestion Loop ------------------
    def _webrtc_thread_entry(self):
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._webrtc_run())
        finally:
            loop.close()

    async def _webrtc_run(self):
        import asyncio
        retry_count = 0
        while self.is_running:
            try:
                await self._webrtc_session()
                retry_count = 0
            except Exception as e:
                retry_count += 1
                print(f" [{self.name}] [WebRTC Info] {e} (Mencoba ulang #{retry_count})...")
                self.is_connected = False

            if self.is_running:
                with self.lock:
                    self.latest_frame = self._generate_mock_frame()
                await asyncio.sleep(2.0)

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
            del_url = urllib.parse.urljoin(self.source, location)
            try:
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
                else:
                    if isinstance(src, str) and "/cam_" in src:
                        fallback_src = src.replace("/cam_", "/ecam_")
                        print(f" [{self.name}] Mainstream gagal, mencoba fallback: {fallback_src}")
                        self.cap.release()
                        self.cap = cv2.VideoCapture(fallback_src, cv2.CAP_FFMPEG)
                        if self.cap.isOpened():
                            ret2, frame2 = self.cap.read()
                            if ret2 and frame2 is not None:
                                print(f" [{self.name}] [OK] Terhubung via fallback: {frame2.shape[1]}x{frame2.shape[0]}")
                                self.source = fallback_src
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
    Mengorkestrasi inferensi pengenalan wajah pada multi-kamera
    serta menghasilkan output frame beranotasi untuk video streaming.
    """
    def __init__(self, cameras: Dict[str, ThreadedCamera], detector: FaceDetector, recognizer: FaceRecognitionEngine):
        self.cameras = cameras
        self.detector = detector
        self.recognizer = recognizer

        # Inisialisasi frame awal dengan mock frame agar tidak kosong
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
                color = (46, 204, 113)
                tag_text = f"{name} ({sim:.1f}%)"
            else:
                color = (41, 128, 250)
                tag_text = f"Unknown (d:{dist:.2f})"

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

            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
            label_y1 = max(0, y - th - 12)
            label_y2 = y
            cv2.rectangle(out, (x, label_y1), (x + tw + 16, label_y2), color, -1)
            cv2.putText(out, tag_text, (x + 8, label_y2 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

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
            for item in reversed(self.detections_history[-10:]):
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
        with self.lock:
            f1 = self.annotated_frames.get("cam1")
            f2 = self.annotated_frames.get("cam2")

        if f1 is None:
            f1 = self.cameras["cam1"].get_frame() if "cam1" in self.cameras else None
        if f2 is None:
            f2 = self.cameras["cam2"].get_frame() if "cam2" in self.cameras else None

        target_h, target_w = 360, 640
        def resize_std(img, label):
            if img is None:
                blank = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                cv2.putText(blank, f"NO SIGNAL ({label})", (160, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 2)
                return blank
            return cv2.resize(img, (target_w, target_h))

        s1 = resize_std(f1, "CAM 1")
        s2 = resize_std(f2, "CAM 2")
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
            --bg-primary: #0b0f19;
            --bg-card: #151d30;
            --border-color: #24304f;
            --accent-green: #10b981;
            --accent-blue: #3b82f6;
            --accent-cyan: #06b6d4;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
        body { background: var(--bg-primary); color: var(--text-main); min-height: 100vh; padding: 20px; }
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); padding-bottom: 16px; margin-bottom: 24px; flex-wrap: wrap; gap: 12px; }
        .logo-title h1 { font-size: 20px; font-weight: 700; color: #fff; letter-spacing: -0.5px; }
        .logo-title p { font-size: 13px; color: var(--text-muted); margin-top: 4px; }
        .badges-group { display: flex; gap: 8px; flex-wrap: wrap; }
        .badge { padding: 5px 12px; border-radius: 9999px; font-size: 12px; font-weight: 600; background: rgba(16, 185, 129, 0.15); color: var(--accent-green); border: 1px solid var(--accent-green); }
        .badge-cyan { background: rgba(6, 182, 212, 0.15); color: var(--accent-cyan); border: 1px solid var(--accent-cyan); }
        
        .view-controls { display: flex; gap: 10px; margin-bottom: 20px; }
        .btn-view { background: var(--bg-card); border: 1px solid var(--border-color); color: var(--text-muted); padding: 8px 16px; border-radius: 8px; font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
        .btn-view.active, .btn-view:hover { background: var(--accent-blue); color: #fff; border-color: var(--accent-blue); }

        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .stat-card { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 16px; }
        .stat-card .label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; }
        .stat-card .val { font-size: 20px; font-weight: 700; color: #fff; margin-top: 6px; }

        .video-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 20px; margin-bottom: 24px; }
        .video-grid.single { grid-template-columns: 1fr; }
        .cam-box { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
        .cam-header { padding: 12px 16px; background: rgba(11, 15, 25, 0.7); display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border-color); }
        .cam-header h3 { font-size: 14px; font-weight: 600; color: var(--text-main); }
        .feed-wrapper { width: 100%; aspect-ratio: 16 / 9; background: #000; position: relative; overflow: hidden; }
        .feed-wrapper img { width: 100%; height: 100%; object-fit: contain; display: block; }
        
        .log-section { background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; }
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
            <h1>E.S.I.S - Multi-Camera Face Recognition Live Dashboard</h1>
            <p>Laboratorium Sistem Digital & Laboratorium OS | WebRTC WHEP + FaceNet ONNX</p>
        </div>
        <div class="badges-group">
            <span class="badge badge-cyan">WEBRTC WHEP DIRECT</span>
            <span class="badge" id="ai-status">RUNPOD GPU NODE ACTIVE</span>
        </div>
    </div>

    <div class="view-controls">
        <button class="btn-view active" onclick="switchView('dual')">Dual Kamera (Grid)</button>
        <button class="btn-view" onclick="switchView('mosaic')">Mosaic Split-Screen</button>
        <button class="btn-view" onclick="switchView('cam1')">Lab Sisdig Saja</button>
        <button class="btn-view" onclick="switchView('cam2')">Lab OS Saja</button>
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
            <div class="label">Cam 1 (Lab Sisdig)</div>
            <div class="val" id="cam1-fps">-</div>
        </div>
        <div class="stat-card">
            <div class="label">Cam 2 (Lab OS)</div>
            <div class="val" id="cam2-fps">-</div>
        </div>
    </div>

    <div class="video-grid" id="main-video-grid">
        <div class="cam-box" id="box-cam1">
            <div class="cam-header">
                <h3>Kamera 1: Lab Sistem Digital</h3>
                <span class="badge">LIVE FEED</span>
            </div>
            <div class="feed-wrapper">
                <img id="img-cam1" src="/video_feed/1" onerror="handleStreamFallback(this, '/snapshot/1')" alt="Feed Lab Sisdig">
            </div>
        </div>
        <div class="cam-box" id="box-cam2">
            <div class="cam-header">
                <h3>Kamera 2: Lab OS</h3>
                <span class="badge">LIVE FEED</span>
            </div>
            <div class="feed-wrapper">
                <img id="img-cam2" src="/video_feed/2" onerror="handleStreamFallback(this, '/snapshot/2')" alt="Feed Lab OS">
            </div>
        </div>
        <div class="cam-box" id="box-mosaic" style="display: none;">
            <div class="cam-header">
                <h3>Mosaic Multi-Camera Live</h3>
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
            event.target.classList.add('active');

            const grid = document.getElementById('main-video-grid');
            const b1 = document.getElementById('box-cam1');
            const b2 = document.getElementById('box-cam2');
            const bm = document.getElementById('box-mosaic');

            if (mode === 'dual') {
                grid.className = 'video-grid';
                b1.style.display = 'block';
                b2.style.display = 'block';
                bm.style.display = 'none';
            } else if (mode === 'mosaic') {
                grid.className = 'video-grid single';
                b1.style.display = 'none';
                b2.style.display = 'none';
                bm.style.display = 'block';
            } else if (mode === 'cam1') {
                grid.className = 'video-grid single';
                b1.style.display = 'block';
                b2.style.display = 'none';
                bm.style.display = 'none';
            } else if (mode === 'cam2') {
                grid.className = 'video-grid single';
                b1.style.display = 'none';
                b2.style.display = 'block';
                bm.style.display = 'none';
            }
        }

        // Fallback otomatis ke snapshot polling jika browser menahan stream multipart
        function handleStreamFallback(imgElement, snapshotUrl) {
            console.warn("MJPEG streaming buffering/interrupted. Activating snapshot polling for", snapshotUrl);
            setInterval(() => {
                imgElement.src = snapshotUrl + "?t=" + Date.now();
            }, 80);
        }

        async function fetchStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                document.getElementById('engine-val').textContent = data.ai_provider;
                document.getElementById('detector-val').textContent = data.detector_backend;

                const c1 = data.cameras['Lab Sisdig'];
                if (c1) {
                    document.getElementById('cam1-fps').textContent = `${c1.fps.toFixed(1)} FPS (${c1.latency_ms}ms)`;
                }
                const c2 = data.cameras['Lab OS'];
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

            # Snapshot Endpoint (Single JPEG image per request)
            if path in ("/snapshot/1", "/snapshot/2", "/snapshot/mosaic"):
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
            elif path == "/video_feed/mosaic":
                target_feed = "mosaic"

            if target_feed:
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate, no-transform")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.send_header("X-Accel-Buffering", "no") # Penting untuk reverse proxy RunPod / Nginx
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
                                self.wfile.flush() # Kirim frame langsung tanpa buffering

                        time.sleep(0.035) # ~28 fps cap
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
        description="E.S.I.S Multi-Camera Face Recognition Node for RunPod / Local Deployment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--cam1", type=str, default=DEFAULT_CAM1, help="URL stream / WHEP kamera Lab Sisdig")
    parser.add_argument("--cam2", type=str, default=DEFAULT_CAM2, help="URL stream / WHEP kamera Lab OS")
    parser.add_argument("--protocol", type=str, default="auto", choices=["auto", "webrtc", "hls", "rtsp"], help="Protokol video stream")
    parser.add_argument("--pinggy", type=str, default="", help="Base URL Pinggy tunnel (contoh: https://fdxuqgcmfo.a.pinggy.link)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ONNX, help="Path model ONNX FaceNet")
    parser.add_argument("--database", type=str, default=DEFAULT_DATABASE_NPZ, help="Path database embedding .npz")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port Web Streaming HTTP Server")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Threshold Euclidean distance (0.7 - 0.8)")
    parser.add_argument("--detector", type=str, default="auto", choices=["auto", "mtcnn", "yunet", "haar"], help="Pilihan backend face detector")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Pilihan komputasi FaceNet")
    parser.add_argument("--mock", action="store_true", help="Jalankan dalam mode simulasi CCTV tanpa stream jaringan")
    parser.add_argument("--display", action="store_true", help="Buka jendela pop-up GUI monitor desktop lokal (cv2.imshow)")

    args = parser.parse_args()

    if args.pinggy:
        p_base = args.pinggy.strip().rstrip("/")
        args.cam1 = f"{p_base}/ecam_lab_sistem_digital/index.m3u8"
        args.cam2 = f"{p_base}/ecam_lab_os/index.m3u8"
        args.protocol = "hls"

    print("=" * 75)
    print("[START] MEMULAI E.S.I.S MULTI-CAMERA FACE RECOGNITION NODE")
    print("=" * 75)
    print(f"  * Model FaceNet   : {args.model}")
    print(f"  * Database NPZ    : {args.database}")
    print(f"  * Kamera 1 (Sisdig: {args.cam1}")
    print(f"  * Kamera 2 (Lab OS: {args.cam2}")
    print(f"  * Mode Protokol   : {args.protocol}")
    print(f"  * HTTP Web Port   : {args.port}")
    print(f"  * Threshold Jarak : {args.threshold}")
    print(f"  * Local Display   : {'AKTIF (Jendela Desktop)' if args.display else 'NONAKTIF (Streaming Web Port ' + str(args.port) + ')'}")
    print("=" * 75)

    detector = FaceDetector(mode=args.detector)

    recognizer = FaceRecognitionEngine(
        model_path=args.model,
        npz_database_path=args.database,
        threshold=args.threshold,
        device=args.device
    )

    cam1 = ThreadedCamera(name="Lab Sisdig", source=args.cam1, protocol=args.protocol, is_mock=args.mock).start()
    cam2 = ThreadedCamera(name="Lab OS", source=args.cam2, protocol=args.protocol, is_mock=args.mock).start()

    cameras = {"cam1": cam1, "cam2": cam2}

    pipeline = PipelineManager(cameras=cameras, detector=detector, recognizer=recognizer).start()

    handler = make_stream_handler(pipeline)
    server = ThreadedHTTPServer(("0.0.0.0", args.port), handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print("\n[OK] SERVER STREAMING HTTP BERHASIL BERJALAN!")
    print(f"  -> Akses Dashboard Web   : http://localhost:{args.port}/")
    print(f"  -> Stream Kamera 1 (Sisdig: http://localhost:{args.port}/video_feed/1")
    print(f"  -> Stream Kamera 2 (Lab OS: http://localhost:{args.port}/video_feed/2")
    print(f"  -> Stream Mosaic (Split)  : http://localhost:{args.port}/video_feed/mosaic")
    print(f"  -> Endpoint API Status    : http://localhost:{args.port}/api/status")
    print("-" * 75)
    if args.display:
        print("[INFO] Jendela monitor desktop lokal dibuka. Tekan 'q' pada jendela video untuk keluar.")
    else:
        print("[INFO] Untuk membuka di monitor lokal, jalankan dengan argumen: python main.py --display")
        print("Tekan [Ctrl + C] di terminal untuk menghentikan program.")
    print("-" * 75)

    try:
        while True:
            if args.display:
                m_frame = pipeline.get_mosaic_frame()
                if m_frame is not None:
                    cv2.imshow("E.S.I.S Face Recognition - Dual Camera Live", m_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            time.sleep(0.05)
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
