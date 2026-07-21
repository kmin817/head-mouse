#!/usr/bin/env python3
"""Move the macOS cursor with head movement, using only the Mac's built-in camera."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import queue
import sys
import time
import traceback
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import objc
import pyautogui
from AppKit import (
    NSBackingStoreBuffered,
    NSColor,
    NSFloatingWindowLevel,
    NSImage,
    NSImageScaleProportionallyUpOrDown,
    NSImageView,
    NSPanel,
    NSRunningApplication,
    NSScreen,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
)
from AVFoundation import (
    AVCaptureDevice,
    AVCaptureDeviceInput,
    AVCaptureDeviceTypeBuiltInWideAngleCamera,
    AVCaptureOutput,
    AVCaptureSession,
    AVCaptureVideoDataOutput,
    AVMediaTypeVideo,
)
from CoreMedia import CMSampleBufferGetImageBuffer
# CoreVideo bindings are exposed through PyObjC's Quartz package.
from Quartz import (
    CVPixelBufferGetBaseAddress,
    CVPixelBufferGetBytesPerRow,
    CVPixelBufferGetHeight,
    CVPixelBufferGetWidth,
    CVPixelBufferLockBaseAddress,
    CVPixelBufferUnlockBaseAddress,
    CGEventCreateKeyboardEvent,
    CGEventKeyboardSetUnicodeString,
    CGEventPost,
    kCGHIDEventTap,
    kCVPixelBufferPixelFormatTypeKey,
    kCVPixelFormatType_32BGRA,
)
from Foundation import NSData, NSObject
from PIL import Image, ImageDraw, ImageFont
import dispatch

try:
    from Vision import VNDetectFaceLandmarksRequest, VNImageRequestHandler
    VISION_IMPORT_ERROR = None
except ImportError as exc:
    VNDetectFaceLandmarksRequest = None
    VNImageRequestHandler = None
    VISION_IMPORT_ERROR = exc

class BuiltInCameraNotFound(RuntimeError):
    """Raised rather than silently opening a Continuity/iPhone camera."""


GUIDE_WIDTH = 480
GUIDE_HEIGHT = 390


def _guide_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a bundled macOS Korean font, with a safe Pillow fallback."""
    candidates = (
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/NotoSansGothic-Regular.ttf",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    )
    for font_path in candidates:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def create_motion_guide() -> np.ndarray:
    """Create the Korean quick-reference panel shown at the bottom-right."""
    # Transparent pixels outside the rounded card make the native borderless
    # panel look like a true overlay instead of a separate application window.
    canvas = Image.new("RGBA", (GUIDE_WIDTH, GUIDE_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    title_font = _guide_font(22)
    label_font = _guide_font(16)
    detail_font = _guide_font(15)
    footer_font = _guide_font(14)

    draw.rounded_rectangle((10, 10, GUIDE_WIDTH - 10, GUIDE_HEIGHT - 10),
                           radius=18, fill=(29, 33, 42, 238),
                           outline=(73, 84, 105, 245), width=2)
    draw.text((24, 20), "얼굴 모션 빠른 안내", font=title_font, fill=(244, 247, 252))
    draw.text((GUIDE_WIDTH - 137, 25), "HEAD MOUSE", font=footer_font,
              fill=(102, 198, 255))
    draw.line((24, 57, GUIDE_WIDTH - 24, 57), fill=(65, 74, 91), width=1)

    motions = (
        ("커서 이동", "고개를 원하는 방향으로 움직이기"),
        ("좌클릭", "양쪽 눈을 빠르게 3번 깜빡이기"),
        ("스크롤 모드", "입을 벌리고 0.7초 유지"),
        ("스크롤", "모드에서 고개를 위·아래로 움직이기"),
        ("드래그", "양쪽 눈썹을 0.5초 올려 켜기/끄기"),
        ("음성 입력", "입 다문 넓은 미소를 2초 유지"),
    )
    row_top = 68
    row_height = 42
    for index, (label, detail) in enumerate(motions):
        y = row_top + index * row_height
        if index % 2 == 0:
            draw.rounded_rectangle((20, y - 3, GUIDE_WIDTH - 20, y + 34),
                                   radius=8, fill=(35, 40, 50))
        draw.text((30, y + 4), label, font=label_font, fill=(104, 201, 255))
        draw.text((150, y + 5), detail, font=detail_font, fill=(225, 230, 239))

    footer_top = 326
    draw.rounded_rectangle((20, footer_top, GUIDE_WIDTH - 20, GUIDE_HEIGHT - 20),
                           radius=10, fill=(39, 47, 61))
    draw.text((31, footer_top + 9), "H  안내 숨기기 / 다시 표시",
              font=footer_font, fill=(255, 215, 110))
    draw.text((273, footer_top + 9), "C  보정   Q  종료",
              font=footer_font, fill=(190, 199, 214))
    draw.text((31, footer_top + 31), "H 키는 미리보기 창을 클릭한 뒤 누르세요.",
              font=footer_font, fill=(155, 166, 184))
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGBA2BGRA)


class MotionGuideOverlay:
    """Borderless, click-through macOS overlay for the gesture quick guide."""

    def __init__(self, guide_image: np.ndarray) -> None:
        ok, encoded = cv2.imencode(".png", guide_image)
        if not ok:
            raise RuntimeError("모션 안내 이미지를 만들지 못했습니다.")
        payload = encoded.tobytes()
        data = NSData.dataWithBytes_length_(payload, len(payload))
        self.image = NSImage.alloc().initWithData_(data)
        if self.image is None:
            raise RuntimeError("모션 안내 이미지를 macOS 패널로 변환하지 못했습니다.")

        screen = NSScreen.mainScreen()
        visible_frame = screen.visibleFrame()
        x = float(visible_frame.origin.x + visible_frame.size.width - GUIDE_WIDTH - 20)
        y = float(visible_frame.origin.y + 20)
        frame = ((x, y), (GUIDE_WIDTH, GUIDE_HEIGHT))
        style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
        self.panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            frame,
            style,
            NSBackingStoreBuffered,
            False,
        )
        self.panel.setOpaque_(False)
        self.panel.setBackgroundColor_(NSColor.clearColor())
        self.panel.setHasShadow_(True)
        self.panel.setLevel_(NSFloatingWindowLevel)
        self.panel.setHidesOnDeactivate_(False)
        self.panel.setIgnoresMouseEvents_(True)
        self.panel.setReleasedWhenClosed_(False)
        self.panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        self.image_view = NSImageView.alloc().initWithFrame_(
            ((0.0, 0.0), (GUIDE_WIDTH, GUIDE_HEIGHT))
        )
        self.image_view.setImage_(self.image)
        self.image_view.setImageScaling_(NSImageScaleProportionallyUpOrDown)
        self.panel.setContentView_(self.image_view)

    def show(self) -> None:
        self.panel.orderFrontRegardless()

    def hide(self) -> None:
        self.panel.orderOut_(None)

    def close(self) -> None:
        self.panel.orderOut_(None)
        self.panel.close()


def find_builtin_camera():
    """Return the Mac's built-in wide-angle camera and never a Continuity camera."""
    devices = AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo)
    built_in = [
        device for device in devices
        if str(device.deviceType()) == str(AVCaptureDeviceTypeBuiltInWideAngleCamera)
    ]
    if not built_in:
        names = ", ".join(str(d.localizedName()) for d in devices) or "없음"
        raise BuiltInCameraNotFound(
            "MacBook 내장 카메라를 찾지 못했습니다. 다른 카메라로 대체하지 않습니다. "
            f"감지된 비디오 장치: {names}"
        )
    return built_in[0]


class FrameReceiver(NSObject):
    """AVFoundation delegate that safely copies BGRA frames into a Python queue."""

    def initWithQueue_(self, frame_queue):
        # NSObject subclasses must use PyObjC's super bridge, not Python's super().
        self = objc.super(FrameReceiver, self).init()
        if self is None:
            return None
        self.frame_queue = frame_queue
        return self

    def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, sample_buffer, connection):
        try:
            pixel_buffer = CMSampleBufferGetImageBuffer(sample_buffer)
            if pixel_buffer is None:
                return
            CVPixelBufferLockBaseAddress(pixel_buffer, 0)
            try:
                width = CVPixelBufferGetWidth(pixel_buffer)
                height = CVPixelBufferGetHeight(pixel_buffer)
                stride = CVPixelBufferGetBytesPerRow(pixel_buffer)
                address = CVPixelBufferGetBaseAddress(pixel_buffer)
                # PyObjC returns a varlist for void*. Its as_buffer method is
                # the supported way to access a known byte length.
                raw = np.frombuffer(address.as_buffer(height * stride), dtype=np.uint8)
                frame = raw.reshape((height, stride))[:, : width * 4].reshape((height, width, 4)).copy()
            finally:
                CVPixelBufferUnlockBaseAddress(pixel_buffer, 0)
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass
        except Exception as exc:
            # Exceptions escaping an Objective-C callback abort the whole app.
            # Keep the capture session alive and make the cause visible instead.
            print(f"카메라 프레임 변환 오류: {exc}", file=sys.stderr)


class BuiltInCamera:
    def __init__(self) -> None:
        device = find_builtin_camera()
        self.name = str(device.localizedName())
        self.frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self.session = AVCaptureSession.alloc().init()
        error = None
        camera_input, error = AVCaptureDeviceInput.deviceInputWithDevice_error_(device, error)
        if camera_input is None:
            raise RuntimeError(f"내장 카메라를 열 수 없습니다: {error}")
        self.session.addInput_(camera_input)
        self.output = AVCaptureVideoDataOutput.alloc().init()
        self.output.setAlwaysDiscardsLateVideoFrames_(True)
        self.output.setVideoSettings_({kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA})
        self.delegate = FrameReceiver.alloc().initWithQueue_(self.frames)
        # A queue label is optional. Passing a Python str here trips an
        # incompatible char conversion in some PyObjC/libdispatch releases.
        self.callback_queue = dispatch.dispatch_queue_create(None, None)
        self.output.setSampleBufferDelegate_queue_(self.delegate, self.callback_queue)
        self.session.addOutput_(self.output)

    def start(self) -> None:
        self.session.startRunning()

    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        try:
            return self.frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        self.session.stopRunning()


@dataclass
class FaceGestureFrame:
    face_center: np.ndarray
    control_point: np.ndarray
    bounding_box: tuple[float, float, float, float]
    left_eye: float
    right_eye: float
    mouth_roundness: float
    mouth_opening: float
    mouth_width: float
    left_brow_gap: float
    right_brow_gap: float


def _region_points(region) -> Optional[np.ndarray]:
    if region is None or region.pointCount() == 0:
        return None
    points = region.normalizedPoints().as_tuple(region.pointCount())
    return np.array([
        (float(point.x), float(point.y)) if hasattr(point, "x") else (float(point[0]), float(point[1]))
        for point in points
    ])


def _shape_ratio(points: np.ndarray) -> float:
    width = float(np.ptp(points[:, 0]))
    height = float(np.ptp(points[:, 1]))
    return height / max(width, 1e-6)


class VisionGestureAnalyzer:
    """Extract face geometry with Apple's built-in Vision framework."""

    def __init__(self) -> None:
        if VISION_IMPORT_ERROR is not None:
            raise RuntimeError(
                "Vision 바인딩이 없습니다. "
                "python3 -m pip install pyobjc-framework-Vision 을 실행하세요."
            ) from VISION_IMPORT_ERROR
        self.request = VNDetectFaceLandmarksRequest.alloc().init()

    def analyze(self, image: np.ndarray) -> Optional[FaceGestureFrame]:
        if image.shape[1] > 640:
            scale = 640.0 / image.shape[1]
            image = cv2.resize(
                image,
                (640, max(1, int(image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            return None
        payload = encoded.tobytes()
        data = NSData.dataWithBytes_length_(payload, len(payload))
        handler = VNImageRequestHandler.alloc().initWithData_options_(data, {})
        success, error = handler.performRequests_error_([self.request], None)
        if not success:
            raise RuntimeError(f"Vision 얼굴 분석 실패: {error}")
        results = self.request.results() or []
        observations = [item for item in results if item.landmarks() is not None]
        if not observations:
            return None
        observation = max(
            observations,
            key=lambda item: float(item.boundingBox().size.width * item.boundingBox().size.height),
        )
        landmarks = observation.landmarks()
        left_eye = _region_points(landmarks.leftEye())
        right_eye = _region_points(landmarks.rightEye())
        outer_lips = _region_points(landmarks.outerLips())
        inner_lips = _region_points(landmarks.innerLips())
        nose = _region_points(landmarks.nose())
        left_brow = _region_points(landmarks.leftEyebrow())
        right_brow = _region_points(landmarks.rightEyebrow())
        if any(region is None for region in (
            left_eye, right_eye, outer_lips, inner_lips, nose, left_brow, right_brow
        )):
            return None

        box = observation.boundingBox()
        # Vision coordinates originate at bottom-left. The preview is mirrored,
        # so x is inverted for cursor control and preview drawing.
        center = np.array([
            1.0 - float(box.origin.x + box.size.width / 2),
            1.0 - float(box.origin.y + box.size.height / 2),
        ])
        # Face-box center mainly reacts to translation. The nose position
        # inside the face reacts to yaw/pitch, so combining both lets a smaller
        # head rotation control the cursor without increasing raw sensitivity.
        nose_local = np.array([
            1.0 - float(nose[:, 0].mean()),
            1.0 - float(nose[:, 1].mean()),
        ])
        control_point = np.clip(
            center + np.array([0.35, 0.25]) * (nose_local - 0.5),
            0.0,
            1.0,
        )
        mirrored_box = (
            1.0 - float(box.origin.x + box.size.width),
            1.0 - float(box.origin.y + box.size.height),
            float(box.size.width),
            float(box.size.height),
        )
        return FaceGestureFrame(
            face_center=center,
            control_point=control_point,
            bounding_box=mirrored_box,
            left_eye=_shape_ratio(left_eye),
            right_eye=_shape_ratio(right_eye),
            mouth_roundness=_shape_ratio(outer_lips),
            mouth_opening=float(np.ptp(inner_lips[:, 1])),
            mouth_width=float(np.ptp(outer_lips[:, 0])),
            left_brow_gap=float(left_brow[:, 1].mean() - left_eye[:, 1].mean()),
            right_brow_gap=float(right_brow[:, 1].mean() - right_eye[:, 1].mean()),
        )


class TripleQuickBlinkGesture:
    """Recognize three quick two-eye blinks as a left click gesture."""

    def __init__(self, minimum: float = 0.05, maximum: float = 0.50,
                 sequence_timeout: float = 1.00) -> None:
        self.minimum = minimum
        self.maximum = maximum
        self.sequence_timeout = sequence_timeout
        self.closed_since: Optional[float] = None
        self.last_blink_at: Optional[float] = None
        self.count = 0

    def reset(self) -> None:
        self.closed_since = None
        self.last_blink_at = None
        self.count = 0

    def update(self, both_closed: bool, both_open: bool, now: float) -> bool:
        # Only expire while waiting with the eyes open; never reset a blink
        # that is already in progress.
        if self.count and self.last_blink_at is not None and self.closed_since is None:
            if not both_closed and now - self.last_blink_at > self.sequence_timeout:
                print(f"빠른 눈 깜빡임 대기 시간 초과 ({self.sequence_timeout:.1f}초) - 초기화")
                self.reset()
        if both_closed and self.closed_since is None:
            self.closed_since = now
            return False
        # A hysteresis band between closed and open prevents landmark noise
        # from splitting one long blink into multiple short detections.
        if both_open and self.closed_since is not None:
            duration = now - self.closed_since
            self.closed_since = None
            if self.minimum <= duration <= self.maximum:
                self.count += 1
                self.last_blink_at = now
                print(f"빠른 눈 깜빡임 감지: {self.count}/3 ({duration:.2f}초)")
                if self.count >= 3:
                    self.reset()
                    return True
            elif duration > self.maximum:
                print(f"빠른 깜빡임 기준보다 오래 감음 ({duration:.2f}초) - 초기화")
                self.reset()
            elif duration >= 0.12:
                print(f"눈 감은 시간이 너무 짧음 ({duration:.2f}초) - 무시")
        return False


class HoldGesture:
    def __init__(self, duration: float) -> None:
        self.duration = duration
        self.started_at: Optional[float] = None
        self.fired = False

    def update(self, active: bool, now: float) -> bool:
        if not active:
            self.started_at = None
            self.fired = False
            return False
        if self.started_at is None:
            self.started_at = now
        if not self.fired and now - self.started_at >= self.duration:
            self.fired = True
            return True
        return False


def type_unicode_text(text: str) -> None:
    """Type Unicode into the currently focused macOS control without using the clipboard."""
    # Quartz keyboard events have a small practical payload limit, so send
    # short chunks. Korean syllables are preserved without depending on the
    # currently selected keyboard layout.
    for start in range(0, len(text), 20):
        chunk = text[start:start + 20]
        utf16_length = len(chunk.encode("utf-16-le")) // 2
        key_down = CGEventCreateKeyboardEvent(None, 0, True)
        CGEventKeyboardSetUnicodeString(key_down, utf16_length, chunk)
        CGEventPost(kCGHIDEventTap, key_down)
        key_up = CGEventCreateKeyboardEvent(None, 0, False)
        CGEventKeyboardSetUnicodeString(key_up, utf16_length, chunk)
        CGEventPost(kCGHIDEventTap, key_up)


class VoiceDictation:
    """Run the signed macOS voice helper without stealing browser focus."""

    def __init__(self, locale_identifier: str) -> None:
        self.locale_identifier = locale_identifier
        self.app = (
            Path(__file__).resolve().parent
            / "VoiceHelper"
            / "HeadMouseVoice.app"
        )
        self.helper = self.app / "Contents" / "MacOS" / "HeadMouseVoice"
        if not self.helper.is_file():
            raise RuntimeError(
                "음성 인식 도우미가 없습니다. README의 VoiceHelper 빌드 명령을 실행하세요."
            )
        if not self.helper.stat().st_mode & 0o111:
            raise RuntimeError(f"음성 인식 도우미 실행 권한이 없습니다: {self.helper}")
        self.listening = False
        self.process: Optional[subprocess.Popen[str]] = None
        self.output_path: Optional[Path] = None
        self.authorization_checked = False
        self.authorization_error: Optional[str] = None
        # Ask for permissions at startup, before the user focuses a search box.
        self.authorization_process, self.authorization_output_path = self._launch(
            ["--authorize-only"]
        )

    def _launch(self, arguments: list[str]) -> tuple[subprocess.Popen[str], Path]:
        descriptor, output_name = tempfile.mkstemp(prefix="head-mouse-voice-", suffix=".json")
        os.close(descriptor)
        output_path = Path(output_name)
        output_path.unlink(missing_ok=True)
        process = subprocess.Popen(
            ["open", "-n", "-W", str(self.app), "--args", *arguments,
             "--output", str(output_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        return process, output_path

    @staticmethod
    def _read_result(process: subprocess.Popen[str], output_path: Path) -> tuple[str, Optional[str]]:
        _, launcher_stderr = process.communicate()
        text = ""
        error: Optional[str] = None
        if output_path.is_file():
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
                text = str(payload.get("text") or "").strip()
                error = str(payload.get("error") or "").strip() or None
            except (OSError, ValueError, TypeError) as exc:
                error = f"음성 인식 결과 파일 오류: {exc}"
            finally:
                output_path.unlink(missing_ok=True)
        elif process.returncode != 0:
            error = (launcher_stderr or "").strip() or "음성 인식 앱을 실행하지 못했습니다."
        return text, error

    def permission_message(self) -> Optional[str]:
        if self.authorization_process.poll() is None:
            return "Head Mouse Voice의 마이크 및 음성 인식 권한을 허용하세요."
        if not self.authorization_checked:
            _, self.authorization_error = self._read_result(
                self.authorization_process, self.authorization_output_path
            )
            self.authorization_checked = True
        return self.authorization_error

    def start(self) -> tuple[bool, str]:
        if self.listening:
            return False, "이미 음성을 듣고 있습니다."
        if self.authorization_process.poll() is None:
            return False, "마이크 및 음성 인식 권한 승인을 기다리는 중입니다."
        permission_error = self.permission_message()
        if permission_error:
            # Retry the authorization helper so permission changes in System
            # Settings are picked up without restarting the Python process.
            self.authorization_process, self.authorization_output_path = self._launch(
                ["--authorize-only"]
            )
            self.authorization_checked = False
            self.authorization_error = None
            return False, permission_error
        self.process, self.output_path = self._launch(
            ["--language", self.locale_identifier]
        )
        self.listening = True
        print("음성 입력 시작: 말씀하세요.")
        return True, "LISTENING"

    def poll(self, now: float) -> Optional[tuple[str, Optional[str]]]:
        if not self.listening or self.process is None or self.process.poll() is None:
            return None
        assert self.output_path is not None
        text, error = self._read_result(self.process, self.output_path)
        self.process = None
        self.output_path = None
        self.listening = False
        return text, error

    def cancel(self) -> None:
        if self.process is not None and self.process.poll() is None:
            for application in NSRunningApplication.runningApplicationsWithBundleIdentifier_(
                "com.igyeongmin.headmouse.voice"
            ):
                application.terminate()
            self.process.terminate()
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1.0)
        self.process = None
        if self.output_path is not None:
            self.output_path.unlink(missing_ok=True)
        self.output_path = None
        self.listening = False


class GestureController:
    CALIBRATION_FRAMES = 24

    def __init__(self) -> None:
        self.triple_quick_blink = TripleQuickBlinkGesture()
        self.mouth_hold = HoldGesture(0.70)
        self.brow_hold = HoldGesture(0.50)
        self.smile_hold = HoldGesture(2.00)
        self.samples: list[FaceGestureFrame] = []
        self.baseline: Optional[np.ndarray] = None
        self.scroll_mode = False
        self.drag_active = False
        self.scroll_anchor_y: Optional[float] = None
        self.scroll_remainder = 0.0
        self.last_scroll_time: Optional[float] = None
        self.face_missing_since: Optional[float] = None
        self.voice_cooldown_until = 0.0

    def calibrate(self) -> None:
        self.release_drag("재보정")
        self.scroll_mode = False
        self.scroll_anchor_y = None
        self.scroll_remainder = 0.0
        self.voice_cooldown_until = 0.0
        self.triple_quick_blink.reset()
        self.mouth_hold = HoldGesture(0.70)
        self.brow_hold = HoldGesture(0.50)
        self.smile_hold = HoldGesture(2.00)
        self.samples.clear()
        self.baseline = None

    @property
    def calibration_progress(self) -> int:
        return min(len(self.samples), self.CALIBRATION_FRAMES)

    def _learn_baseline(self, frame: FaceGestureFrame) -> bool:
        if self.baseline is not None:
            return True
        self.samples.append(frame)
        if len(self.samples) < self.CALIBRATION_FRAMES:
            return False
        values = np.array([
            [sample.left_eye, sample.right_eye, sample.mouth_roundness,
             sample.mouth_opening, sample.mouth_width, sample.left_brow_gap,
             sample.right_brow_gap]
            for sample in self.samples
        ])
        self.baseline = np.median(values, axis=0)
        print("얼굴 제스처 보정 완료")
        return True

    def release_drag(self, reason: Optional[str] = None) -> None:
        if self.drag_active:
            pyautogui.mouseUp(button="left", _pause=False)
            self.drag_active = False
            if reason:
                print(f"드래그 해제: {reason}")

    def face_missing(self, now: float) -> None:
        if self.face_missing_since is None:
            self.face_missing_since = now
        elif now - self.face_missing_since >= 1.0:
            self.release_drag("얼굴 인식 중단")

    def voice_finished(self, now: float) -> None:
        self.voice_cooldown_until = now + 1.5
        self.triple_quick_blink.reset()
        self.mouth_hold = HoldGesture(0.70)
        self.brow_hold = HoldGesture(0.50)
        self.smile_hold = HoldGesture(2.00)

    def update(self, frame: FaceGestureFrame, now: float) -> list[str]:
        self.face_missing_since = None
        if not self._learn_baseline(frame):
            return []
        (left_eye_base, right_eye_base, mouth_base, mouth_opening_base,
         mouth_width_base, left_brow_base, right_brow_base) = self.baseline
        left_eye_level = frame.left_eye / max(left_eye_base, 1e-6)
        right_eye_level = frame.right_eye / max(right_eye_base, 1e-6)
        average_eye_level = (left_eye_level + right_eye_level) / 2.0
        # Requiring both eyes to be somewhat closed prevents a wink from
        # counting, while the average makes the detector tolerant of one eye's
        # noisier Vision landmarks. A higher open threshold adds hysteresis.
        both_closed = (
            max(left_eye_level, right_eye_level) < 0.78
            and average_eye_level < 0.65
        )
        both_open = min(left_eye_level, right_eye_level) > 0.82
        wide_smile = (
            frame.mouth_width > mouth_width_base * 1.12
            and frame.mouth_roundness < max(0.42, mouth_base * 1.30)
        )
        # The inner-lip vertical gap detects an ordinarily opened mouth without
        # requiring the lips to form a round O. Excluding a wide smile keeps the
        # voice gesture separate from the scroll-mode gesture.
        mouth_open = (
            frame.mouth_opening > max(
                mouth_opening_base * 1.65,
                mouth_opening_base + 0.035,
            )
            and not wide_smile
        )
        brows_up = (
            frame.left_brow_gap > left_brow_base + 0.035
            and frame.right_brow_gap > right_brow_base + 0.035
        )
        events: list[str] = []

        if now >= self.voice_cooldown_until and self.smile_hold.update(wide_smile, now):
            self.scroll_mode = False
            self.scroll_anchor_y = None
            self.release_drag("음성 입력 시작")
            self.triple_quick_blink.reset()
            self.mouth_hold = HoldGesture(0.70)
            self.brow_hold = HoldGesture(0.50)
            events.append("VOICE START")
            return events

        if self.mouth_hold.update(mouth_open, now):
            self.scroll_mode = not self.scroll_mode
            self.scroll_anchor_y = frame.face_center[1]
            self.scroll_remainder = 0.0
            self.last_scroll_time = now
            if self.scroll_mode:
                self.release_drag("스크롤 모드 활성화")
            events.append("SCROLL ON" if self.scroll_mode else "SCROLL OFF")

        if not self.scroll_mode and self.brow_hold.update(brows_up, now):
            if self.drag_active:
                self.release_drag()
                events.append("DRAG OFF")
            else:
                pyautogui.mouseDown(button="left", _pause=False)
                self.drag_active = True
                events.append("DRAG ON")
        elif self.scroll_mode:
            self.brow_hold.update(False, now)

        if not self.scroll_mode and not self.drag_active:
            if self.triple_quick_blink.update(both_closed, both_open, now):
                pyautogui.click(button="left", _pause=False)
                events.append("LEFT CLICK")
        else:
            self.triple_quick_blink.reset()

        if self.scroll_mode and self.scroll_anchor_y is not None:
            dt = min(max(now - (self.last_scroll_time or now), 0.0), 0.10)
            self.last_scroll_time = now
            offset = self.scroll_anchor_y - frame.face_center[1]
            deadzone = 0.025
            if abs(offset) > deadzone:
                intensity = min((abs(offset) - deadzone) / 0.12, 1.0)
                ticks_per_second = 4.0 + 22.0 * intensity ** 1.5
                self.scroll_remainder += np.sign(offset) * ticks_per_second * dt
                ticks = int(self.scroll_remainder)
                if ticks:
                    pyautogui.scroll(ticks, _pause=False)
                    self.scroll_remainder -= ticks
        return events


@dataclass
class CursorController:
    screen_width: int
    screen_height: int
    sensitivity: float
    smoothing: float
    deadzone: float = 0.012
    max_offset: float = 0.12
    center: Optional[np.ndarray] = None
    cursor: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None
    last_time: Optional[float] = None
    motion_started_at: Optional[np.ndarray] = None
    motion_direction: Optional[np.ndarray] = None
    precision_delay: float = 0.25
    acceleration_time: float = 0.75

    def calibrate(self, tracking_point: np.ndarray) -> None:
        self.center = tracking_point.copy()
        self.cursor = np.array(pyautogui.position(), dtype=float)
        self.velocity = np.zeros(2, dtype=float)
        self.motion_started_at = np.zeros(2, dtype=float)
        self.motion_direction = np.zeros(2, dtype=float)
        self.last_time = time.monotonic()

    def stop(self) -> None:
        if self.velocity is not None:
            self.velocity.fill(0.0)
        if self.motion_started_at is not None:
            self.motion_started_at.fill(0.0)
        if self.motion_direction is not None:
            self.motion_direction.fill(0.0)
        self.last_time = time.monotonic()

    def move(self, tracking_point: np.ndarray) -> None:
        if self.center is None:
            self.calibrate(tracking_point)
            return
        now = time.monotonic()
        dt = min(max(now - (self.last_time or now), 0.001), 0.10)
        self.last_time = now
        delta = tracking_point - self.center

        # Relative control: webcam/face-detector jitter around center falls in
        # the deadzone. Outside it, a small head turn produces continuous motion.

        # X축과 Y축의 신체 가동 범위와 카메라 화면 비율이 다르므로 Y축(상하)을 더 민감하게 설정합니다.
        # Y축은 데드존을 20% 줄이고, 최대 속도 도달 범위를 40% 줄여 더 적은 고개 움직임으로도 커서가 잘 이동하게 합니다.
        effective_deadzone = np.array([self.deadzone, self.deadzone * 0.8])
        effective_max_offset = np.array([self.max_offset, self.max_offset * 0.6])

        distance = np.abs(delta)
        outside_deadzone = np.maximum(distance - effective_deadzone, 0.0)
        normalized = np.clip(outside_deadzone / (effective_max_offset - effective_deadzone), 0.0, 1.0)
        direction = np.sign(delta)
        screen_size = np.array([self.screen_width, self.screen_height], dtype=float)
        precision_speed = screen_size * 0.025
        maximum_speed = screen_size * (0.33 * self.sensitivity)
        target_velocity = np.zeros(2, dtype=float)

        if self.velocity is None:
            self.velocity = np.zeros(2, dtype=float)
        if self.cursor is None:
            self.cursor = np.array(pyautogui.position(), dtype=float)
        if self.motion_started_at is None:
            self.motion_started_at = np.zeros(2, dtype=float)
        if self.motion_direction is None:
            self.motion_direction = np.zeros(2, dtype=float)

        # Start slowly for precise placement. Holding the same direction first
        # waits briefly, then smoothly accelerates so a small head tilt can
        # still cross the entire screen. Each axis resets independently when it
        # returns to the deadzone or reverses direction.
        for axis in range(2):
            if outside_deadzone[axis] == 0:
                self.velocity[axis] = 0.0
                self.motion_direction[axis] = 0.0
                self.motion_started_at[axis] = 0.0
                continue

            if self.motion_direction[axis] != direction[axis]:
                self.velocity[axis] = 0.0
                self.motion_direction[axis] = direction[axis]
                self.motion_started_at[axis] = now

            held_for = now - self.motion_started_at[axis]
            ramp = np.clip(
                (held_for - self.precision_delay) / self.acceleration_time,
                0.0,
                1.0,
            )
            # Smoothstep avoids a noticeable speed jump when acceleration begins.
            ramp = ramp * ramp * (3.0 - 2.0 * ramp)
            sustained_speed = maximum_speed[axis] * (
                0.35 + 0.65 * normalized[axis] ** 2
            )
            sustained_speed = max(sustained_speed, precision_speed[axis])
            speed = precision_speed[axis] + (
                sustained_speed - precision_speed[axis]
            ) * ramp
            target_velocity[axis] = direction[axis] * speed
            self.velocity[axis] += self.smoothing * (
                target_velocity[axis] - self.velocity[axis]
            )

        self.cursor += self.velocity * dt
        self.cursor[0] = np.clip(self.cursor[0], 0, self.screen_width - 1)
        self.cursor[1] = np.clip(self.cursor[1], 0, self.screen_height - 1)
        pyautogui.moveTo(int(self.cursor[0]), int(self.cursor[1]), _pause=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MacBook 내장 카메라 전용 머리 마우스")
    parser.add_argument("--sensitivity", type=float, default=1.8,
                        help="커서 최대 이동 속도 배율 (기본값: 1.8)")
    parser.add_argument("--smoothing", type=float, default=0.75,
                        help="0~1 사이의 속도 반응값; 높을수록 빠르게 반응 (기본값: 0.75)")
    parser.add_argument("--language", default="ko-KR",
                        help="음성 인식 언어 코드 (기본값: ko-KR)")
    parser.add_argument("--hide-guide", action="store_true",
                        help="시작할 때 우측 하단 모션 안내를 숨김")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sensitivity <= 0 or not 0 < args.smoothing <= 1:
        print("--sensitivity는 양수이고 --smoothing은 0~1 사이여야 합니다.", file=sys.stderr)
        return 2

    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = False
    try:
        camera = BuiltInCamera()
    except BuiltInCameraNotFound as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"카메라 초기화 오류: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    screen_width, screen_height = pyautogui.size()
    controller = CursorController(screen_width, screen_height, args.sensitivity, args.smoothing)
    try:
        gesture_analyzer = VisionGestureAnalyzer()
    except RuntimeError as exc:
        print(f"제스처 초기화 오류: {exc}", file=sys.stderr)
        camera.close()
        return 1
    try:
        voice = VoiceDictation(args.language)
    except RuntimeError as exc:
        print(f"음성 입력 초기화 오류: {exc}", file=sys.stderr)
        camera.close()
        return 1
    gestures = GestureController()
    event_text = ""
    event_display_until = 0.0
    last_gesture_frame: Optional[FaceGestureFrame] = None
    preview_title = "Head Mouse - C: calibrate, H: guide, Q: quit"
    preview_width, preview_height = 320, 240
    cv2.namedWindow(preview_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(preview_title, preview_width, preview_height)
    # OpenCV uses the same top-left screen coordinate convention as pyautogui.
    cv2.moveWindow(preview_title, 20, max(20, screen_height - preview_height - 70))
    guide_overlay = MotionGuideOverlay(create_motion_guide())
    guide_visible = not args.hide_guide
    if guide_visible:
        guide_overlay.show()
    print(f"사용 카메라: {camera.name} (MacBook 내장 카메라 전용)")
    print("양쪽 눈을 빠르게 세 번 깜빡임: 좌클릭")
    print("입 벌리기 0.7초: 스크롤 모드 | 양쪽 눈썹 0.5초: 드래그 토글")
    print("넓게 미소 짓기 2초: 음성 입력 시작")
    permission_message = voice.permission_message()
    if permission_message:
        print(f"음성 입력 권한: {permission_message}")
    print("H: 모션 안내 켜기/끄기 | C: 보정 | Q 또는 Esc: 종료")
    camera.start()
    try:
        while True:
            frame = camera.read()
            if frame is None:
                print("카메라 프레임을 받지 못했습니다. 카메라 권한을 확인하세요.", file=sys.stderr)
                break
            raw_image = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            gesture_frame = gesture_analyzer.analyze(raw_image)
            image = cv2.flip(raw_image, 1)
            now = time.monotonic()
            dictation_result = voice.poll(now)
            if dictation_result is not None:
                recognized_text, recognition_error = dictation_result
                gestures.voice_finished(now)
                if recognized_text:
                    type_unicode_text(recognized_text)
                    print(f"음성 입력 완료: {recognized_text}")
                    event_text = "DICTATION DONE"
                else:
                    print(f"음성 입력 실패: {recognition_error}", file=sys.stderr)
                    event_text = "VOICE ERROR"
                event_display_until = now + 1.5
            if gesture_frame is not None:
                last_gesture_frame = gesture_frame
                was_scroll_mode = gestures.scroll_mode
                if voice.listening:
                    # Dictation takes exclusive control of facial gestures.
                    # Speaking can open the mouth, so keep scroll mode off and
                    # do not run any click, drag, or scroll gesture detectors.
                    gestures.scroll_mode = False
                    gestures.scroll_anchor_y = None
                    events = []
                else:
                    events = gestures.update(gesture_frame, now)
                if "VOICE START" in events:
                    started, message = voice.start()
                    if started:
                        events = ["LISTENING"]
                    else:
                        print(f"음성 입력 시작 실패: {message}", file=sys.stderr)
                        gestures.voice_finished(now)
                        events = ["VOICE ERROR"]
                if voice.listening:
                    controller.stop()
                elif gestures.scroll_mode:
                    controller.stop()
                else:
                    if was_scroll_mode:
                        controller.calibrate(gesture_frame.control_point)
                    controller.move(gesture_frame.control_point)
                if events:
                    event_text = events[-1]
                    event_display_until = now + 1.2
                    print(" | ".join(events))

                h, w = image.shape[:2]
                bx, by, bw, bh = gesture_frame.bounding_box
                x, y = int(bx * w), int(by * h)
                width, height = int(bw * w), int(bh * h)
                mode_color = (0, 0, 255) if gestures.drag_active else (
                    (0, 165, 255) if gestures.scroll_mode else (0, 255, 0)
                )
                cv2.rectangle(image, (x, y), (x + width, y + height), mode_color, 2)
                cv2.circle(
                    image,
                    (int(gesture_frame.control_point[0] * w), int(gesture_frame.control_point[1] * h)),
                    6,
                    mode_color,
                    -1,
                )
                cv2.putText(image, "Tracking", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, mode_color, 2)
                if gestures.baseline is None:
                    progress = gestures.calibration_progress
                    cv2.putText(image, f"Calibrating {progress}/{gestures.CALIBRATION_FRAMES}",
                                (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                else:
                    blink_count = gestures.triple_quick_blink.count
                    cv2.putText(image, f"Quick blink: {blink_count}/3", (12, 58),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
            else:
                gestures.face_missing(now)
                controller.stop()
                cv2.putText(image, "Face not found", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)

            if gestures.scroll_mode:
                cv2.putText(image, "SCROLL MODE", (12, 88), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 165, 255), 2)
            if gestures.drag_active:
                cv2.rectangle(image, (4, 4), (image.shape[1] - 5, image.shape[0] - 5),
                              (0, 0, 255), 8)
                cv2.putText(image, "DRAG ON", (12, 118), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 255), 2)
            if voice.listening:
                cv2.rectangle(image, (4, 4), (image.shape[1] - 5, image.shape[0] - 5),
                              (255, 120, 0), 8)
                cv2.putText(image, "LISTENING...", (12, 180), cv2.FONT_HERSHEY_SIMPLEX,
                            0.80, (255, 120, 0), 2)
            if now < event_display_until:
                cv2.putText(image, event_text, (12, 150), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 0), 2)
            preview = cv2.resize(image, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
            cv2.imshow(preview_title, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("h"), ord("H")):
                guide_visible = not guide_visible
                if guide_visible:
                    guide_overlay.show()
                    print("모션 안내 표시")
                else:
                    guide_overlay.hide()
                    print("모션 안내 숨김")
            if key in (ord("c"), ord("C")) and last_gesture_frame is not None:
                voice.cancel()
                controller.calibrate(last_gesture_frame.control_point)
                gestures.calibrate()
                print("커서 및 얼굴 제스처 재보정 시작")
    finally:
        voice.cancel()
        gestures.release_drag("프로그램 종료")
        camera.close()
        guide_overlay.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
