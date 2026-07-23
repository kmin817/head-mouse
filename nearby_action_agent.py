"""Local macOS agent that presses a named actionable element near the cursor."""

from __future__ import annotations

import ctypes
import difflib
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from AppKit import NSRunningApplication, NSWorkspace


K_CF_STRING_ENCODING_UTF8 = 0x08000100
K_AX_VALUE_CGPOINT_TYPE = 1
K_AX_VALUE_CGSIZE_TYPE = 2
K_AX_ERROR_SUCCESS = 0


class CGPoint(ctypes.Structure):
    _fields_ = (("x", ctypes.c_double), ("y", ctypes.c_double))


class CGSize(ctypes.Structure):
    _fields_ = (("width", ctypes.c_double), ("height", ctypes.c_double))


@dataclass
class NearbyElement:
    label: str
    role: str
    source: str
    frame: tuple[float, float, float, float]
    distance: float
    element: int = field(repr=False)


@dataclass
class AgentResult:
    is_command: bool
    success: bool = False
    message: str = ""
    target: Optional[str] = None
    blocked: bool = False


class AccessibilityBridge:
    """Small ctypes bridge for APIs not exported by pyobjc-framework-Quartz."""

    ATTRIBUTES = (
        "AXChildren",
        "AXVisibleChildren",
        "AXParent",
        "AXEnhancedUserInterface",
        "AXTitle",
        "AXDescription",
        "AXHelp",
        "AXValue",
        "AXRole",
        "AXPosition",
        "AXSize",
    )

    def __init__(self) -> None:
        self.cf = ctypes.CDLL(
            "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
        )
        self.ax = ctypes.CDLL(
            "/System/Library/Frameworks/ApplicationServices.framework/"
            "Frameworks/HIServices.framework/HIServices"
        )
        self._configure_functions()
        self.cf_true = int(ctypes.c_void_p.in_dll(self.cf, "kCFBooleanTrue").value)
        self.cf_false = int(ctypes.c_void_p.in_dll(self.cf, "kCFBooleanFalse").value)
        self.strings = {
            value: self._create_string(value)
            for value in (*self.ATTRIBUTES, "AXPress")
        }

    def _configure_functions(self) -> None:
        self.cf.CFStringCreateWithCString.argtypes = (
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_uint32,
        )
        self.cf.CFStringCreateWithCString.restype = ctypes.c_void_p
        self.cf.CFStringGetLength.argtypes = (ctypes.c_void_p,)
        self.cf.CFStringGetLength.restype = ctypes.c_long
        self.cf.CFStringGetMaximumSizeForEncoding.argtypes = (
            ctypes.c_long,
            ctypes.c_uint32,
        )
        self.cf.CFStringGetMaximumSizeForEncoding.restype = ctypes.c_long
        self.cf.CFStringGetCString.argtypes = (
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_long,
            ctypes.c_uint32,
        )
        self.cf.CFStringGetCString.restype = ctypes.c_bool
        self.cf.CFArrayGetCount.argtypes = (ctypes.c_void_p,)
        self.cf.CFArrayGetCount.restype = ctypes.c_long
        self.cf.CFArrayGetValueAtIndex.argtypes = (ctypes.c_void_p, ctypes.c_long)
        self.cf.CFArrayGetValueAtIndex.restype = ctypes.c_void_p
        self.cf.CFGetTypeID.argtypes = (ctypes.c_void_p,)
        self.cf.CFGetTypeID.restype = ctypes.c_ulong
        self.cf.CFStringGetTypeID.restype = ctypes.c_ulong
        self.cf.CFArrayGetTypeID.restype = ctypes.c_ulong
        self.cf.CFRetain.argtypes = (ctypes.c_void_p,)
        self.cf.CFRetain.restype = ctypes.c_void_p
        self.cf.CFRelease.argtypes = (ctypes.c_void_p,)

        self.ax.AXIsProcessTrusted.restype = ctypes.c_bool
        self.ax.AXUIElementCreateApplication.argtypes = (ctypes.c_int,)
        self.ax.AXUIElementCreateApplication.restype = ctypes.c_void_p
        self.ax.AXUIElementCreateSystemWide.restype = ctypes.c_void_p
        self.ax.AXUIElementCopyElementAtPosition.argtypes = (
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_void_p),
        )
        self.ax.AXUIElementCopyElementAtPosition.restype = ctypes.c_int32
        self.ax.AXUIElementCopyAttributeValue.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )
        self.ax.AXUIElementCopyAttributeValue.restype = ctypes.c_int32
        self.ax.AXUIElementSetAttributeValue.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        self.ax.AXUIElementSetAttributeValue.restype = ctypes.c_int32
        self.ax.AXUIElementCopyActionNames.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        )
        self.ax.AXUIElementCopyActionNames.restype = ctypes.c_int32
        self.ax.AXUIElementPerformAction.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
        )
        self.ax.AXUIElementPerformAction.restype = ctypes.c_int32
        self.ax.AXValueGetValue.argtypes = (
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
        )
        self.ax.AXValueGetValue.restype = ctypes.c_bool

    def _create_string(self, value: str) -> int:
        result = self.cf.CFStringCreateWithCString(
            None,
            value.encode("utf-8"),
            K_CF_STRING_ENCODING_UTF8,
        )
        if not result:
            raise RuntimeError(f"macOS 문자열 생성 실패: {value}")
        return int(result)

    def trusted(self) -> bool:
        return bool(self.ax.AXIsProcessTrusted())

    def create_application(self, pid: int) -> Optional[int]:
        result = self.ax.AXUIElementCreateApplication(pid)
        return int(result) if result else None

    def element_at_position(self, x: float, y: float) -> Optional[int]:
        system = self.ax.AXUIElementCreateSystemWide()
        if not system:
            return None
        try:
            output = ctypes.c_void_p()
            error = self.ax.AXUIElementCopyElementAtPosition(
                system,
                float(x),
                float(y),
                ctypes.byref(output),
            )
            if error == K_AX_ERROR_SUCCESS and output.value:
                return int(output.value)
            return None
        finally:
            self.release(int(system))

    def copy_attribute(self, element: int, name: str) -> Optional[int]:
        output = ctypes.c_void_p()
        error = self.ax.AXUIElementCopyAttributeValue(
            element,
            self.strings[name],
            ctypes.byref(output),
        )
        if error == K_AX_ERROR_SUCCESS and output.value:
            return int(output.value)
        return None

    def set_boolean_attribute(self, element: int, name: str, enabled: bool) -> int:
        value = self.cf_true if enabled else self.cf_false
        return int(
            self.ax.AXUIElementSetAttributeValue(
                element,
                self.strings[name],
                value,
            )
        )

    def string_value(self, value: int) -> Optional[str]:
        if self.cf.CFGetTypeID(value) != self.cf.CFStringGetTypeID():
            return None
        length = self.cf.CFStringGetLength(value)
        capacity = self.cf.CFStringGetMaximumSizeForEncoding(
            length,
            K_CF_STRING_ENCODING_UTF8,
        ) + 1
        buffer = ctypes.create_string_buffer(capacity)
        if not self.cf.CFStringGetCString(
            value,
            buffer,
            capacity,
            K_CF_STRING_ENCODING_UTF8,
        ):
            return None
        return buffer.value.decode("utf-8", errors="replace")

    def string_attribute(self, element: int, name: str) -> Optional[str]:
        value = self.copy_attribute(element, name)
        if value is None:
            return None
        try:
            return self.string_value(value)
        finally:
            self.release(value)

    def children(self, element: int) -> tuple[list[int], Optional[int]]:
        value = self.copy_attribute(element, "AXChildren")
        if value is None:
            value = self.copy_attribute(element, "AXVisibleChildren")
        if value is None:
            return [], None
        if self.cf.CFGetTypeID(value) != self.cf.CFArrayGetTypeID():
            self.release(value)
            return [], None
        count = min(int(self.cf.CFArrayGetCount(value)), 500)
        items = [
            int(self.cf.CFArrayGetValueAtIndex(value, index))
            for index in range(count)
        ]
        return [item for item in items if item], value

    def parent(self, element: int) -> Optional[int]:
        """Return a retained parent reference, or None when no parent exists."""
        return self.copy_attribute(element, "AXParent")

    def action_names(self, element: int) -> list[str]:
        output = ctypes.c_void_p()
        error = self.ax.AXUIElementCopyActionNames(element, ctypes.byref(output))
        if error != K_AX_ERROR_SUCCESS or not output.value:
            return []
        array = int(output.value)
        try:
            count = int(self.cf.CFArrayGetCount(array))
            result: list[str] = []
            for index in range(count):
                value = int(self.cf.CFArrayGetValueAtIndex(array, index))
                text = self.string_value(value)
                if text:
                    result.append(text)
            return result
        finally:
            self.release(array)

    def _ax_value(self, element: int, name: str, value_type: int,
                  output: ctypes.Structure) -> bool:
        value = self.copy_attribute(element, name)
        if value is None:
            return False
        try:
            return bool(self.ax.AXValueGetValue(value, value_type, ctypes.byref(output)))
        finally:
            self.release(value)

    def frame(self, element: int) -> Optional[tuple[float, float, float, float]]:
        point = CGPoint()
        size = CGSize()
        if not self._ax_value(element, "AXPosition", K_AX_VALUE_CGPOINT_TYPE, point):
            return None
        if not self._ax_value(element, "AXSize", K_AX_VALUE_CGSIZE_TYPE, size):
            return None
        if size.width <= 0 or size.height <= 0:
            return None
        return (float(point.x), float(point.y), float(size.width), float(size.height))

    def retain(self, value: int) -> None:
        self.cf.CFRetain(value)

    def release(self, value: Optional[int]) -> None:
        if value:
            self.cf.CFRelease(value)

    def press(self, element: int) -> int:
        return int(self.ax.AXUIElementPerformAction(element, self.strings["AXPress"]))

    def close(self) -> None:
        for value in self.strings.values():
            self.release(value)
        self.strings.clear()


class NearbyActionAgent:
    """Resolve a spoken action against clickable UI elements near the cursor."""

    COMMAND_PATTERNS = (
        re.compile(
            r"^\s*(.+?)(?:을|를)?\s*"
            r"((?:켜|열어|눌러|골라|띄워|보여)\s*(?:줘|주세요)?|"
            r"(?:터치|탭)\s*(?:해)?\s*(?:줘|주세요)?|"
            r"(?:들어가|이동)\s*(?:해)?\s*(?:줘|주세요)?|"
            r"실행\s*(?:해|시켜)?\s*(?:줘|주세요)?|"
            r"클릭\s*(?:해)?\s*(?:줘|주세요)?|"
            r"선택\s*(?:해)?\s*(?:줘|주세요))"
            r"\s*[.!?]?\s*$"
        ),
        re.compile(
            r"^\s*(?:open|click|press|launch|tap|select|show|go\s+to)\s+"
            r"(.+?)\s*[.!?]?\s*$",
            re.I,
        ),
    )
    GENERIC_TARGETS = {
        "이거", "이것", "여기", "이버튼", "버튼", "이아이콘", "아이콘",
        "이항목", "항목", "이링크", "링크", "저링크", "그링크",
    }
    HIGH_RISK_TERMS = (
        "삭제", "지우기", "초기화", "결제", "구매", "주문", "송금", "전송",
        "제출", "허용", "동의", "업로드", "다운로드", "설치", "게시", "구독",
        "로그인", "로그아웃", "비밀번호", "delete", "erase", "reset", "pay",
        "purchase", "order", "transfer", "send", "submit", "allow", "accept",
        "upload", "download", "install", "post", "subscribe", "signin", "signout",
    )
    ALIASES = {
        "메모장": ("메모", "notes", "stickies", "텍스트편집", "textedit"),
        "메모": ("메모장", "notes"),
        "애플뮤직": ("음악", "music", "applemusic"),
        "음악": ("애플뮤직", "music", "applemusic"),
        "크롬": ("googlechrome", "chrome", "구글크롬"),
        "구글크롬": ("googlechrome", "chrome", "크롬"),
        "사파리": ("safari",),
        "파인더": ("finder",),
        "계산기": ("calculator",),
        "캘린더": ("calendar", "달력"),
        "설정": ("시스템설정", "systemsettings", "systempreferences"),
    }

    def __init__(self, radius: float = 280.0) -> None:
        self.radius = radius
        self.bridge = AccessibilityBridge()
        self.elements: list[NearbyElement] = []
        self.last_error: Optional[str] = None
        self.last_cursor: Optional[tuple[float, float]] = None
        self.last_frontmost = "알 수 없음"
        self.last_web_accessibility = "해당 없음"
        self.last_resolution_candidates: list[str] = []
        self.enhanced_pids: set[int] = set()

    @staticmethod
    def _normalize(value: str) -> str:
        value = unicodedata.normalize("NFKC", value).casefold()
        return "".join(character for character in value if character.isalnum())

    @staticmethod
    def _distance_to_frame(
        point: tuple[float, float], frame: tuple[float, float, float, float]
    ) -> float:
        px, py = point
        x, y, width, height = frame
        dx = max(x - px, 0.0, px - (x + width))
        dy = max(y - py, 0.0, py - (y + height))
        return math.hypot(dx, dy)

    def clear(self) -> None:
        for candidate in self.elements:
            self.bridge.release(candidate.element)
        self.elements.clear()
        self.last_error = None

    def _collect_tree(
        self,
        root: int,
        source: str,
        cursor: tuple[float, float],
        max_depth: int = 8,
        max_nodes: int = 800,
    ) -> None:
        visited: set[int] = set()
        node_count = 0

        def visit(element: int, depth: int) -> None:
            nonlocal node_count
            if element in visited or node_count >= max_nodes:
                return
            visited.add(element)
            node_count += 1

            frame = self.bridge.frame(element)
            actions = self.bridge.action_names(element)
            if frame is not None and "AXPress" in actions:
                distance = self._distance_to_frame(cursor, frame)
                if distance <= self.radius:
                    labels = []
                    for attribute in ("AXTitle", "AXDescription", "AXHelp", "AXValue"):
                        text = self.bridge.string_attribute(element, attribute)
                        if text and text not in labels:
                            labels.append(text)
                    role = self.bridge.string_attribute(element, "AXRole") or ""
                    label = " · ".join(labels) or role
                    if label:
                        self.bridge.retain(element)
                        self.elements.append(
                            NearbyElement(label, role, source, frame, distance, element)
                        )

            if depth >= max_depth:
                return
            children, array = self.bridge.children(element)
            try:
                for child in children:
                    visit(child, depth + 1)
            finally:
                self.bridge.release(array)

        visit(root, 0)

    def capture(self, cursor: tuple[float, float]) -> int:
        self.clear()
        self.last_cursor = cursor
        self.last_resolution_candidates = []
        if not self.bridge.trusted():
            self.last_error = (
                "손쉬운 사용 권한이 필요합니다. 시스템 설정 → 개인정보 보호 및 보안 → "
                "손쉬운 사용에서 실행 중인 Terminal 또는 IDE를 허용하세요."
            )
            return 0

        roots: list[tuple[int, str, bool, int, int]] = []

        # Chrome normally delays exposing web-page accessibility nodes until an
        # assistive client requests enhanced UI. Request it before hit-testing;
        # current Chromium may debounce activation, so handle() captures again
        # after the user has finished speaking.
        frontmost = NSWorkspace.sharedWorkspace().frontmostApplication()
        front_root: Optional[int] = None
        if frontmost is not None:
            front_pid = int(frontmost.processIdentifier())
            front_name = str(frontmost.localizedName() or "현재 앱")
            front_bundle = str(frontmost.bundleIdentifier() or "")
            self.last_frontmost = f"{front_name} ({front_bundle or 'bundle id 없음'})"
            front_root = self.bridge.create_application(front_pid)
            is_chromium = any(
                marker in front_bundle.casefold()
                for marker in ("chrome", "chromium", "brave", "edge", "arc")
            )
            if front_root is not None and is_chromium and front_pid not in self.enhanced_pids:
                # Reading AXRole also enables Chrome's native accessibility mode
                # on recent versions; enhanced UI requests the complete web tree.
                self.bridge.string_attribute(front_root, "AXRole")
                error = self.bridge.set_boolean_attribute(
                    front_root,
                    "AXEnhancedUserInterface",
                    True,
                )
                if error == K_AX_ERROR_SUCCESS:
                    self.enhanced_pids.add(front_pid)
                    self.last_web_accessibility = "Chrome 웹 접근성 활성화 요청 성공"
                else:
                    self.last_web_accessibility = f"Chrome 웹 접근성 활성화 요청 실패 (AX {error})"
            elif is_chromium and front_pid in self.enhanced_pids:
                self.last_web_accessibility = "Chrome 웹 접근성 활성화 유지 중"
            elif is_chromium:
                self.last_web_accessibility = "Chrome 접근성 루트를 찾지 못함"
            else:
                self.last_web_accessibility = "해당 없음"
        else:
            self.last_frontmost = "알 수 없음"
            self.last_web_accessibility = "해당 없음"

        element = self.bridge.element_at_position(*cursor)
        if element is not None:
            # Web browsers often expose the text under the pointer as
            # AXStaticText while the clickable AXPress action lives on a parent
            # link/button. Walk upward and inspect each ancestor plus siblings.
            cursor_element: Optional[int] = element
            seen_ancestors: set[int] = set()
            ancestor_level = 0
            while cursor_element is not None and ancestor_level < 7:
                if cursor_element in seen_ancestors:
                    self.bridge.release(cursor_element)
                    break
                seen_ancestors.add(cursor_element)
                roots.append((
                    cursor_element,
                    "커서 위치" if ancestor_level == 0 else "커서 주변",
                    True,
                    3,
                    350,
                ))
                cursor_element = self.bridge.parent(cursor_element)
                ancestor_level += 1

        dock_apps = NSRunningApplication.runningApplicationsWithBundleIdentifier_(
            "com.apple.dock"
        )
        if dock_apps:
            dock_root = self.bridge.create_application(int(dock_apps[0].processIdentifier()))
            if dock_root is not None:
                roots.append((dock_root, "Dock", True, 8, 800))

        if (
            frontmost is not None
            and str(frontmost.bundleIdentifier() or "") != "com.apple.dock"
            and front_root is not None
        ):
            roots.append((
                front_root,
                str(frontmost.localizedName() or "현재 앱"),
                True,
                10,
                1600,
            ))
        elif front_root is not None:
            self.bridge.release(front_root)

        seen_roots: set[int] = set()
        try:
            for root, source, _owned, max_depth, max_nodes in roots:
                if root in seen_roots:
                    continue
                seen_roots.add(root)
                self._collect_tree(
                    root,
                    source,
                    cursor,
                    max_depth=max_depth,
                    max_nodes=max_nodes,
                )
        finally:
            for root, _source, owned, _max_depth, _max_nodes in roots:
                if owned:
                    self.bridge.release(root)

        # The same element can be reached from the cursor, Dock, and app roots.
        unique: dict[tuple[str, tuple[int, int, int, int]], NearbyElement] = {}
        for candidate in self.elements:
            key = (
                candidate.label,
                tuple(round(value) for value in candidate.frame),
            )
            existing = unique.get(key)
            if existing is None or candidate.distance < existing.distance:
                if existing is not None:
                    self.bridge.release(existing.element)
                unique[key] = candidate
            else:
                self.bridge.release(candidate.element)
        self.elements = sorted(unique.values(), key=lambda item: item.distance)
        return len(self.elements)

    def _parse_command(self, transcript: str) -> Optional[str]:
        for index, pattern in enumerate(self.COMMAND_PATTERNS):
            match = pattern.match(transcript)
            if not match:
                continue
            target = match.group(1).strip()
            if index == 0:
                target = re.sub(r"(?:을|를|으로|로)$", "", target).strip()
                target = re.sub(
                    r"\s*(?:앱|어플|프로그램)?\s*(?:좀)?\s*$",
                    "",
                    target,
                ).strip()
                without_hint = re.sub(
                    r"\s*(?:텍스트|글자|문구|항목|버튼|링크|아이콘)\s*$",
                    "",
                    target,
                ).strip()
                # Keep generic commands such as "이 버튼 눌러줘" intact.
                if self._normalize(without_hint) not in {"", "이", "저", "그"}:
                    target = without_hint
            return target or None
        return None

    def _variants(self, value: str) -> set[str]:
        normalized = self._normalize(value)
        variants = {normalized}
        variants.update(self.ALIASES.get(normalized, ()))
        return {self._normalize(item) for item in variants if item}

    @classmethod
    def _is_high_risk(cls, transcript: str, target: str) -> bool:
        normalized = "".join((transcript + target).casefold().split())
        return any(term in normalized for term in cls.HIGH_RISK_TERMS)

    @staticmethod
    def _text_similarity(left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        if left == right:
            return 1.0
        if min(len(left), len(right)) >= 2 and (left in right or right in left):
            return 0.92
        return difflib.SequenceMatcher(None, left, right).ratio()

    def _semantic_score(self, target: str, candidate: NearbyElement) -> float:
        targets = self._variants(target)
        labels = self._variants(candidate.label)
        return max(
            (self._text_similarity(left, right) for left in targets for right in labels),
            default=0.0,
        )

    def handle(self, transcript: str) -> AgentResult:
        target = self._parse_command(transcript)
        if target is None:
            self.clear()
            return AgentResult(is_command=False)
        if self._is_high_risk(transcript, target):
            self.clear()
            return AgentResult(
                True,
                False,
                "삭제·결제·전송·권한 허용 등 위험한 명령은 자동 실행하지 않습니다.",
                target,
                True,
            )
        # Refresh at command time. Chrome may take a short time to publish its
        # full web accessibility tree after AXEnhancedUserInterface is enabled,
        # and the page may also have changed while the user was speaking.
        if self.last_cursor is not None:
            self.capture(self.last_cursor)
        self.last_resolution_candidates = [item.label for item in self.elements[:12]]
        if self.last_error:
            message = self.last_error
            self.clear()
            return AgentResult(True, False, message, target)
        if not self.elements:
            self.clear()
            return AgentResult(
                True,
                False,
                "커서 근처에서 실행할 수 있는 요소를 찾지 못했습니다.",
                target,
            )

        normalized_target = self._normalize(target)
        generic = normalized_target in self.GENERIC_TARGETS
        ranked: list[tuple[float, float, NearbyElement]] = []
        for candidate in self.elements:
            semantic = 1.0 if generic else self._semantic_score(target, candidate)
            proximity = max(0.0, 1.0 - candidate.distance / self.radius)
            score = semantic * 0.85 + proximity * 0.15
            ranked.append((score, semantic, candidate))
        ranked.sort(key=lambda item: (item[0], -item[2].distance), reverse=True)
        best_score, best_semantic, best = ranked[0]

        if not generic and (best_semantic < 0.60 or best_score < 0.62):
            self.clear()
            return AgentResult(
                True,
                False,
                f"커서 근처에서 ‘{target}’과 일치하는 요소를 찾지 못했습니다.",
                target,
            )
        if len(ranked) > 1:
            second_score, second_semantic, second = ranked[1]
            if (
                not generic
                and second_semantic >= 0.75
                and best_score - second_score < 0.04
                and best.label != second.label
            ):
                self.clear()
                return AgentResult(
                    True,
                    False,
                    f"‘{target}’과 비슷한 요소가 여러 개라 실행하지 않았습니다.",
                    target,
                )

        label = best.label
        error = self.bridge.press(best.element)
        self.clear()
        if error != K_AX_ERROR_SUCCESS:
            return AgentResult(
                True,
                False,
                f"‘{label}’ 액션 실행에 실패했습니다. (AX 오류 {error})",
                target,
            )
        return AgentResult(
            True,
            True,
            f"‘{label}’을(를) 실행했습니다.",
            target,
        )

    def close(self) -> None:
        self.clear()
        for pid in self.enhanced_pids:
            application = self.bridge.create_application(pid)
            if application is None:
                continue
            try:
                self.bridge.set_boolean_attribute(
                    application,
                    "AXEnhancedUserInterface",
                    False,
                )
            finally:
                self.bridge.release(application)
        self.enhanced_pids.clear()
        self.bridge.close()
