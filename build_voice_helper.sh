#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
APP_DIR="$SCRIPT_DIR/VoiceHelper/HeadMouseVoice.app"
INFO_PLIST="$APP_DIR/Contents/Info.plist"
EXECUTABLE="$APP_DIR/Contents/MacOS/HeadMouseVoice"

mkdir -p "$APP_DIR/Contents/MacOS"
xcrun swiftc "$SCRIPT_DIR/VoiceHelper/main.swift" \
  -o "$EXECUTABLE" \
  -framework Speech \
  -framework AVFoundation \
  -Xlinker -sectcreate \
  -Xlinker __TEXT \
  -Xlinker __info_plist \
  -Xlinker "$INFO_PLIST"
codesign --force --sign - "$APP_DIR"
codesign --verify --deep --strict "$APP_DIR"
echo "Built: $APP_DIR"
