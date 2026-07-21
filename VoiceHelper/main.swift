import AVFoundation
import Foundation
import Speech

let arguments = CommandLine.arguments
let authorizeOnly = arguments.contains("--authorize-only")
let outputPath: String? = {
    guard let index = arguments.firstIndex(of: "--output"), index + 1 < arguments.count else {
        return nil
    }
    return arguments[index + 1]
}()
let language: String = {
    guard let index = arguments.firstIndex(of: "--language"), index + 1 < arguments.count else {
        return "ko-KR"
    }
    return arguments[index + 1]
}()

func emit(text: String = "", error: String? = nil) {
    guard let outputPath else { return }
    let object: [String: Any] = ["text": text, "error": error ?? NSNull()]
    guard let data = try? JSONSerialization.data(withJSONObject: object) else { return }
    try? data.write(to: URL(fileURLWithPath: outputPath), options: .atomic)
}

func waitForPermissions() -> Bool {
    var speechDone = false
    var speechAllowed = false
    let speechStatus = SFSpeechRecognizer.authorizationStatus()
    if speechStatus == .notDetermined {
        SFSpeechRecognizer.requestAuthorization { status in
            speechAllowed = status == .authorized
            speechDone = true
        }
    } else {
        speechAllowed = speechStatus == .authorized
        speechDone = true
    }

    var microphoneDone = false
    var microphoneAllowed = false
    let microphoneStatus = AVCaptureDevice.authorizationStatus(for: .audio)
    if microphoneStatus == .notDetermined {
        AVCaptureDevice.requestAccess(for: .audio) { allowed in
            microphoneAllowed = allowed
            microphoneDone = true
        }
    } else {
        microphoneAllowed = microphoneStatus == .authorized
        microphoneDone = true
    }

    let deadline = Date().addingTimeInterval(60)
    while (!speechDone || !microphoneDone) && Date() < deadline {
        RunLoop.current.run(until: Date().addingTimeInterval(0.05))
    }
    if !speechAllowed {
        fputs("음성 인식 권한이 필요합니다. 시스템 설정 → 개인정보 보호 및 보안 → 음성 인식에서 Head Mouse Voice를 허용하세요.\n", stderr)
    }
    if !microphoneAllowed {
        fputs("마이크 권한이 필요합니다. 시스템 설정 → 개인정보 보호 및 보안 → 마이크에서 Head Mouse Voice를 허용하세요.\n", stderr)
    }
    return speechAllowed && microphoneAllowed
}

guard waitForPermissions() else {
    emit(error: "마이크 및 음성 인식 권한이 필요합니다.")
    exit(2)
}
if authorizeOnly {
    emit()
    exit(0)
}

guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: language)),
      recognizer.isAvailable else {
    fputs("현재 \(language) 음성 인식 서비스를 사용할 수 없습니다.\n", stderr)
    emit(error: "현재 \(language) 음성 인식 서비스를 사용할 수 없습니다.")
    exit(3)
}

let engine = AVAudioEngine()
let request = SFSpeechAudioBufferRecognitionRequest()
request.shouldReportPartialResults = true
if #available(macOS 13.0, *) {
    request.addsPunctuation = true
}

let input = engine.inputNode
let format = input.outputFormat(forBus: 0)
guard format.sampleRate > 0 else {
    fputs("사용 가능한 마이크 입력 형식을 찾지 못했습니다.\n", stderr)
    emit(error: "사용 가능한 마이크 입력 형식을 찾지 못했습니다.")
    exit(4)
}

let stateQueue = DispatchQueue(label: "com.igyeongmin.headmouse.voice-state")
var transcript = ""
var lastResultAt = Date()
var finished = false
var recognitionError: Error?

input.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
    request.append(buffer)
}

let task = recognizer.recognitionTask(with: request) { result, error in
    stateQueue.sync {
        if let result {
            let value = result.bestTranscription.formattedString.trimmingCharacters(in: .whitespacesAndNewlines)
            if !value.isEmpty && value != transcript {
                transcript = value
                lastResultAt = Date()
            }
            if result.isFinal {
                finished = true
            }
        }
        if let error {
            recognitionError = error
            finished = true
        }
    }
}

do {
    engine.prepare()
    try engine.start()
} catch {
    input.removeTap(onBus: 0)
    fputs("마이크 시작 오류: \(error.localizedDescription)\n", stderr)
    emit(error: "마이크 시작 오류: \(error.localizedDescription)")
    exit(5)
}

let startedAt = Date()
while true {
    RunLoop.current.run(until: Date().addingTimeInterval(0.05))
    let snapshot = stateQueue.sync { (transcript, lastResultAt, finished, recognitionError) }
    let elapsed = Date().timeIntervalSince(startedAt)
    let silence = Date().timeIntervalSince(snapshot.1)
    if snapshot.2 || elapsed >= 15 || (!snapshot.0.isEmpty && silence >= 1.5) || (snapshot.0.isEmpty && elapsed >= 5) {
        break
    }
}

engine.stop()
input.removeTap(onBus: 0)
request.endAudio()
task.finish()

let finalState = stateQueue.sync { (transcript, recognitionError) }
if !finalState.0.isEmpty {
    emit(text: finalState.0)
    print(finalState.0)
    exit(0)
}
if let error = finalState.1 {
    fputs("음성 인식 오류: \(error.localizedDescription)\n", stderr)
    emit(error: "음성 인식 오류: \(error.localizedDescription)")
} else {
    fputs("음성을 인식하지 못했습니다.\n", stderr)
    emit(error: "음성을 인식하지 못했습니다.")
}
exit(6)
