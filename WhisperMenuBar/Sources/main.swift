import Cocoa
import Carbon.HIToolbox

// MARK: - WhisperClient (Python通信)

class WhisperClient {
    private var socketPath: String
    private var inputStream: InputStream?
    private var outputStream: OutputStream?

    var onPartialResult: ((String) -> Void)?
    var onFinalResult: ((String) -> Void)?
    var onError: ((String) -> Void)?

    init(socketPath: String = "/tmp/whisper-voice.sock") {
        self.socketPath = socketPath
    }

    func connect() -> Bool {
        var readStream: Unmanaged<CFReadStream>?
        var writeStream: Unmanaged<CFWriteStream>?

        CFStreamCreatePairWithSocketToHost(nil, socketPath as CFString, 0, &readStream, &writeStream)

        // Unix socketを使用
        let socket = socket(AF_UNIX, SOCK_STREAM, 0)
        guard socket >= 0 else {
            onError?("Failed to create socket")
            return false
        }

        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        socketPath.withCString { ptr in
            withUnsafeMutablePointer(to: &addr.sun_path.0) { dest in
                _ = strcpy(dest, ptr)
            }
        }

        let result = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPtr in
                Darwin.connect(socket, sockaddrPtr, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }

        guard result >= 0 else {
            close(socket)
            onError?("Failed to connect to whisper server")
            return false
        }

        inputStream = InputStream(fileAtPath: "/dev/fd/\(socket)")
        outputStream = OutputStream(toFileAtPath: "/dev/fd/\(socket)", append: false)

        return true
    }

    func sendCommand(_ command: String) {
        guard let output = outputStream else { return }
        let data = (command + "\n").data(using: .utf8)!
        _ = data.withUnsafeBytes { ptr in
            output.write(ptr.bindMemory(to: UInt8.self).baseAddress!, maxLength: data.count)
        }
    }

    func startRecording() {
        sendCommand("START")
    }

    func stopRecording() {
        sendCommand("STOP")
    }

    func disconnect() {
        inputStream?.close()
        outputStream?.close()
        inputStream = nil
        outputStream = nil
    }
}

// MARK: - PopoverViewController

class PopoverViewController: NSViewController {
    private var statusLabel: NSTextField!
    private var textLabel: NSTextField!
    private var hotkeyName: String

    init(hotkeyName: String) {
        self.hotkeyName = hotkeyName
        super.init(nibName: nil, bundle: nil)
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func loadView() {
        let containerView = NSView(frame: NSRect(x: 0, y: 0, width: 350, height: 80))
        containerView.wantsLayer = true

        // ステータスラベル
        statusLabel = NSTextField(labelWithString: "🎤 \(hotkeyName) を押して録音")
        statusLabel.font = NSFont.systemFont(ofSize: 12)
        statusLabel.textColor = NSColor.secondaryLabelColor
        statusLabel.frame = NSRect(x: 15, y: 50, width: 320, height: 20)
        containerView.addSubview(statusLabel)

        // テキストラベル
        textLabel = NSTextField(labelWithString: "")
        textLabel.font = NSFont.systemFont(ofSize: 14)
        textLabel.textColor = NSColor.labelColor
        textLabel.frame = NSRect(x: 15, y: 10, width: 320, height: 35)
        textLabel.maximumNumberOfLines = 2
        textLabel.lineBreakMode = .byWordWrapping
        containerView.addSubview(textLabel)

        self.view = containerView
    }

    func setRecording(_ isRecording: Bool) {
        DispatchQueue.main.async {
            if isRecording {
                self.statusLabel.stringValue = "🔴 録音中..."
                self.statusLabel.textColor = NSColor.systemRed
            } else {
                self.statusLabel.stringValue = "🎤 \(self.hotkeyName) を押して録音"
                self.statusLabel.textColor = NSColor.secondaryLabelColor
            }
        }
    }

    func setText(_ text: String) {
        DispatchQueue.main.async {
            self.textLabel.stringValue = text
        }
    }

    func setFinalText(_ text: String) {
        DispatchQueue.main.async {
            self.statusLabel.stringValue = "✓ クリップボードにコピーしました"
            self.statusLabel.textColor = NSColor.systemGreen
            self.textLabel.stringValue = text
        }
    }

    func reset() {
        DispatchQueue.main.async {
            self.statusLabel.stringValue = "🎤 \(self.hotkeyName) を押して録音"
            self.statusLabel.textColor = NSColor.secondaryLabelColor
            self.textLabel.stringValue = ""
        }
    }
}

// MARK: - StatusBarController

class StatusBarController {
    private var statusItem: NSStatusItem!
    private var popover: NSPopover!
    private var popoverViewController: PopoverViewController!
    private var eventMonitor: Any?
    private var hotkeyName: String

    var onQuit: (() -> Void)?

    init(hotkeyName: String) {
        self.hotkeyName = hotkeyName
        setupStatusItem()
        setupPopover()
    }

    private func setupStatusItem() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)

        if let button = statusItem.button {
            button.title = "🎤"
        }

        // メニュー設定
        let menu = NSMenu()
        menu.addItem(NSMenuItem(title: "終了", action: #selector(quit), keyEquivalent: "q"))
        menu.items.forEach { $0.target = self }
        statusItem.menu = menu
    }

    private func setupPopover() {
        popover = NSPopover()
        popover.behavior = .transient
        popover.animates = true

        popoverViewController = PopoverViewController(hotkeyName: hotkeyName)
        popover.contentViewController = popoverViewController
        popover.contentSize = NSSize(width: 350, height: 80)
    }

    @objc private func quit() {
        onQuit?()
        NSApplication.shared.terminate(nil)
    }

    func showPopover() {
        guard let button = statusItem.button else { return }
        popover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
    }

    func hidePopover() {
        popover.performClose(nil)
    }

    func setRecording(_ isRecording: Bool) {
        DispatchQueue.main.async {
            if let button = self.statusItem.button {
                button.title = isRecording ? "🔴" : "🎤"
            }
            self.popoverViewController.setRecording(isRecording)
            if isRecording {
                self.showPopover()
            }
        }
    }

    func setText(_ text: String) {
        popoverViewController.setText(text)
    }

    func setFinalText(_ text: String) {
        popoverViewController.setFinalText(text)

        // 2秒後にポップオーバーを閉じてリセット
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            self.hidePopover()
            self.popoverViewController.reset()
        }
    }
}

// MARK: - HotkeyManager

class HotkeyManager {
    private var eventTap: CFMachPort?
    private var runLoopSource: CFRunLoopSource?
    private var targetKeyCode: CGKeyCode
    private var isKeyDown = false

    var onKeyDown: (() -> Void)?
    var onKeyUp: (() -> Void)?

    init(keyCode: CGKeyCode) {
        self.targetKeyCode = keyCode
    }

    func start() -> Bool {
        let eventMask = (1 << CGEventType.keyDown.rawValue) | (1 << CGEventType.keyUp.rawValue) | (1 << CGEventType.flagsChanged.rawValue)

        guard let tap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .defaultTap,
            eventsOfInterest: CGEventMask(eventMask),
            callback: { (proxy, type, event, refcon) -> Unmanaged<CGEvent>? in
                let manager = Unmanaged<HotkeyManager>.fromOpaque(refcon!).takeUnretainedValue()
                return manager.handleEvent(proxy: proxy, type: type, event: event)
            },
            userInfo: Unmanaged.passUnretained(self).toOpaque()
        ) else {
            print("Failed to create event tap. Check accessibility permissions.")
            return false
        }

        eventTap = tap
        runLoopSource = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        CFRunLoopAddSource(CFRunLoopGetCurrent(), runLoopSource, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)

        return true
    }

    private func handleEvent(proxy: CGEventTapProxy, type: CGEventType, event: CGEvent) -> Unmanaged<CGEvent>? {
        if type == .flagsChanged {
            let keyCode = CGKeyCode(event.getIntegerValueField(.keyboardEventKeycode))

            if keyCode == targetKeyCode {
                let flags = event.flags

                // 右Ctrlキーの場合
                if targetKeyCode == 62 { // kVK_RightControl
                    let isPressed = flags.contains(.maskControl)
                    if isPressed && !isKeyDown {
                        isKeyDown = true
                        DispatchQueue.main.async { self.onKeyDown?() }
                    } else if !isPressed && isKeyDown {
                        isKeyDown = false
                        DispatchQueue.main.async { self.onKeyUp?() }
                    }
                }
            }
        }

        // F9キーなどのファンクションキー
        if type == .keyDown {
            let keyCode = CGKeyCode(event.getIntegerValueField(.keyboardEventKeycode))
            if keyCode == targetKeyCode && !isKeyDown {
                isKeyDown = true
                DispatchQueue.main.async { self.onKeyDown?() }
            }
        } else if type == .keyUp {
            let keyCode = CGKeyCode(event.getIntegerValueField(.keyboardEventKeycode))
            if keyCode == targetKeyCode && isKeyDown {
                isKeyDown = false
                DispatchQueue.main.async { self.onKeyUp?() }
            }
        }

        return Unmanaged.passRetained(event)
    }

    func stop() {
        if let tap = eventTap {
            CGEvent.tapEnable(tap: tap, enable: false)
        }
        if let source = runLoopSource {
            CFRunLoopRemoveSource(CFRunLoopGetCurrent(), source, .commonModes)
        }
        eventTap = nil
        runLoopSource = nil
    }
}

// MARK: - AppDelegate

class AppDelegate: NSObject, NSApplicationDelegate {
    private var statusBarController: StatusBarController!
    private var hotkeyManager: HotkeyManager!
    private var whisperProcess: Process?
    private var outputPipe: Pipe?
    private var isRecording = false

    // 設定
    private let hotkeyCode: CGKeyCode = 101  // F9 = 101, Right Ctrl = 62
    private let hotkeyName = "F9"
    private let whisperModel = "large-v3-turbo"  // base, small, large-v3-turbo
    private let whisperLanguage = "ja"

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Dockアイコンを非表示
        NSApp.setActivationPolicy(.accessory)

        // ステータスバー設定
        statusBarController = StatusBarController(hotkeyName: hotkeyName)
        statusBarController.onQuit = { [weak self] in
            self?.cleanup()
        }

        // ホットキー設定
        hotkeyManager = HotkeyManager(keyCode: hotkeyCode)
        hotkeyManager.onKeyDown = { [weak self] in
            self?.startRecording()
        }
        hotkeyManager.onKeyUp = { [weak self] in
            self?.stopRecording()
        }

        if !hotkeyManager.start() {
            showAccessibilityAlert()
        }

        print("🎤 メニューバーで待機中... \(hotkeyName) を押して録音")
    }

    private func showAccessibilityAlert() {
        let alert = NSAlert()
        alert.messageText = "アクセシビリティ権限が必要です"
        alert.informativeText = "システム設定 > プライバシーとセキュリティ > アクセシビリティ でこのアプリを許可してください。"
        alert.alertStyle = .warning
        alert.addButton(withTitle: "システム設定を開く")
        alert.addButton(withTitle: "キャンセル")

        if alert.runModal() == .alertFirstButtonReturn {
            NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")!)
        }
    }

    private func startRecording() {
        guard !isRecording else { return }
        isRecording = true

        statusBarController.setRecording(true)
        statusBarController.setText("")

        // whisper-realtime voice を起動（シングルショットモード）
        let scriptPath = Bundle.main.bundlePath + "/../../../"

        whisperProcess = Process()
        whisperProcess?.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        whisperProcess?.arguments = [
            "uv", "run", "--project", scriptPath,
            "whisper-realtime", "voice-single",
            "-m", whisperModel,
            "-l", whisperLanguage
        ]
        whisperProcess?.currentDirectoryURL = URL(fileURLWithPath: scriptPath)

        outputPipe = Pipe()
        whisperProcess?.standardOutput = outputPipe
        whisperProcess?.standardError = outputPipe

        outputPipe?.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if let output = String(data: data, encoding: .utf8), !output.isEmpty {
                DispatchQueue.main.async {
                    self?.handleWhisperOutput(output)
                }
            }
        }

        do {
            try whisperProcess?.run()
        } catch {
            print("Failed to start whisper: \(error)")
            isRecording = false
            statusBarController.setRecording(false)
        }
    }

    private func stopRecording() {
        guard isRecording else { return }
        isRecording = false

        // プロセスにSIGINTを送信して録音停止
        if let process = whisperProcess, process.isRunning {
            process.interrupt()
        }
    }

    private func handleWhisperOutput(_ output: String) {
        let lines = output.components(separatedBy: "\n")
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }

            if trimmed.hasPrefix("PARTIAL:") {
                let text = String(trimmed.dropFirst(8))
                statusBarController.setText(text)
            } else if trimmed.hasPrefix("FINAL:") {
                let text = String(trimmed.dropFirst(6))
                copyToClipboard(text)
                statusBarController.setFinalText(text)

                // 録音状態をリセット
                isRecording = false
                statusBarController.setRecording(false)

                // プロセスクリーンアップ
                whisperProcess = nil
                outputPipe = nil
            }
        }
    }

    private func copyToClipboard(_ text: String) {
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)
    }

    private func cleanup() {
        hotkeyManager.stop()
        whisperProcess?.terminate()
    }

    func applicationWillTerminate(_ notification: Notification) {
        cleanup()
    }
}

// MARK: - Main

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()
