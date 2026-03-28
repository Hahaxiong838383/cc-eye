/**
 cc_audio_bridge.swift — VP 音频输入桥接（无播放节点，避免冲突）

 VP 启用后输入变为 7ch。逐通道发给 Python 分析哪个有 AEC。
 播放由 Python sd.play 处理，VP 在系统层追踪输出设备。

 协议：
   Swift → Python (stdout): [0x10][len:4][ch:1][pcm data]
   Python → Swift (stdin):  [0x03] = 退出
*/

import AVFoundation
import Foundation

let MSG_EXIT: UInt8  = 0x03
let MSG_MIC: UInt8   = 0x10
let MSG_READY: UInt8 = 0x12

let TAP_BUFFER_SIZE: UInt32 = 4096
let CHUNK_SIZE: Int = 1536  // ~32ms @ 48kHz

var engine: AVAudioEngine!
var isRunning = true
let writeLock = NSLock()

func sendMessage(_ type: UInt8, _ payload: Data = Data()) {
    writeLock.lock()
    defer { writeLock.unlock() }
    var header = Data()
    header.append(type)
    var len = UInt32(payload.count).littleEndian
    header.append(Data(bytes: &len, count: 4))
    FileHandle.standardOutput.write(header)
    if !payload.isEmpty { FileHandle.standardOutput.write(payload) }
}

func log(_ msg: String) {
    FileHandle.standardError.write("[bridge] \(msg)\n".data(using: .utf8)!)
}

// 缓冲（每通道分别缓冲）
var channelBuffers: [[Float]] = []
let bufLock = NSLock()

func onTap(_ buffer: AVAudioPCMBuffer, _ time: AVAudioTime) {
    guard let channelData = buffer.floatChannelData else { return }
    let frameCount = Int(buffer.frameLength)
    let channelCount = Int(buffer.format.channelCount)
    if frameCount == 0 { return }

    bufLock.lock()
    // 确保缓冲数量对
    while channelBuffers.count < channelCount {
        channelBuffers.append([])
    }

    // 收集每个通道的数据
    for ch in 0..<channelCount {
        let ptr = channelData[ch]
        let samples = Array(UnsafeBufferPointer(start: ptr, count: frameCount))
        channelBuffers[ch].append(contentsOf: samples)
    }

    // 只发 ch0（VP 后所有通道一样）
    while channelBuffers[0].count >= CHUNK_SIZE {
        let chunk = Array(channelBuffers[0].prefix(CHUNK_SIZE))
        channelBuffers[0].removeFirst(CHUNK_SIZE)
        // 同步丢弃其他通道
        for ch in 1..<channelCount {
            if channelBuffers[ch].count >= CHUNK_SIZE {
                channelBuffers[ch].removeFirst(CHUNK_SIZE)
            }
        }

        let payload = chunk.withUnsafeBufferPointer { ptr -> Data in
            Data(bytes: ptr.baseAddress!, count: ptr.count * MemoryLayout<Float>.stride)
        }
        sendMessage(MSG_MIC, payload)
    }
    bufLock.unlock()
}

func setup() -> Bool {
    engine = AVAudioEngine()
    let inputNode = engine.inputNode

    // VP 启用
    do {
        try inputNode.setVoiceProcessingEnabled(true)
        log("VP 启用成功")

        // 禁用 ducking（关键：防止 VP 压低系统输出音量）
        if #available(macOS 14.0, *) {
            let duckingConfig = AVAudioVoiceProcessingOtherAudioDuckingConfiguration(
                enableAdvancedDucking: false,
                duckingLevel: .min
            )
            inputNode.voiceProcessingOtherAudioDuckingConfiguration = duckingConfig
            log("Ducking 已禁用")
        }
    } catch {
        log("VP 启用失败: \(error)")
    }

    let fmt = inputNode.inputFormat(forBus: 0)
    log("输入: \(fmt.sampleRate)Hz, \(fmt.channelCount)ch")

    // Tap（无播放节点，不会冲突）
    inputNode.installTap(onBus: 0, bufferSize: TAP_BUFFER_SIZE, format: fmt) { buffer, time in
        onTap(buffer, time)
    }

    do {
        try engine.start()
        log("Engine 启动成功")
        return true
    } catch {
        log("Engine 失败: \(error)")
        return false
    }
}

log("cc_audio_bridge 启动")
guard setup() else { exit(1) }
sendMessage(MSG_READY)
log("就绪")

let stdin = FileHandle.standardInput
while isRunning {
    let data = stdin.readData(ofLength: 5)
    if data.isEmpty { break }
    if data.count >= 1 && data[0] == MSG_EXIT { break }
}

engine.inputNode.removeTap(onBus: 0)
engine.stop()
log("退出")
