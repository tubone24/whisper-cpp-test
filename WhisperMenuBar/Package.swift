// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "WhisperMenuBar",
    platforms: [
        .macOS(.v12)
    ],
    targets: [
        .executableTarget(
            name: "WhisperMenuBar",
            path: "Sources"
        )
    ]
)
