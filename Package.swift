// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "BraTSCoreML",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "BraTSCoreML",
            targets: ["BraTSCoreML"]
        ),
    ],
    targets: [
        .target(
            name: "BraTSCoreML",
            path: "Sources/BraTSCoreML"
        ),
        .testTarget(
            name: "BraTSCoreMLTests",
            dependencies: ["BraTSCoreML"],
            path: "Tests/BraTSCoreMLTests"
        ),
    ]
)