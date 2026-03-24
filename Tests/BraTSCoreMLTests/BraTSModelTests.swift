import XCTest
@testable import BraTSCoreML

final class BraTSModelTests: XCTestCase {

    func testZScoreNormalizationOnNonzeroVoxels() {
        let spatialCount = 8
        let channels = 2
        var data: [Float] = [
            0, 1, 2, 3, 0, 0, 0, 0,
            5, 5, 5, 5, 5, 5, 5, 5,
        ]

        VolumeNormalizer.zScoreNormalize(
            &data, channels: channels, spatialCount: spatialCount
        )

        XCTAssertEqual(data[0], 0, accuracy: 1e-6)
        XCTAssertEqual(data[4], 0, accuracy: 1e-6)

        XCTAssertEqual(data[1], -1.2247, accuracy: 0.01)
        XCTAssertEqual(data[2], 0.0, accuracy: 0.01)
        XCTAssertEqual(data[3], 1.2247, accuracy: 0.01)

        for i in 8..<16 {
            XCTAssertEqual(data[i], 0.0, accuracy: 0.01)
        }
    }

    func testZScoreNormalizationAllZeros() {
        var data: [Float] = [Float](repeating: 0, count: 16)
        VolumeNormalizer.zScoreNormalize(&data, channels: 2, spatialCount: 8)

        for val in data {
            XCTAssertFalse(val.isNaN)
            XCTAssertEqual(val, 0)
        }
    }

    func testMultiArrayBuilderShape() throws {
        let data = [Float](repeating: 0, count: BraTSConfig.inputElementCount)
        let array = try MultiArrayBuilder.buildInput(from: data)

        XCTAssertEqual(array.shape.count, 5)
        XCTAssertEqual(array.shape[0], 1)
        XCTAssertEqual(array.shape[1], 4)
        XCTAssertEqual(array.shape[2], 128)
        XCTAssertEqual(array.shape[3], 128)
        XCTAssertEqual(array.shape[4], 128)
    }

    func testMultiArrayBuilderDataIntegrity() throws {
        var data = [Float](repeating: 0, count: BraTSConfig.inputElementCount)
        data[0] = 42.0
        data[BraTSConfig.inputElementCount - 1] = -7.5

        let array = try MultiArrayBuilder.buildInput(from: data)
        let ptr = array.dataPointer.bindMemory(
            to: Float.self, capacity: data.count
        )

        XCTAssertEqual(ptr[0], 42.0)
        XCTAssertEqual(ptr[BraTSConfig.inputElementCount - 1], -7.5)
    }

    func testRealModelLoadAndPredict() throws {
        let testFile = URL(fileURLWithPath: #file)
        let repoRoot = testFile
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()

        let mlpackageURL = repoRoot
            .appendingPathComponent("models")
            .appendingPathComponent("brats_unet3d_int4.mlpackage")

        guard FileManager.default.fileExists(atPath: mlpackageURL.path) else {
            throw XCTSkip("brats_unet3d_int4.mlpackage not found, skipping integration test")
        }

        let segmenter = try BraTSSegmenter(mlpackageURL: mlpackageURL)

        var randomData = (0..<BraTSConfig.inputElementCount).map { _ in Float.random(in: -1...1) }
        VolumeNormalizer.zScoreNormalize(&randomData)

        let result = try segmenter.predict(data: randomData)

        XCTAssertEqual(result.logits.shape.count, 5)
        XCTAssertEqual(result.logits.shape[0], 1)
        XCTAssertEqual(result.logits.shape[1], 4)
        XCTAssertEqual(result.logits.shape[2], 128)
        XCTAssertEqual(result.logits.shape[3], 128)
        XCTAssertEqual(result.logits.shape[4], 128)

        let labels = result.labelMap
        let expectedCount = BraTSConfig.depth * BraTSConfig.height * BraTSConfig.width
        XCTAssertEqual(labels.count, expectedCount)

        for label in labels {
            XCTAssertTrue(label >= 0 && label < 4, "Label \(label) out of range [0,3]")
        }

        let counts = result.classCounts
        let total = counts.values.reduce(0, +)
        XCTAssertEqual(total, expectedCount)
    }
}