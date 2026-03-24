import Foundation
import CoreML

/// Input configuration matching the BraTS 2020 training pipeline.
public enum BraTSConfig {
    /// Number of MRI modalities (FLAIR, T1, T1ce, T2).
    public static let channelCount = 4
    /// Spatial dimensions the model expects.
    public static let depth = 128
    public static let height = 128
    public static let width = 128
    /// Total element count for one input tensor.
    public static let inputElementCount = channelCount * depth * height * width
    /// Segmentation class names in output order.
    public static let classLabels = ["background", "NCR/NET", "edema", "enhancing"]
}

/// Z-score normalization matching the Python training code:
/// for each channel, normalize only the nonzero voxels.
public struct VolumeNormalizer {

    private init() {}

    /// Normalize a flat array of `channelCount * D * H * W` float values.
    /// Each channel is normalized independently over its nonzero voxels.
    /// Modifies `data` in place and returns it.
    @discardableResult
    public static func zScoreNormalize(
        _ data: inout [Float],
        channels: Int = BraTSConfig.channelCount,
        spatialCount: Int = BraTSConfig.depth * BraTSConfig.height * BraTSConfig.width
    ) -> [Float] {
        precondition(
            data.count == channels * spatialCount,
            "Expected \(channels * spatialCount) elements, got \(data.count)"
        )

        for ch in 0..<channels {
            let offset = ch * spatialCount

            var sum: Double = 0
            var sumSq: Double = 0
            var nonzeroCount = 0

            for i in 0..<spatialCount {
                let val = data[offset + i]
                if val != 0 {
                    let d = Double(val)
                    sum += d
                    sumSq += d * d
                    nonzeroCount += 1
                }
            }

            guard nonzeroCount > 0 else { continue }

            let mean = sum / Double(nonzeroCount)
            let variance = (sumSq / Double(nonzeroCount)) - (mean * mean)
            let std = (variance.squareRoot()) + 1e-8

            for i in 0..<spatialCount {
                if data[offset + i] != 0 {
                    data[offset + i] = Float((Double(data[offset + i]) - mean) / std)
                }
            }
        }

        return data
    }
}

/// Creates an `MLMultiArray` from a flat Float array in the shape the model expects.
public struct MultiArrayBuilder {

    private init() {}

    /// Build a `[1, 4, 128, 128, 128]` MLMultiArray from flat float data.
    /// The data must already be in CDHW order.
    public static func buildInput(from data: [Float]) throws -> MLMultiArray {
        precondition(
            data.count == BraTSConfig.inputElementCount,
            "Expected \(BraTSConfig.inputElementCount) elements, got \(data.count)"
        )

        let shape: [NSNumber] = [1, 4, 128, 128, 128]
        let array = try MLMultiArray(shape: shape, dataType: .float32)

        let ptr = array.dataPointer.bindMemory(
            to: Float.self,
            capacity: data.count
        )
        data.withUnsafeBufferPointer { buf in
            ptr.update(from: buf.baseAddress!, count: buf.count)
        }

        return array
    }
}
