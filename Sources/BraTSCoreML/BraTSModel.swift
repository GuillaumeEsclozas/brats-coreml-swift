import Foundation
import CoreML

/// Wraps a CoreML 3D U-Net for brain lesion segmentation.
///
/// The model takes a 5D tensor `[1, 4, 128, 128, 128]` and returns
/// raw logits with the same shape. Call `argmax` on the channel
/// dimension to get the segmentation label map.
public struct BraTSSegmenter {

    private let model: MLModel

    /// The output feature name. CoreML sometimes renames outputs internally,
    /// so we read it from the spec rather than hardcoding.
    private let outputName: String

    /// Load a CoreML model from a compiled `.mlmodelc` directory
    /// or an `.mlpackage` path.
    public init(compiledModelURL: URL) throws {
        self.model = try MLModel(contentsOf: compiledModelURL)

        guard let firstOutput = model.modelDescription.outputDescriptionsByName.keys.first else {
            fatalError("Model has no outputs. Check the mlpackage.")
        }
        self.outputName = firstOutput
    }

    /// Load from an `.mlpackage` by compiling it at runtime.
    /// Slower on first call (compilation), but convenient for testing.
    public init(mlpackageURL: URL) throws {
        let compiled = try MLModel.compileModel(at: mlpackageURL)
        try self.init(compiledModelURL: compiled)
    }

    /// Run segmentation on a preprocessed input tensor.
    ///
    /// - Parameter input: `MLMultiArray` of shape `[1, 4, 128, 128, 128]`,
    ///   z-score normalized per channel on nonzero voxels.
    /// - Returns: A `SegmentationResult` containing the raw logits and
    ///   the argmax label map.
    public func predict(input: MLMultiArray) throws -> SegmentationResult {
        let provider = try MLDictionaryFeatureProvider(
            dictionary: ["input": MLFeatureValue(multiArray: input)]
        )

        let prediction = try model.prediction(from: provider)

        guard let outputValue = prediction.featureValue(for: outputName),
              let outputArray = outputValue.multiArrayValue else {
            throw BraTSError.missingOutput(name: outputName)
        }

        return SegmentationResult(logits: outputArray)
    }

    /// Convenience: run segmentation from raw float data.
    ///
    /// - Parameter data: Flat float array of length `4 * 128 * 128 * 128`,
    ///   already in CDHW order and z-score normalized.
    /// - Returns: A `SegmentationResult`.
    public func predict(data: [Float]) throws -> SegmentationResult {
        let input = try MultiArrayBuilder.buildInput(from: data)
        return try predict(input: input)
    }
}

/// Result of a segmentation prediction.
public struct SegmentationResult {
    /// Raw model output, shape `[1, 4, 128, 128, 128]`.
    public let logits: MLMultiArray

    /// Argmax over the class dimension (axis 1), producing a flat
    /// array of Int32 labels of length `128 * 128 * 128`.
    /// Labels: 0=background, 1=NCR/NET, 2=edema, 3=enhancing.
    public var labelMap: [Int32] {
        let spatialCount = BraTSConfig.depth * BraTSConfig.height * BraTSConfig.width
        let classCount = BraTSConfig.channelCount
        var labels = [Int32](repeating: 0, count: spatialCount)

        let ptr = logits.dataPointer.bindMemory(
            to: Float.self,
            capacity: classCount * spatialCount
        )

        for voxel in 0..<spatialCount {
            var bestClass: Int32 = 0
            var bestValue = ptr[voxel]

            for c in 1..<classCount {
                let value = ptr[c * spatialCount + voxel]
                if value > bestValue {
                    bestValue = value
                    bestClass = Int32(c)
                }
            }
            labels[voxel] = bestClass
        }

        return labels
    }

    /// Count of voxels per class in the label map.
    public var classCounts: [String: Int] {
        let labels = labelMap
        var counts = [String: Int]()
        for (i, name) in BraTSConfig.classLabels.enumerated() {
            counts[name] = labels.filter { $0 == Int32(i) }.count
        }
        return counts
    }
}

/// Errors specific to BraTS model operations.
public enum BraTSError: Error, CustomStringConvertible {
    case missingOutput(name: String)

    public var description: String {
        switch self {
        case .missingOutput(let name):
            return "Model did not produce expected output '\(name)'. Check the mlpackage spec."
        }
    }
}
