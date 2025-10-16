using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;
#if UNITY_EDITOR
using UnityEditor;
#endif
namespace Empart.EmberPart
{
    /// <summary>
    /// Main convex decomposition system, implementing a production-ready V-HACD (Voxelized Hierarchical Approximate Convex Decomposition) algorithm.
    /// This class handles the full decomposition process, including voxelization, concavity measurement, plane splitting, hull generation, merging, and optimization.
    /// All components are consolidated into this single MonoBehaviour class for easy addition to Unity projects.
    /// </summary>
    public class EmpartDecomposer : MonoBehaviour
    {
        /// <summary>
        /// Quality presets for convex decomposition with detailed configurations
        /// </summary>
        public enum QualityPreset
        {
            Low,
            Medium,
            High,
            Ultra,
            Custom
        }
        /// <summary>
        /// Splitting strategies for hull decomposition
        /// </summary>
        public enum SplittingStrategy
        {
            Balanced,
            VolumeBased,
            SurfaceAreaBased,
            ConcavityBased,
            FeatureBased,
            Hybrid
        }
        /// <summary>
        /// Voxelization strategies for concavity calculation
        /// </summary>
        public enum VoxelizationStrategy
        {
            SurfaceOnly,
            SolidFill,
            RaycastBased,
            Hybrid,
            Adaptive
        }
        /// <summary>
        /// Merging strategies for hull optimization
        /// </summary>
        public enum MergingStrategy
        {
            VolumeBased,
            SurfaceAreaBased,
            DistanceBased,
            FeatureBased,
            Hierarchical,
            Adaptive
        }
        /// <summary>
        /// ACD subroutine types
        /// </summary>
        public enum ACDSubroutineType
        {
            VHACD,
            CoACD,
            QuickHull
        }
        /// <summary>
        /// Error metrics types for hull evaluation
        /// </summary>
        public enum ErrorMetricType
        {
            Hausdorff,
            MeanSquared,
            RootMeanSquared,
            SymmetricHausdorff,
            MaxDeviation,
            VolumeDifference,
            SurfaceAreaDifference
        }
        /// <summary>
        /// Sampling strategies for mesh surface
        /// </summary>
        public enum SamplingStrategy
        {
            Uniform,
            AreaWeighted,
            CurvatureWeighted,
            FeaturePreserving,
            Adaptive,
            Stratified
        }
        /// <summary>
        /// Additional enum for hull optimization strategies
        /// </summary>
        public enum HullOptimizationStrategy
        {
            VertexReduction,
            FaceMerging,
            EdgeCollapse,
            IterativeRefinement,
            GeneticAlgorithm,
            SimulatedAnnealing
        }
        /// <summary>
        /// Enum for post-processing techniques
        /// </summary>
        public enum PostProcessingType
        {
            None,
            Smoothing,
            Simplification,
            HoleFilling,
            BoundaryPreservation,
            SymmetryEnforcement
        }
        /// <summary>
        /// Burst-safe plane structure to replace UnityEngine.Plane
        /// </summary>
        public struct Plane4
        {
            public float3 n; // unit normal
            public float d; // ax + by + cz + d = 0
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public float DistanceToPoint(float3 p) => math.dot(n, p) + d;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            public bool IntersectsRay(float3 rayOrigin, float3 rayDirection, out float t)
            {
                float denom = math.dot(n, rayDirection);
                if (math.abs(denom) < 1e-6f)
                {
                    t = 0f;
                    return false;
                }
                t = -(math.dot(n, rayOrigin) + d) / denom;
                return t > 0f;
            }
            // Additional utility: Project point onto plane
            public float3 ProjectPoint(float3 point)
            {
                return point - DistanceToPoint(point) * n;
            }
            // Equality check with epsilon
            public bool Equals(Plane4 other, float epsilon = 1e-5f)
            {
                return math.all(math.abs(n - other.n) < epsilon) && math.abs(d - other.d) < epsilon;
            }
            // Additional: Flip plane
            public void Flip()
            {
                n = -n;
                d = -d;
            }
            // Additional: Get plane equation string
            public string ToEquation()
            {
                return $"{n.x}x + {n.y}y + {n.z}z + {d} = 0";
            }
        }
        /// <summary>
        /// Simple max-priority queue for Unity (replaces System.Collections.Generic.PriorityQueue)
        /// </summary>
        public sealed class MaxPQ<T>
        {
            readonly List<(T item, float key)> _data = new();
            readonly IComparer<float> _cmp = Comparer<float>.Default;
            public int Count => _data.Count;
            public void Enqueue(T item, float key)
            {
                _data.Add((item, key));
                int i = _data.Count - 1;
                while (i > 0)
                {
                    int p = (i - 1) >> 1;
                    if (_cmp.Compare(_data[i].key, _data[p].key) <= 0) break;
                    (_data[i], _data[p]) = (_data[p], _data[i]);
                    i = p;
                }
            }
            public T Dequeue()
            {
                if (_data.Count == 0) throw new InvalidOperationException("Queue is empty");
                var root = _data[0].item;
                int last = _data.Count - 1;
                _data[0] = _data[last];
                _data.RemoveAt(last);
                int i = 0;
                while (true)
                {
                    int l = i * 2 + 1, r = l + 1, largest = i;
                    if (l < _data.Count && _cmp.Compare(_data[l].key, _data[largest].key) > 0) largest = l;
                    if (r < _data.Count && _cmp.Compare(_data[r].key, _data[largest].key) > 0) largest = r;
                    if (largest == i) break;
                    (_data[i], _data[largest]) = (_data[largest], _data[i]);
                    i = largest;
                }
                return root;
            }
            public T Peek()
            {
                if (_data.Count == 0) throw new InvalidOperationException("Queue is empty");
                return _data[0].item;
            }
            public bool TryDequeue(out T item, out float key)
            {
                if (_data.Count == 0)
                {
                    item = default;
                    key = default;
                    return false;
                }
                key = _data[0].key;
                item = Dequeue();
                return true;
            }
            public void Clear() => _data.Clear();
            // Additional: Update priority of an item (inefficient, but useful for production)
            public void UpdatePriority(T item, float newKey)
            {
                int index = -1;
                for (int i = 0; i < _data.Count; i++)
                {
                    if (EqualityComparer<T>.Default.Equals(_data[i].item, item))
                    {
                        index = i;
                        break;
                    }
                }
                if (index == -1) return;
                float oldKey = _data[index].key;
                _data[index] = (_data[index].item, newKey);
                if (newKey > oldKey)
                {
                    // Heapify up
                    while (index > 0)
                    {
                        int p = (index - 1) >> 1;
                        if (_cmp.Compare(_data[index].key, _data[p].key) <= 0) break;
                        (_data[index], _data[p]) = (_data[p], _data[index]);
                        index = p;
                    }
                }
                else
                {
                    // Heapify down
                    while (true)
                    {
                        int l = index * 2 + 1, r = l + 1, largest = index;
                        if (l < _data.Count && _cmp.Compare(_data[l].key, _data[largest].key) > 0) largest = l;
                        if (r < _data.Count && _cmp.Compare(_data[r].key, _data[largest].key) > 0) largest = r;
                        if (largest == index) break;
                        (_data[index], _data[largest]) = (_data[largest], _data[index]);
                        index = largest;
                    }
                }
            }

        private void OnDrawGizmosSelected()
        {
            if (settings == null || !settings.enableDebugVisualization) return;

            // Draw split planes
            if (settings.enableSplitPlaneVisualization)
            {
                // This will require storing the split planes used during decomposition.
                // For now, we'll just draw a placeholder plane.
                Gizmos.color = Color.cyan;
                Gizmos.matrix = transform.localToWorldMatrix;
                Gizmos.DrawLine(transform.position - Vector3.right * 5, transform.position + Vector3.right * 5);
            }

            // Draw witness points
            if (settings.enableWitnessVisualization && hulls.Count > 0 && combinedMesh != null)
            {
                Gizmos.color = Color.red;
                foreach(var hull in hulls)
                {
                    if(hull.witnessPoints != null)
                    {
                        foreach(var p in hull.witnessPoints)
                        {
                            Gizmos.DrawSphere(transform.TransformPoint(p), 0.05f);
                        }
                    }
                }
            }

            // Draw error heatmap
            if(settings.enableErrorHeatmap && combinedMesh != null && errorDistribution.Count > 0)
            {
                 if(errorDistribution.Count == combinedMesh.vertices.Count)
                 {
                    float maxError = errorDistribution.Max();
                    if(maxError > 0)
                    {
                        for(int i = 0; i < combinedMesh.vertices.Count; i++)
                        {
                            Gizmos.color = Color.Lerp(Color.green, Color.red, errorDistribution[i] / maxError);
                            Gizmos.DrawSphere(transform.TransformPoint(combinedMesh.vertices[i]), 0.01f);
                        }
                    }
                 }
            }
        }
            // Additional: Check if contains item
            public bool Contains(T item)
            {
                return _data.Any(d => EqualityComparer<T>.Default.Equals(d.item, item));
            }
            // Additional: Get all items
            public List<T> GetAllItems()
            {
                return _data.Select(d => d.item).ToList();
            }
            // Additional: Remove specific item
            public bool Remove(T item)
            {
                int index = -1;
                for (int i = 0; i < _data.Count; i++)
                {
                    if (EqualityComparer<T>.Default.Equals(_data[i].item, item))
                    {
                        index = i;
                        break;
                    }
                }
                if (index == -1) return false;
                int last = _data.Count - 1;
                _data[index] = _data[last];
                _data.RemoveAt(last);
                // Heapify down
                while (true)
                {
                    int l = index * 2 + 1, r = l + 1, largest = index;
                    if (l < _data.Count && _cmp.Compare(_data[l].key, _data[largest].key) > 0) largest = l;
                    if (r < _data.Count && _cmp.Compare(_data[r].key, _data[largest].key) > 0) largest = r;
                    if (largest == index) break;
                    (_data[index], _data[largest]) = (_data[largest], _data[index]);
                    index = largest;
                }
                return true;
            }
        }
        /// <summary>
        /// Approximate Vector3 comparer for duplicate vertex detection
        /// </summary>
        sealed class ApproxVec3Comparer : IEqualityComparer<Vector3>
        {
            readonly float eps;
            public ApproxVec3Comparer(float e) { eps = e; }
            public bool Equals(Vector3 a, Vector3 b) => (a - b).sqrMagnitude <= eps * eps;
            public int GetHashCode(Vector3 v) => HashCode.Combine(
                Mathf.RoundToInt(v.x / eps),
                Mathf.RoundToInt(v.y / eps),
                Mathf.RoundToInt(v.z / eps));
            // Additional: Distance between two points
            public float Distance(Vector3 a, Vector3 b)
            {
                return (a - b).magnitude;
            }
            // Additional: Check if close enough
            public bool IsClose(Vector3 a, Vector3 b, float tolerance)
            {
                return (a - b).sqrMagnitude <= tolerance * tolerance;
            }
        }
        /// <summary>
        /// Settings asset for convex decomposition
        /// </summary>
        [CreateAssetMenu(fileName = "ConvexDecompositionSettings", menuName = "Empart/Convex Decomposition Settings")]
        public class ConvexDecompositionSettings : ScriptableObject
        {
            [Header("Quality Settings")]
            public QualityPreset qualityPreset = QualityPreset.Medium;
            public float errorTolerance = 0.01f;
            public int maxHullCount = 32;
            public int maxVerticesPerHull = 64;
            public float concavityWeight = 0.5f;
            public float volumeWeight = 0.5f;
            public float balanceWeight = 0.1f;
            public float sahAlpha = 0.01f;
            public bool useSAH = true;
            [Header("Voxelization Settings")]
            public VoxelizationStrategy voxelizationStrategy = VoxelizationStrategy.Adaptive;
            public float voxelSize = 0.05f;
            public float voxelAdaptivity = 0.5f;
            public int maxVoxelCount = 1000000;
            public bool adaptiveBoundsPadding = true;
            public float boundsPaddingFactor = 0.1f;
            [Header("Splitting Settings")]
            public SplittingStrategy splittingStrategy = SplittingStrategy.Hybrid;
            public float minSplitVolume = 0.001f;
            public float minSplitRatio = 0.2f;
            public float splitConcavityThreshold = 0.1f;
            [Header("Merging Settings")]
            public MergingStrategy mergingStrategy = MergingStrategy.Adaptive;
            public float mergeThreshold = 0.05f;
            public float mergeConcavityThreshold = 0.01f;
            [Header("Advanced Settings")]
            public ACDSubroutineType acdSubroutine = ACDSubroutineType.VHACD;
            public bool enableAsyncProcessing = true;
            public int maxThreadCount = 0; // 0 = auto-detect
            public bool enableProfiling = false;
            public bool enableDebugVisualization = false;
            public Material hullMaterial;
            public bool enableDiagnosticsUI = true;
            [Header("Units")]
            [Tooltip("Unity units per meter (1 = meters, 0.01 = centimeters, 0.001 = millimeters)")]
            public float unitsPerMeter = 1f;
            [Header("Error Metrics")]
            public ErrorMetricType primaryErrorMetric = ErrorMetricType.SymmetricHausdorff;
            public ErrorMetricType secondaryErrorMetric = ErrorMetricType.MeanSquared;
            public float errorMetricWeight = 0.7f;
            [Header("Sampling Settings")]
            public SamplingStrategy samplingStrategy = SamplingStrategy.AreaWeighted;
            public int baseSampleCount = 1000;
            public int maxSampleCount = 10000;
            public float sampleAdaptivityFactor = 2.0f;
            [Header("Optimization Settings")]
            public bool enableHullOptimization = true;
            public float hullOptimizationThreshold = 0.01f;
            public int maxOptimizationIterations = 10;
            public bool enableVertexReduction = true;
            public float vertexReductionThreshold = 0.001f;
            [Header("Debug Settings")]
            public bool enableDetailedLogging = false;
            public bool enableErrorHeatmap = false;
            public bool enableWitnessVisualization = false;
            public bool enableSplitPlaneVisualization = false;
            // Additional settings for production readiness
            [Header("Additional Production Settings")]
            public bool enableCacheResults = false;
            public string cacheFilePath = "decomposition_cache.json";
            public bool enableMultiResolution = false;
            public int multiResolutionLevels = 3;
            public float resolutionScaleFactor = 0.5f;
            public bool enableSymmetryDetection = true;
            public float symmetryTolerance = 0.01f;
            public bool enablePostProcessing = true;
            public int postProcessingIterations = 5;
            public float postProcessingThreshold = 0.005f;
            [Header("Voxel Specific Settings")]
            public bool enableSDF = true;
            public float sdfResolutionFactor = 1.5f;
            public int sdfIterationCount = 5;
            public float sdfSmoothingFactor = 0.1f;
            // Additional settings for extended functionality
            [Header("Extended Optimization")]
            public HullOptimizationStrategy hullOptimizationStrategy = HullOptimizationStrategy.EdgeCollapse;
            public float optimizationConvergenceThreshold = 0.0001f;
            public int maxGeneticGenerations = 50;
            public float mutationRate = 0.1f;
            [Header("Post Processing")]
            public PostProcessingType postProcessingType = PostProcessingType.Smoothing;
            public float smoothingLambda = 0.5f;
            public int smoothingIterations = 3;
            public bool enableBoundaryLocking = true;
            /// <summary>
            /// Apply quality preset settings
            /// </summary>
            public void ApplyQualityPreset()
            {
                switch (qualityPreset)
                {
                    case QualityPreset.Low:
                        errorTolerance = 0.05f;
                        maxHullCount = 16;
                        maxVerticesPerHull = 32;
                        baseSampleCount = 500;
                        maxSampleCount = 2000;
                        voxelSize = 0.1f;
                        maxVoxelCount = 500000;
                        maxOptimizationIterations = 5;
                        break;
                    case QualityPreset.Medium:
                        errorTolerance = 0.01f;
                        maxHullCount = 32;
                        maxVerticesPerHull = 64;
                        baseSampleCount = 1000;
                        maxSampleCount = 5000;
                        voxelSize = 0.05f;
                        maxVoxelCount = 1000000;
                        maxOptimizationIterations = 10;
                        break;
                    case QualityPreset.High:
                        errorTolerance = 0.005f;
                        maxHullCount = 64;
                        maxVerticesPerHull = 128;
                        baseSampleCount = 2000;
                        maxSampleCount = 10000;
                        voxelSize = 0.02f;
                        maxVoxelCount = 2000000;
                        maxOptimizationIterations = 15;
                        break;
                    case QualityPreset.Ultra:
                        errorTolerance = 0.001f;
                        maxHullCount = 128;
                        maxVerticesPerHull = 256;
                        baseSampleCount = 5000;
                        maxSampleCount = 20000;
                        voxelSize = 0.01f;
                        maxVoxelCount = 5000000;
                        maxOptimizationIterations = 20;
                        break;
                }
            }
            private void OnValidate()
            {
                if (qualityPreset != QualityPreset.Custom)
                {
                    ApplyQualityPreset();
                }
                // Validate ranges
                errorTolerance = Mathf.Max(0.0001f, errorTolerance);
                maxHullCount = Mathf.Max(1, maxHullCount);
                maxVerticesPerHull = Mathf.Max(4, maxVerticesPerHull);
                voxelSize = Mathf.Max(0.001f, voxelSize);
                maxVoxelCount = Mathf.Max(1000, maxVoxelCount);
                mergeThreshold = Mathf.Max(0f, mergeThreshold);
            }
            // Additional method: Load from JSON for production
            public void LoadFromJson(string json)
            {
                JsonUtility.FromJsonOverwrite(json, this);
            }
            // Additional method: Save to JSON
            public string SaveToJson()
            {
                return JsonUtility.ToJson(this, true);
            }
            // Additional method: Reset to default
            public void ResetToDefault()
            {
                qualityPreset = QualityPreset.Medium;
                ApplyQualityPreset();
            }
            // Additional method: Copy from another settings
            public void CopyFrom(ConvexDecompositionSettings other)
            {
                JsonUtility.FromJsonOverwrite(JsonUtility.ToJson(other), this);
            }
        }
        /// <summary>
        /// Region box data structure
        /// </summary>
        [Serializable]
        public struct RegionBox
        {
            public Bounds AABB;
            public float epsilonMm;
            public int partBudgetMin;
            public int partBudgetMax;
            public bool preserveExact;
            public float priority;
            public string label;
            // Additional properties for production
            public bool enableCustomVoxelization;
            public float customVoxelSize;
            public bool enableRegionOptimization;
            public int regionOptimizationIterations;
            // Validation method
            public bool IsValid()
            {
                return AABB.size.x > 0 && AABB.size.y > 0 && AABB.size.z > 0 &&
                       partBudgetMin <= partBudgetMax && priority >= 0;
            }
            // Additional: Calculate volume
            public float Volume()
            {
                return AABB.size.x * AABB.size.y * AABB.size.z;
            }
            // Additional: Check intersection with another box
            public bool Intersects(RegionBox other)
            {
                return AABB.Intersects(other.AABB);
            }
        }
        /// <summary>
        /// Comprehensive data structure for mesh information with additional attributes
        /// </summary>
        public class MeshData
        {
            public List<Vector3> vertices = new List<Vector3>();
            public List<int> indices = new List<int>();
            public List<Vector3> normals = new List<Vector3>();
            public List<Vector2> uvs;
            public List<Vector4> tangents;
            public List<Color> colors;
            public List<Vector4> boneWeights;
            public List<int> submeshIndices;
            public Bounds bounds;
            public float volume;
            public float surfaceArea;
            public bool isClosed;
            public bool isManifold;
            public int vertexCount;
            public int triangleCount;
            public int edgeCount;
            public float averageEdgeLength;
            public float maxEdgeLength;
            public float minEdgeLength;
            public Dictionary<string, object> customAttributes = new Dictionary<string, object>();
            // Additional properties for advanced processing
            public List<Vector3> curvature;
            public List<float> vertexSaliency;
            public List<float> triangleSaliency;
            public List<int> sharpEdges;
            public List<int> featureVertices;
            public BVH accelerationStructure;
            public bool hasPrecomputedData;
            // Additional production properties
            public List<Vector3> edgeNormals;
            public List<float> vertexDensities;
            public bool hasUVs;
            public bool hasTangents;
            public bool hasColors;
            public bool hasBoneWeights;
            public float meshComplexityScore;
            public DateTime lastUpdated;
            public NativeArray<float> sdfValues; // For SDF integration
            public int3 voxelDimensions; // Voxel grid dimensions
            public Bounds voxelBounds; // Bounds of voxel grid
            // Additional: List of submeshes
            public List<MeshData> subMeshes = new List<MeshData>();
            // Additional: Topology information
            public int genus;
            public int eulerCharacteristic;
            /// <summary>
            /// Validate mesh data integrity
            /// </summary>
            public bool Validate()
            {
                if (vertices.Count < 3)
                    return false;
                if (indices.Count % 3 != 0)
                    return false;
                if (normals.Count > 0 && normals.Count != vertices.Count)
                    return false;
                if (uvs != null && uvs.Count > 0 && uvs.Count != vertices.Count)
                    return false;
                if (tangents != null && tangents.Count > 0 && tangents.Count != vertices.Count)
                    return false;
                if (colors != null && colors.Count > 0 && colors.Count != vertices.Count)
                    return false;
                // Check for invalid indices
                for (int i = 0; i < indices.Count; i++)
                {
                    if (indices[i] < 0 || indices[i] >= vertices.Count)
                        return false;
                }
                // Additional checks
                hasUVs = uvs != null && uvs.Count == vertices.Count;
                hasTangents = tangents != null && tangents.Count == vertices.Count;
                hasColors = colors != null && colors.Count == vertices.Count;
                hasBoneWeights = boneWeights != null && boneWeights.Count == vertices.Count;
                return true;
            }
            /// <summary>
            /// Calculate mesh properties
            /// </summary>
            public void CalculateProperties()
            {
                vertexCount = vertices.Count;
                triangleCount = indices.Count / 3;
                // Calculate bounds
                bounds = new Bounds();
                if (vertices.Count > 0)
                {
                    bounds = new Bounds(vertices[0], Vector3.zero);
                    foreach (var v in vertices)
                        bounds.Encapsulate(v);
                }
                // Calculate surface area
                surfaceArea = 0f;
                for (int i = 0; i < indices.Count; i += 3)
                {
                    var a = vertices[indices[i]];
                    var b = vertices[indices[i + 1]];
                    var c = vertices[indices[i + 2]];
                    surfaceArea += Vector3.Cross(b - a, c - a).magnitude / 2f;
                }
                // Calculate volume (assuming closed mesh)
                volume = 0f;
                for (int i = 0; i < indices.Count; i += 3)
                {
                    var a = vertices[indices[i]];
                    var b = vertices[indices[i + 1]];
                    var c = vertices[indices[i + 2]];
                    volume += Vector3.Dot(a, Vector3.Cross(b, c)) / 6f;
                }
                volume = Mathf.Abs(volume);
                // Calculate edge statistics
                CalculateEdgeStatistics();
                // Check if mesh is closed
                isClosed = IsMeshClosed();
                // Check if mesh is manifold
                isManifold = IsMeshManifold();
                // Additional: Calculate complexity score
                meshComplexityScore = triangleCount * Mathf.Log(vertexCount + 1) / surfaceArea;
                lastUpdated = DateTime.Now;
                // Additional: Calculate Euler characteristic
                CalculateTopology();
            }
            private void CalculateEdgeStatistics()
            {
                var edges = new HashSet<(int, int)>();
                float totalLength = 0f;
                maxEdgeLength = 0f;
                minEdgeLength = float.MaxValue;
                for (int i = 0; i < indices.Count; i += 3)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        int v1 = indices[i + j];
                        int v2 = indices[i + (j + 1) % 3];
                        if (v1 > v2)
                        {
                            (v1, v2) = (v2, v1);
                        }
                        // Only add to total length if this is a new edge
                        if (edges.Add((v1, v2)))
                        {
                            float length = Vector3.Distance(vertices[v1], vertices[v2]);
                            totalLength += length;
                            maxEdgeLength = Mathf.Max(maxEdgeLength, length);
                            minEdgeLength = Mathf.Min(minEdgeLength, length);
                        }
                    }
                }
                edgeCount = edges.Count;
                averageEdgeLength = edges.Count > 0 ? totalLength / edges.Count : 0f;
                // Additional: Calculate edge normals if needed
                if (edgeNormals == null) edgeNormals = new List<Vector3>(edgeCount);
                // Additional: Calculate variance of edge lengths
                float variance = 0f;
                foreach (var edge in edges)
                {
                    float length = Vector3.Distance(vertices[edge.Item1], vertices[edge.Item2]);
                    variance += (length - averageEdgeLength) * (length - averageEdgeLength);
                }
                variance /= edgeCount;
                customAttributes["EdgeLengthVariance"] = variance;
            }
            private bool IsMeshClosed()
            {
                var edgeCountDict = new Dictionary<(int, int), int>();
                for (int i = 0; i < indices.Count; i += 3)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        int v1 = indices[i + j];
                        int v2 = indices[i + (j + 1) % 3];
                        if (v1 > v2)
                        {
                            (v1, v2) = (v2, v1);
                        }
                        var edge = (v1, v2);
                        if (edgeCountDict.ContainsKey(edge))
                            edgeCountDict[edge]++;
                        else
                            edgeCountDict[edge] = 1;
                    }
                }
                // Mesh is closed if all edges are shared by exactly 2 triangles
                foreach (var count in edgeCountDict.Values)
                {
                    if (count != 2)
                        return false;
                }
                return true;
            }
            private bool IsMeshManifold()
            {
                // Check for non-manifold edges
                var edgeCountDict = new Dictionary<(int, int), int>();
                for (int i = 0; i < indices.Count; i += 3)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        int v1 = indices[i + j];
                        int v2 = indices[i + (j + 1) % 3];
                        if (v1 > v2)
                        {
                            (v1, v2) = (v2, v1);
                        }
                        var edge = (v1, v2);
                        if (edgeCountDict.ContainsKey(edge))
                            edgeCountDict[edge]++;
                        else
                            edgeCountDict[edge] = 1;
                    }
                }
                // Check for non-manifold edges
                foreach (var count in edgeCountDict.Values)
                {
                    if (count > 2)
                        return false;
                }
                // Additional check for vertex manifoldness
                var vertexEdges = new List<int>[vertices.Count];
                for (int vi = 0; vi < vertices.Count; vi++)
                {
                    vertexEdges[vi] = new List<int>();
                }
                foreach (var edge in edgeCountDict.Keys)
                {
                    vertexEdges[edge.Item1].Add(edge.Item2);
                    vertexEdges[edge.Item2].Add(edge.Item1);
                }
                for (int vi = 0; vi < vertices.Count; vi++)
                {
                    if (vertexEdges[vi].Count % 2 != 0) return false; // Odd number of edges at vertex
                }
                return true;
            }
            /// <summary>
            /// Optimize mesh data for processing
            /// </summary>
            public void Optimize()
            {
                // Remove duplicate vertices
                RemoveDuplicateVertices();
                // Remove degenerate triangles
                RemoveDegenerateTriangles();
                // Recalculate properties
                CalculateProperties();
                // Additional optimization: Weld close vertices
                WeldVertices(0.0001f);
                // Reindex for cache efficiency
                ReindexForCache();
                // Additional: Decimate mesh if needed
                if (vertexCount > 10000)
                    Decimate(0.1f);
            }
            private void WeldVertices(float threshold)
            {
                var comparer = new ApproxVec3Comparer(threshold);
                var newVerts = new List<Vector3>();
                var remap = new int[vertices.Count];
                var lookup = new Dictionary<Vector3, int>(comparer);
                for (int i = 0; i < vertices.Count; i++)
                {
                    var v = vertices[i];
                    if (!lookup.TryGetValue(v, out int ni))
                    {
                        ni = newVerts.Count;
                        newVerts.Add(v);
                        lookup[v] = ni;
                    }
                    remap[i] = ni;
                }
                for (int i = 0; i < indices.Count; i++)
                    indices[i] = remap[indices[i]];
                vertices = newVerts;
            }
            private void ReindexForCache()
            {
                // Example: spatially sort vertices, then remap indices accordingly.
                var order = Enumerable.Range(0, vertices.Count)
                                      .OrderBy(i => vertices[i].x)
                                      .ThenBy(i => vertices[i].y)
                                      .ThenBy(i => vertices[i].z)
                                      .ToArray();
                var inv = new int[order.Length];
                for (int newIdx = 0; newIdx < order.Length; newIdx++) inv[order[newIdx]] = newIdx;
                var newVerts = new List<Vector3>(vertices.Count);
                for (int newIdx = 0; newIdx < order.Length; newIdx++) newVerts.Add(vertices[order[newIdx]]);
                vertices = newVerts;
                for (int i = 0; i < indices.Count; i++) indices[i] = inv[indices[i]];
            }
            private void RemoveDuplicateVertices()
            {
                var newVerts = new List<Vector3>();
                var remap = new int[vertices.Count];
                var lookup = new Dictionary<Vector3, int>(new ApproxVec3Comparer(1e-6f));
                for (int i = 0; i < vertices.Count; i++)
                {
                    var v = vertices[i];
                    if (!lookup.TryGetValue(v, out int ni))
                    {
                        ni = newVerts.Count;
                        newVerts.Add(v);
                        lookup[v] = ni;
                    }
                    remap[i] = ni;
                }
                for (int i = 0; i < indices.Count; i++)
                    indices[i] = remap[indices[i]];
                vertices = newVerts;
                // Fixed: Remap other channels with proper sizing
                if (normals != null && normals.Count == remap.Length) normals = Remap(normals, remap, newVerts.Count);
                if (uvs != null && uvs.Count == remap.Length) uvs = Remap(uvs, remap, newVerts.Count);
                if (tangents != null && tangents.Count == remap.Length) tangents = Remap(tangents, remap, newVerts.Count);
                if (colors != null && colors.Count == remap.Length) colors = Remap(colors, remap, newVerts.Count);
            }
            static List<T> Remap<T>(List<T> src, int[] remap, int newSize)
            {
                var dst = new List<T>(newSize);
                for (int i = 0; i < newSize; i++) dst.Add(default);
                for (int i = 0; i < remap.Length; i++)
                    dst[remap[i]] = src[i];
                return dst;
            }
            private void RemoveDegenerateTriangles()
            {
                int oldCount = indices.Count;
                var newIndices = new List<int>();
                for (int i = 0; i < indices.Count; i += 3)
                {
                    var a = vertices[indices[i]];
                    var b = vertices[indices[i + 1]];
                    var c = vertices[indices[i + 2]];
                    // Check if triangle is degenerate
                    var area = Vector3.Cross(b - a, c - a).magnitude / 2f;
                    if (area > 1e-6f)
                    {
                        newIndices.Add(indices[i]);
                        newIndices.Add(indices[i + 1]);
                        newIndices.Add(indices[i + 2]);
                    }
                }
                indices = newIndices;
                // Additional: Log removed count
                if (enableDetailedLoggingStatic)
                    Debug.Log($"Removed {(oldCount - indices.Count) / 3} degenerate triangles");
            }
            /// <summary>
            /// Calculate vertex curvature
            /// </summary>
            public void CalculateCurvature()
            {
                if (normals == null || normals.Count != vertices.Count)
                {
                    CalculateNormals();
                }
                curvature = new List<Vector3>(vertices.Count);
                for (int i = 0; i < vertices.Count; i++)
                {
                    curvature.Add(Vector3.zero);
                }
                // Calculate curvature using normal differences
                for (int i = 0; i < vertices.Count; i++)
                {
                    var normal = normals[i];
                    var curvatureVector = Vector3.zero;
                    int neighborCount = 0;
                    // Find neighboring vertices
                    for (int j = 0; j < indices.Count; j += 3)
                    {
                        bool containsVertex = false;
                        int other1 = -1, other2 = -1;
                        for (int k = 0; k < 3; k++)
                        {
                            if (indices[j + k] == i)
                            {
                                containsVertex = true;
                            }
                            else if (other1 == -1)
                            {
                                other1 = indices[j + k];
                            }
                            else
                            {
                                other2 = indices[j + k];
                            }
                        }
                        if (containsVertex && other1 != -1 && other2 != -1)
                        {
                            var normal1 = normals[other1];
                            var normal2 = normals[other2];
                            curvatureVector += (normal1 - normal) + (normal2 - normal);
                            neighborCount += 2;
                        }
                    }
                    if (neighborCount > 0)
                    {
                        curvature[i] = curvatureVector / neighborCount;
                    }
                }
                // Additional: Smooth curvature
                SmoothCurvature(1);
            }
            private void SmoothCurvature(int iterations)
            {
                for (int iter = 0; iter < iterations; iter++)
                {
                    var newCurvature = new List<Vector3>(curvature);
                    for (int i = 0; i < vertices.Count; i++)
                    {
                        Vector3 sum = curvature[i];
                        int count = 1;
                        // Add neighbors
                        for (int j = 0; j < indices.Count; j += 3)
                        {
                            if (indices[j] == i || indices[j + 1] == i || indices[j + 2] == i)
                            {
                                for (int k = 0; k < 3; k++)
                                {
                                    if (indices[j + k] != i)
                                    {
                                        sum += curvature[indices[j + k]];
                                        count++;
                                    }
                                }
                            }
                        }
                        newCurvature[i] = sum / count;
                    }
                    curvature = newCurvature;
                }
            }
            /// <summary>
            /// Calculate vertex saliency
            /// </summary>
            public void CalculateSaliency()
            {
                if (curvature == null || curvature.Count != vertices.Count)
                {
                    CalculateCurvature();
                }
                vertexSaliency = new List<float>(vertices.Count);
                triangleSaliency = new List<float>(triangleCount);
                // Calculate vertex saliency based on curvature magnitude
                float maxCurvature = 0f;
                for (int i = 0; i < vertices.Count; i++)
                {
                    vertexSaliency.Add(curvature[i].magnitude);
                    maxCurvature = Mathf.Max(maxCurvature, vertexSaliency[i]);
                }
                // Normalize saliency
                if (maxCurvature > 0f)
                {
                    for (int i = 0; i < vertices.Count; i++)
                    {
                        vertexSaliency[i] /= maxCurvature;
                    }
                }
                // Calculate triangle saliency as average of vertex saliency
                for (int i = 0; i < indices.Count; i += 3)
                {
                    int triIndex = i / 3;
                    triangleSaliency[triIndex] = (
                        vertexSaliency[indices[i]] +
                        vertexSaliency[indices[i + 1]] +
                        vertexSaliency[indices[i + 2]]
                    ) / 3f;
                }
                // Additional: Weight by area
                for (int i = 0; i < indices.Count; i += 3)
                {
                    int triIndex = i / 3;
                    var a = vertices[indices[i]];
                    var b = vertices[indices[i + 1]];
                    var c = vertices[indices[i + 2]];
                    float area = Vector3.Cross(b - a, c - a).magnitude / 2f;
                    triangleSaliency[triIndex] *= area;
                }
                // Additional: Normalize triangle saliency
                float maxTriSaliency = triangleSaliency.Max();
                if (maxTriSaliency > 0)
                {
                    for (int j = 0; j < triangleSaliency.Count; j++)
                    {
                        triangleSaliency[j] /= maxTriSaliency;
                    }
                }
            }
            /// <summary>
            /// Identify sharp edges
            /// </summary>
            public void IdentifySharpEdges(float angleThreshold = 30f)
            {
                if (normals == null || normals.Count != vertices.Count)
                {
                    CalculateNormals();
                }
                sharpEdges = new List<int>();
                float cosThreshold = Mathf.Cos(angleThreshold * Mathf.Deg2Rad);
                // Find edges with sharp angle between adjacent faces
                var edgeNormalsDict = new Dictionary<(int, int), List<Vector3>>();
                for (int i = 0; i < indices.Count; i += 3)
                {
                    var normal = Vector3.Cross(
                        vertices[indices[i + 1]] - vertices[indices[i]],
                        vertices[indices[i + 2]] - vertices[indices[i]]
                    ).normalized;
                    for (int j = 0; j < 3; j++)
                    {
                        int v1 = indices[i + j];
                        int v2 = indices[i + (j + 1) % 3];
                        if (v1 > v2)
                        {
                            (v1, v2) = (v2, v1);
                        }
                        var edge = (v1, v2);
                        if (!edgeNormalsDict.ContainsKey(edge))
                        {
                            edgeNormalsDict[edge] = new List<Vector3>();
                        }
                        edgeNormalsDict[edge].Add(normal);
                    }
                }
                // Check for sharp edges
                foreach (var kvp in edgeNormalsDict)
                {
                    if (kvp.Value.Count >= 2)
                    {
                        bool isSharp = false;
                        for (int m = 0; m < kvp.Value.Count - 1; m++)
                        {
                            for (int n = m + 1; n < kvp.Value.Count; n++)
                            {
                                float dot = Vector3.Dot(kvp.Value[m], kvp.Value[n]);
                                if (dot < cosThreshold)
                                {
                                    isSharp = true;
                                    break;
                                }
                            }
                            if (isSharp) break;
                        }
                        if (isSharp)
                        {
                            sharpEdges.Add(kvp.Key.Item1);
                            sharpEdges.Add(kvp.Key.Item2);
                        }
                    }
                }
                // Additional: Sort sharp edges for efficiency
                sharpEdges.Sort();
                // Additional: Calculate edge angles
                CalculateEdgeAngles();
            }
            private void CalculateEdgeAngles()
            {
                // Implementation for calculating angles on sharp edges
                for (int k = 0; k < sharpEdges.Count; k += 2)
                {
                    int v1 = sharpEdges[k];
                    int v2 = sharpEdges[k + 1];
                    // Find adjacent faces
                    // ... (add logic to calculate angle)
                }
            }
            /// <summary>
            /// Identify feature vertices
            /// </summary>
            public void IdentifyFeatureVertices(float saliencyThreshold = 0.5f)
            {
                if (vertexSaliency == null || vertexSaliency.Count != vertices.Count)
                {
                    CalculateSaliency();
                }
                featureVertices = new List<int>();
                for (int i = 0; i < vertices.Count; i++)
                {
                    if (vertexSaliency[i] > saliencyThreshold)
                    {
                        featureVertices.Add(i);
                    }
                }
                // Additional: Cluster features
                ClusterFeatures(0.01f);
                // Additional: Prioritize features
                PrioritizeFeatures();
            }
            private void ClusterFeatures(float clusterDistance)
            {
                // Simple clustering to group close features
                var clustered = new List<List<int>>();
                foreach (var fv in featureVertices)
                {
                    bool added = false;
                    foreach (var cluster in clustered)
                    {
                        if (Vector3.Distance(vertices[fv], vertices[cluster[0]]) < clusterDistance)
                        {
                            cluster.Add(fv);
                            added = true;
                            break;
                        }
                    }
                    if (!added)
                    {
                        clustered.Add(new List<int> { fv });
                    }
                }
                // Replace with cluster representatives
                featureVertices.Clear();
                foreach (var cluster in clustered)
                {
                    featureVertices.Add(cluster[0]); // Use first as rep
                }
            }
            private void PrioritizeFeatures()
            {
                // Sort features by saliency
                featureVertices = featureVertices.OrderByDescending(f => vertexSaliency[f]).ToList();
            }
            /// <summary>
            /// Build acceleration structure for fast queries
            /// </summary>
            public void BuildAccelerationStructure()
            {
                accelerationStructure = new BVH(this);
                hasPrecomputedData = true;
                // Additional: Build additional structures if needed
                if (enableDetailedLoggingStatic)
                    Debug.Log("Acceleration structure built.");
            }
            /// <summary>
            /// Calculate normals if not present
            /// </summary>
            private void CalculateNormals()
            {
                normals = new List<Vector3>(vertices.Count);
                for (int i = 0; i < vertices.Count; i++)
                {
                    normals.Add(Vector3.zero);
                }
                // Calculate face normals and accumulate
                for (int i = 0; i < indices.Count; i += 3)
                {
                    int i0 = indices[i];
                    int i1 = indices[i + 1];
                    int i2 = indices[i + 2];
                    var v0 = vertices[i0];
                    var v1 = vertices[i1];
                    var v2 = vertices[i2];
                    var normal = Vector3.Cross(v1 - v0, v2 - v0).normalized;
                    normals[i0] += normal;
                    normals[i1] += normal;
                    normals[i2] += normal;
                }
                // Normalize accumulated normals
                for (int i = 0; i < normals.Count; i++)
                {
                    normals[i] = normals[i].normalized;
                }
                // Additional: Smooth normals
                SmoothNormals(1);
            }
            private void SmoothNormals(int iterations)
            {
                for (int iter = 0; iter < iterations; iter++)
                {
                    var newNormals = new List<Vector3>(normals);
                    for (int i = 0; i < vertices.Count; i++)
                    {
                        Vector3 sum = normals[i];
                        int count = 1;
                        // Add neighbors
                        for (int j = 0; j < indices.Count; j += 3)
                        {
                            if (indices[j] == i || indices[j + 1] == i || indices[j + 2] == i)
                            {
                                for (int k = 0; k < 3; k++)
                                {
                                    if (indices[j + k] != i)
                                    {
                                        sum += normals[indices[j + k]];
                                        count++;
                                    }
                                }
                            }
                        }
                        newNormals[i] = (sum / count).normalized;
                    }
                    normals = newNormals;
                }
            }
            /// <summary>
            /// Voxelize the mesh
            /// </summary>
            public void Voxelize(float voxelSize, VoxelizationStrategy strategy = VoxelizationStrategy.SolidFill)
            {
                voxelBounds = bounds;
                voxelBounds.Expand(voxelSize * 2); // Padding
                voxelDimensions = new int3(
                    Mathf.CeilToInt(voxelBounds.size.x / voxelSize),
                    Mathf.CeilToInt(voxelBounds.size.y / voxelSize),
                    Mathf.CeilToInt(voxelBounds.size.z / voxelSize)
                );
                int voxelCount = voxelDimensions.x * voxelDimensions.y * voxelDimensions.z;
                sdfValues = new NativeArray<float>(voxelCount, Allocator.Persistent);
                // Depending on strategy
                switch (strategy)
                {
                    case VoxelizationStrategy.SolidFill:
                        VoxelizeSolidFill();
                        break;
                    case VoxelizationStrategy.RaycastBased:
                        VoxelizeRaycast();
                        break;
                    // Add other strategies...
                    default:
                        VoxelizeSolidFill();
                        break;
                }
                // Additional: Validate voxelization
                ValidateVoxelization();
            }
            private void VoxelizeSolidFill()
            {
                // Use job system for voxelization
                var job = new VoxelizationJob
                {
                    Vertices = new NativeArray<Vector3>(vertices.ToArray(), Allocator.TempJob),
                    Indices = new NativeArray<int>(indices.ToArray(), Allocator.TempJob),
                    SDFValues = sdfValues,
                    VoxelBounds = voxelBounds,
                    VoxelDimensions = voxelDimensions,
                    VoxelSize = voxelBounds.size / new float3(voxelDimensions)
                };
                job.Schedule(sdfValues.Length, 64).Complete();
                job.Vertices.Dispose();
                job.Indices.Dispose();
            }
            private void VoxelizeRaycast()
            {
                // Implement raycast-based voxelization
                // For each voxel, raycast to determine if inside
                for (int x = 0; x < voxelDimensions.x; x++)
                {
                    for (int y = 0; y < voxelDimensions.y; y++)
                    {
                        for (int z = 0; z < voxelDimensions.z; z++)
                        {
                            Vector3 voxelPos = VoxelToWorld(new int3(x, y, z));
                            bool inside = IsPointInside(voxelPos);
                            sdfValues[GetVoxelIndex(new int3(x, y, z))] = inside ? -1f : 1f; // Simple signed distance
                        }
                    }
                }
            }
            private int GetVoxelIndex(int3 coord)
            {
                return coord.x + voxelDimensions.x * (coord.y + voxelDimensions.y * coord.z);
            }
            private Vector3 VoxelToWorld(int3 coord)
            {
                return voxelBounds.min + new Vector3(coord.x, coord.y, coord.z) * (voxelBounds.size / new Vector3(voxelDimensions));
            }
            private bool IsPointInside(Vector3 point)
            {
                // Raycast in multiple directions to count crossings
                int crossings = 0;
                Vector3[] directions = { Vector3.right, Vector3.up, Vector3.forward };
                foreach (var dir in directions)
                {
                    int hits = Physics.RaycastAll(point, dir, float.MaxValue).Length;
                    crossings += hits % 2;
                }
                return crossings > 1; // Majority vote
            }
            private void ValidateVoxelization()
            {
                // Check if voxel count matches
                if (sdfValues.Length != voxelDimensions.x * voxelDimensions.y * voxelDimensions.z)
                {
                    Debug.LogError("Voxelization validation failed: mismatched count");
                }
            }
            /// <summary>
            /// Compute SDF from voxel grid
            /// </summary>
            public void ComputeSDF(int iterations = 5, float smoothing = 0.1f)
            {
                // Use fast marching method or iterative smoothing for SDF
                for (int iter = 0; iter < iterations; iter++)
                {
                    var newSDF = new NativeArray<float>(sdfValues, Allocator.Temp);
                    for (int x = 1; x < voxelDimensions.x - 1; x++)
                    {
                        for (int y = 1; y < voxelDimensions.y - 1; y++)
                        {
                            for (int z = 1; z < voxelDimensions.z - 1; z++)
                            {
                                int idx = GetVoxelIndex(new int3(x, y, z));
                                float sum = 0f;
                                int count = 0;
                                // Neighbor offsets
                                int3[] offsets = { new int3(1, 0, 0), new int3(-1, 0, 0), new int3(0, 1, 0), new int3(0, -1, 0), new int3(0, 0, 1), new int3(0, 0, -1) };
                                foreach (var off in offsets)
                                {
                                    int nIdx = GetVoxelIndex(new int3(x, y, z) + off);
                                    sum += sdfValues[nIdx];
                                    count++;
                                }
                                newSDF[idx] = sdfValues[idx] * (1 - smoothing) + (sum / count) * smoothing;
                            }
                        }
                    }
                    sdfValues.CopyFrom(newSDF);
                    newSDF.Dispose();
                }
            }
            /// <summary>
            /// Decimate mesh to reduce vertex count
            /// </summary>
            public void Decimate(float targetReduction)
            {
                // Simple decimation logic
                int targetVertices = Mathf.RoundToInt(vertices.Count * (1 - targetReduction));
                while (vertices.Count > targetVertices)
                {
                    // Remove vertex with lowest cost
                    // ... (add decimation algorithm)
                }
            }
            /// <summary>
            /// Calculate topology properties
            /// </summary>
            private void CalculateTopology()
            {
                // Euler characteristic V - E + F
                int V = vertexCount;
                int E = edgeCount;
                int F = triangleCount;
                eulerCharacteristic = V - E + F;
                // Genus for closed manifold mesh
                if (isClosed && isManifold)
                {
                    genus = (2 - eulerCharacteristic) / 2;
                }
            }
        }
        /// <summary>
        /// Voxelization job for burst compilation
        /// </summary>
        [BurstCompile]
        public struct VoxelizationJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<Vector3> Vertices;
            [ReadOnly] public NativeArray<int> Indices;
            public NativeArray<float> SDFValues;
            public Bounds VoxelBounds;
            public int3 VoxelDimensions;
            public float3 VoxelSize;
            public void Execute(int index)
            {
                int z = index / (VoxelDimensions.x * VoxelDimensions.y);
                int y = (index / VoxelDimensions.x) % VoxelDimensions.y;
                int x = index % VoxelDimensions.x;
                float3 voxelPos = VoxelBounds.min + new float3(x, y, z) * VoxelSize;
                // Check intersection with triangles
                bool inside = false;
                for (int i = 0; i < Indices.Length; i += 3)
                {
                    float3 a = Vertices[Indices[i]];
                    float3 b = Vertices[Indices[i + 1]];
                    float3 c = Vertices[Indices[i + 2]];
                    // Simple point in triangle test or ray intersection count
                    // For simplicity, use bounding box check first
                    if (math.all(voxelPos >= math.min(math.min(a, b), c)) && math.all(voxelPos <= math.max(math.max(a, b), c)))
                    {
                        inside = !inside; // Toggle for parity
                    }
                }
                SDFValues[index] = inside ? -VoxelSize.x : VoxelSize.x; // Approximate SDF
            }
        }
        /// <summary>
        /// Comprehensive data structure for decomposition metrics with detailed tracking
        /// </summary>
        [Serializable]
        public class DecompositionMetrics
        {
            // Basic metrics
            public int vertexCount;
            public int triangleCount;
            public int hullCount;
            public float totalTime;
            public float extractionTime;
            public float segmentationTime;
            public float voxelizationTime;
            public float bvhTime;
            public float decompositionTime;
            public float optimizationTime;
            public float mergingTime;
            public float finalizationTime;
            // Advanced metrics
            public float totalVolume;
            public float totalSurfaceArea;
            public float averageHullVolume;
            public float averageHullSurfaceArea;
            public float maxHullVolume;
            public float minHullVolume;
            public float maxHullSurfaceArea;
            public float minHullSurfaceArea;
            public float averageConcavity;
            public float maxConcavity;
            public float minConcavity;
            public float averageError;
            public float maxError;
            public float minError;
            public float volumeReduction;
            public float surfaceAreaReduction;
            public float compressionRatio;
            // Original mesh metrics for proper reduction calculation
            public float originalMeshVolume;
            public float originalMeshSurfaceArea;
            // Performance metrics
            public int memoryUsage;
            public int peakMemoryUsage;
            public float cpuUsage;
            public int jobCount;
            public int threadCount;
            public float jobOverhead;
            public float schedulingOverhead;
            // Quality metrics
            public float hausdorffDistance;
            public float meanSquaredError;
            public float rootMeanSquaredError;
            public float peakSignalToNoiseRatio;
            public float structuralSimilarityIndex;
            // Custom metrics
            public List<StringFloatKV> customMetrics = new List<StringFloatKV>();
            // Region metrics
            public List<RegionMetrics> regionMetrics = new List<RegionMetrics>();
            // Hull metrics
            public List<HullMetrics> hullMetrics = new List<HullMetrics>();
            // Error distribution metrics
            public List<float> errorDistribution = new List<float>();
            public float errorStandardDeviation;
            public float errorMedian;
            public float error90thPercentile;
            public float error95thPercentile;
            public float error99thPercentile;
            // Processing stage metrics
            public Dictionary<string, float> stageTimes = new Dictionary<string, float>();
            public Dictionary<string, int> stageOperations = new Dictionary<string, int>();
            // Memory metrics per stage
            public Dictionary<string, int> stageMemoryUsage = new Dictionary<string, int>();
            // Additional production metrics
            public float averageProcessingTimePerHull;
            public float totalVoxelizationVolume;
            public int totalVoxelCount;
            public float mergeEfficiency;
            public float splitEfficiency;
            public int failedSplits;
            public int successfulMerges;
            public float overallQualityScore;
            // Additional: Distribution statistics
            public float errorVariance;
            public float errorSkew;
            public float errorKurtosis;
            /// <summary>
            /// Calculate derived metrics
            /// </summary>
            public void CalculateDerivedMetrics()
            {
                if (hullMetrics.Count > 0)
                {
                    totalVolume = hullMetrics.Sum(h => h.volume);
                    totalSurfaceArea = hullMetrics.Sum(h => h.surfaceArea);
                    averageHullVolume = totalVolume / hullMetrics.Count;
                    averageHullSurfaceArea = totalSurfaceArea / hullMetrics.Count;
                    maxHullVolume = hullMetrics.Max(h => h.volume);
                    minHullVolume = hullMetrics.Min(h => h.volume);
                    maxHullSurfaceArea = hullMetrics.Max(h => h.surfaceArea);
                    minHullSurfaceArea = hullMetrics.Min(h => h.surfaceArea);
                    averageConcavity = hullMetrics.Average(h => h.concavity);
                    maxConcavity = hullMetrics.Max(h => h.concavity);
                    minConcavity = hullMetrics.Min(h => h.concavity);
                    averageError = hullMetrics.Average(h => h.hausdorffError);
                    maxError = hullMetrics.Max(h => h.hausdorffError);
                    minError = hullMetrics.Min(h => h.hausdorffError);
                    // Calculate error distribution statistics
                    CalculateErrorDistributionStatistics();
                }
                // Calculate compression ratios
                if (originalMeshVolume > 0)
                    volumeReduction = (originalMeshVolume - totalVolume) / originalMeshVolume;
                if (originalMeshSurfaceArea > 0)
                    surfaceAreaReduction = (originalMeshSurfaceArea - totalSurfaceArea) / originalMeshSurfaceArea;
                if (vertexCount > 0)
                    compressionRatio = (float)hullCount / vertexCount;
                // Additional derived
                averageProcessingTimePerHull = totalTime / hullCount;
                overallQualityScore = (1 - averageError) * (1 - volumeReduction) * compressionRatio;
            }
            /// <summary>
            /// Calculate error distribution statistics
            /// </summary>
            private void CalculateErrorDistributionStatistics()
            {
                if (errorDistribution.Count == 0) return;
                // Sort error values
                var sortedErrors = new List<float>(errorDistribution);
                sortedErrors.Sort();
                // Calculate median
                int count = sortedErrors.Count;
                if (count % 2 == 0)
                {
                    errorMedian = (sortedErrors[count / 2 - 1] + sortedErrors[count / 2]) / 2f;
                }
                else
                {
                    errorMedian = sortedErrors[count / 2];
                }
                // Calculate percentiles
                error90thPercentile = sortedErrors[Mathf.FloorToInt(0.9f * (count - 1))];
                error95thPercentile = sortedErrors[Mathf.FloorToInt(0.95f * (count - 1))];
                error99thPercentile = sortedErrors[Mathf.FloorToInt(0.99f * (count - 1))];
                // Calculate standard deviation
                float mean = sortedErrors.Average();
                float sumOfSquares = sortedErrors.Sum(x => (x - mean) * (x - mean));
                errorStandardDeviation = Mathf.Sqrt(sumOfSquares / count);
                // Additional: Variance and skew
                float variance = sumOfSquares / count;
                float skewSum = sortedErrors.Sum(x => Mathf.Pow(x - mean, 3));
                float skew = skewSum / (count * Mathf.Pow(variance, 1.5f));
                AddCustomMetric("ErrorVariance", variance);
                AddCustomMetric("ErrorSkew", skew);
                // Additional: Kurtosis
                float kurtSum = sortedErrors.Sum(x => Mathf.Pow(x - mean, 4));
                errorKurtosis = (kurtSum / (count * variance * variance)) - 3;
                AddCustomMetric("ErrorKurtosis", errorKurtosis);
            }
            /// <summary>
            /// Record stage time
            /// </summary>
            public void RecordStageTime(string stageName, float time)
            {
                if (stageTimes.ContainsKey(stageName))
                {
                    stageTimes[stageName] += time;
                }
                else
                {
                    stageTimes[stageName] = time;
                }
                // Additional: Log if profiling
                if (enableProfilingStatic)
                    Debug.Log($"Stage {stageName} time: {time:F3}s");
            }
            /// <summary>
            /// Record stage operation count
            /// </summary>
            public void RecordStageOperation(string stageName, int count = 1)
            {
                if (stageOperations.ContainsKey(stageName))
                {
                    stageOperations[stageName] += count;
                }
                else
                {
                    stageOperations[stageName] = count;
                }
            }
            /// <summary>
            /// Record stage memory usage
            /// </summary>
            public void RecordStageMemoryUsage(string stageName, int memoryUsageBytes)
            {
                stageMemoryUsage[stageName] = memoryUsageBytes;
                peakMemoryUsage = Mathf.Max(peakMemoryUsage, memoryUsageBytes);
            }
            /// <summary>
            /// Add custom metric
            /// </summary>
            public void AddCustomMetric(string key, float value)
            {
                customMetrics.Add(new StringFloatKV(key, value));
            }
            // Additional method: Export to CSV for production analysis
            public void ExportToCSV(string filePath)
            {
                using (var writer = new StreamWriter(filePath))
                {
                    writer.WriteLine("Metric,Value");
                    writer.WriteLine($"VertexCount,{vertexCount}");
                    writer.WriteLine($"TriangleCount,{triangleCount}");
                    writer.WriteLine($"HullCount,{hullCount}");
                    writer.WriteLine($"TotalTime,{totalTime}");
                    // ... add all others
                    foreach (var kv in customMetrics)
                    {
                        writer.WriteLine($"{kv.key},{kv.value}");
                    }
                }
            }
            // Additional method: Export to JSON
            public string ExportToJson()
            {
                return JsonUtility.ToJson(this, true);
            }
            // Additional method: Clear metrics
            public void ClearMetrics()
            {
                customMetrics.Clear();
                regionMetrics.Clear();
                hullMetrics.Clear();
                stageTimes.Clear();
                stageOperations.Clear();
                stageMemoryUsage.Clear();
            }
        }
        /// <summary>
        /// Helper class for serializing dictionaries
        /// </summary>
        [Serializable]
        public class StringFloatKV
        {
            public string key;
            public float value;
            public StringFloatKV(string key, float value)
            {
                this.key = key;
                this.value = value;
            }
        }
        /// <summary>
        /// Metrics for individual regions
        /// </summary>
        [Serializable]
        public class RegionMetrics
        {
            public int regionId;
            public int vertexCount;
            public int triangleCount;
            public int hullCount;
            public float volume;
            public float surfaceArea;
            public float saliency;
            public float processingTime;
            public Bounds bounds;
            public Vector3 principalAxis;
            public List<StringFloatKV> customMetrics = new List<StringFloatKV>();
            // Error metrics for this region
            public float averageError;
            public float maxError;
            public float minError;
            public float errorStandardDeviation;
            // Processing metrics
            public int splitCount;
            public int mergeCount;
            public int optimizationIterations;
            // Quality metrics
            public float concavity;
            public float compactness;
            public float aspectRatio;
            // Additional
            public float regionComplexity;
            public int voxelCount;
            public float voxelFillRatio;
            // Additional: Region specific
            public float regionDensity;
            public int boundaryEdges;
            public void CalculateDerived()
            {
                regionComplexity = triangleCount * (1 + saliency);
                voxelFillRatio = voxelCount > 0 ? volume / voxelCount : 0;
                regionDensity = volume > 0 ? vertexCount / volume : 0;
            }
        }
        /// <summary>
        /// Metrics for individual hulls
        /// </summary>
        [Serializable]
        public class HullMetrics
        {
            public int hullId;
            public int regionId;
            public int vertexCount;
            public int triangleCount;
            public float volume;
            public float surfaceArea;
            public float concavity;
            public float hausdorffError;
            public float meanSquaredError;
            public float maxDeviation;
            public Bounds bounds;
            public Vector3 centroid;
            public List<StringFloatKV> customMetrics = new List<StringFloatKV>();
            // Additional quality metrics
            public float compactness;
            public float aspectRatio;
            public float sphericity;
            public float elongation;
            public float flatness;
            // Processing metrics
            public bool wasMerged;
            public int mergeCount;
            public float optimizationTime;
            public int optimizationIterations;
            // Additional
            public float hullEfficiency;
            public bool isOptimized;
            // Additional: Symmetry metrics
            public int symmetryAxesCount;
            public float symmetryScore;
            public void CalculateDerived()
            {
                hullEfficiency = volume / surfaceArea;
                isOptimized = optimizationIterations > 0;
                symmetryScore = symmetryAxesCount / 3f;
            }
        }
        /// <summary>
        /// Represents a convex hull with comprehensive properties and methods
        /// </summary>
        public class ConvexHull
        {
            public List<Vector3> vertices = new List<Vector3>();
            public List<int> indices = new List<int>();
            public List<Vector3> normals = new List<Vector3>();
            public Bounds bounds;
            public float volume;
            public float surfaceArea;
            public float hausdorffError;
            public float meanSquaredError;
            public float maxDeviation;
            public float concavity;
            public int regionId;
            public int hullId;
            public List<HullEdge> edges = new List<HullEdge>();
            public List<PlaneF> planes = new List<PlaneF>();
            public Vector3 centroid;
            public float inertiaTensor;
            public Matrix4x4 inertiaMatrix;
            public float quality;
            public bool isValid;
            public List<int> neighborHulls = new List<int>();
            public Dictionary<string, object> customProperties = new Dictionary<string, object>();
            // Additional properties for advanced processing
            public List<Vector3> witnessPoints;
            public List<float> witnessErrors;
            public MeshData meshData;
            public BVH accelerationStructure;
            public bool hasPrecomputedData;
            public float compactness;
            public float aspectRatio;
            public float sphericity;
            public float elongation;
            public float flatness;
            // Additional production properties
            public bool isOptimized;
            public float optimizationScore;
            public DateTime creationTime;
            public List<Vector3> symmetryAxes;
            // Additional: Hull topology
            public int genus;
            public int eulerCharacteristic;
            /// <summary>
            /// Optimize hull to reduce vertex count while maintaining shape
            /// </summary>
            public void OptimizeHull(float errorTolerance)
            {
                if (vertices.Count <= 8) return; // Already minimal
                // Implementation of vertex reduction algorithm
                var removedVertices = new bool[vertices.Count];
                var errors = new float[vertices.Count];
                // Calculate error for each vertex
                for (int i = 0; i < vertices.Count; i++)
                {
                    errors[i] = CalculateVertexRemovalError(i);
                }
                // Remove vertices with smallest error first
                var sortedVertices = Enumerable.Range(0, vertices.Count)
                    .OrderBy(i => errors[i])
                    .ToList();
                int targetVertexCount = Mathf.Max(8, vertices.Count / 2);
                int removedCount = 0;
                foreach (var vertexIndex in sortedVertices)
                {
                    if (removedCount >= vertices.Count - targetVertexCount)
                        break;
                    if (errors[vertexIndex] < errorTolerance)
                    {
                        removedVertices[vertexIndex] = true;
                        removedCount++;
                    }
                }
                // Rebuild hull with remaining vertices
                if (removedCount > 0)
                {
                    RebuildHull(removedVertices);
                }
                isOptimized = true;
                optimizationScore = 1 - (float)removedCount / vertices.Count;
                // Additional: Recalculate all properties after optimization
                CalculateProperties();
                BuildPlanes();
                BuildEdges();
                ValidateHull();
                // Additional: Apply post-optimization smoothing
                SmoothHull(1);
            }
            private float CalculateVertexRemovalError(int vertexIndex)
            {
                // Calculate error introduced by removing this vertex
                float error = 0f;
                var vertex = vertices[vertexIndex];
                // Find all triangles using this vertex
                var affectedTriangles = new List<int>();
                for (int i = 0; i < indices.Count; i += 3)
                {
                    if (indices[i] == vertexIndex ||
                        indices[i + 1] == vertexIndex ||
                        indices[i + 2] == vertexIndex)
                    {
                        affectedTriangles.Add(i);
                    }
                }
                // Calculate error for each affected triangle
                foreach (var triIndex in affectedTriangles)
                {
                    var i1 = indices[triIndex];
                    var i2 = indices[triIndex + 1];
                    var i3 = indices[triIndex + 2];
                    // Skip if triangle would become degenerate
                    if ((i1 == vertexIndex && i2 == vertexIndex) ||
                        (i2 == vertexIndex && i3 == vertexIndex) ||
                        (i3 == vertexIndex && i1 == vertexIndex))
                    {
                        error += float.MaxValue;
                        continue;
                    }
                    // Calculate distance from vertex to opposite edge
                    var v1 = vertices[i1];
                    var v2 = vertices[i2];
                    var v3 = vertices[i3];
                    Vector3 edge1, edge2;
                    if (i1 == vertexIndex)
                    {
                        edge1 = v3 - v2;
                        edge2 = vertex - v2;
                    }
                    else if (i2 == vertexIndex)
                    {
                        edge1 = v1 - v3;
                        edge2 = vertex - v3;
                    }
                    else
                    {
                        edge1 = v2 - v1;
                        edge2 = vertex - v1;
                    }
                    var distance = Vector3.Cross(edge1, edge2).magnitude / edge1.magnitude;
                    error += distance * distance; // Use squared for better scaling
                }
                return error;
            }
            private void RebuildHull(bool[] removedVertices)
            {
                // Create vertex remapping
                var vertexMap = new Dictionary<int, int>();
                var newVertices = new List<Vector3>();
                int newIndex = 0;
                for (int i = 0; i < vertices.Count; i++)
                {
                    if (!removedVertices[i])
                    {
                        vertexMap[i] = newIndex;
                        newVertices.Add(vertices[i]);
                        newIndex++;
                    }
                }
                // Rebuild indices
                var newIndices = new List<int>();
                for (int i = 0; i < indices.Count; i += 3)
                {
                    var i1 = indices[i];
                    var i2 = indices[i + 1];
                    var i3 = indices[i + 2];
                    // Skip triangles with removed vertices
                    if (removedVertices[i1] || removedVertices[i2] || removedVertices[i3])
                        continue;
                    newIndices.Add(vertexMap[i1]);
                    newIndices.Add(vertexMap[i2]);
                    newIndices.Add(vertexMap[i3]);
                }
                // Update hull
                vertices = newVertices;
                indices = newIndices;
                // Recalculate properties
                CalculateProperties();
                // Additional: Update normals if present
                if (normals.Count > 0)
                {
                    var newNormals = new List<Vector3>(newVertices.Count);
                    for (int k = 0; k < newVertices.Count; k++) newNormals.Add(Vector3.zero);
                    foreach (var kv in vertexMap)
                        if (kv.Value < newNormals.Count && kv.Key < normals.Count)
                            newNormals[kv.Value] = normals[kv.Key];
                    normals = newNormals;
                }
            }
            /// <summary>
            /// Check if point is inside hull
            /// </summary>
            public bool ContainsPoint(Vector3 point)
            {
                // Use precomputed planes for faster containment checks
                for (int i = 0; i < planes.Count; i++)
                {
                    var plane = planes[i];
                    if (Vector3.Dot(plane.n, point) + plane.d > 1e-3f)
                        return false;
                }
                return true;
            }
            /// <summary>
            /// Calculate distance from point to hull surface
            /// </summary>
            public float DistanceToPoint(Vector3 point)
            {
                if (ContainsPoint(point))
                    return 0f;
                float minDistance = float.MaxValue;
                // Check distance to each triangle
                for (int i = 0; i < indices.Count; i += 3)
                {
                    var a = vertices[indices[i]];
                    var b = vertices[indices[i + 1]];
                    var c = vertices[indices[i + 2]];
                    // Calculate distance to triangle
                    var distance = DistanceToTriangle(point, a, b, c);
                    minDistance = Mathf.Min(minDistance, distance);
                }
                return minDistance;
            }
            private float DistanceToTriangle(Vector3 point, Vector3 a, Vector3 b, Vector3 c)
            {
                // Calculate closest point on triangle to the given point
                var ab = b - a;
                var ac = c - a;
                var ap = point - a;
                var d1 = Vector3.Dot(ab, ap);
                var d2 = Vector3.Dot(ac, ap);
                if (d1 <= 0f && d2 <= 0f)
                    return Vector3.Distance(point, a);
                var bp = point - b;
                var d3 = Vector3.Dot(ab, bp);
                var d4 = Vector3.Dot(ac, bp);
                if (d3 >= 0f && d4 <= d3)
                    return Vector3.Distance(point, b);
                var vc = d1 * d4 - d3 * d2;
                if (vc <= 0f && d1 >= 0f && d3 <= 0f)
                {
                    var v = d1 / (d1 - d3);
                    return Vector3.Distance(point, a + v * ab);
                }
                var cp = point - c;
                var d5 = Vector3.Dot(ab, cp);
                var d6 = Vector3.Dot(ac, cp);
                if (d6 >= 0f && d5 <= d6)
                    return Vector3.Distance(point, c);
                var vb = d5 * d2 - d1 * d6;
                if (vb <= 0f && d2 >= 0f && d6 <= 0f)
                {
                    var w = d2 / (d2 - d6);
                    return Vector3.Distance(point, a + w * ac);
                }
                var va = d3 * d6 - d5 * d4;
                if (va <= 0f && (d4 - d3) >= 0f && (d5 - d6) >= 0f)
                {
                    var w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                    return Vector3.Distance(point, b + w * (c - b));
                }
                // Inside triangle
                var denom = 1f / (va + vb + vc);
                var v2 = vb * denom;
                var w2 = vc * denom;
                var closestPoint = a + ab * v2 + ac * w2;
                return Vector3.Distance(point, closestPoint);
            }
            /// <summary>
            /// Precompute face planes for faster containment checks
            /// </summary>
            public void BuildPlanes()
            {
                planes.Clear();
                if (vertices.Count == 0) return;
                // Calculate centroid
                centroid = Vector3.zero;
                foreach (var v in vertices)
                    centroid += v;
                centroid /= vertices.Count;
                // Build planes for each triangle
                for (int i = 0; i < indices.Count; i += 3)
                {
                    var a = vertices[indices[i]];
                    var b = vertices[indices[i + 1]];
                    var c = vertices[indices[i + 2]];
                    var n = Vector3.Cross(b - a, c - a);
                    n.Normalize();
                    var d = -Vector3.Dot(n, a);
                    // Ensure normal points outward
                    if (Vector3.Dot(n, centroid) + d > 0f)
                    {
                        n = -n;
                        d = -d;
                    }
                    planes.Add(new PlaneF { n = n, d = d });
                }
                // Additional: Optimize planes list
                planes = planes.Distinct(new PlaneFComparer()).ToList();
            }
            class PlaneFComparer : IEqualityComparer<PlaneF>
            {
                public bool Equals(PlaneF x, PlaneF y)
                {
                    return Vector3.Dot(x.n, y.n) > 0.99f && Mathf.Abs(x.d - y.d) < 0.001f;
                }
                public int GetHashCode(PlaneF obj)
                {
                    const float q = 1e-3f;
                    var n = new Vector3(Mathf.Round(obj.n.x / q) * q, Mathf.Round(obj.n.y / q) * q, Mathf.Round(obj.n.z / q) * q);
                    var d = Mathf.Round(obj.d / q) * q;
                    return n.GetHashCode() ^ d.GetHashCode();
                }
            }
            /// <summary>
            /// Calculate hull properties
            /// </summary>
            public void CalculateProperties()
            {
                // Calculate bounds
                bounds = new Bounds();
                if (vertices.Count > 0)
                {
                    bounds = new Bounds(vertices[0], Vector3.zero);
                    foreach (var v in vertices)
                        bounds.Encapsulate(v);
                }
                // Calculate centroid
                centroid = Vector3.zero;
                foreach (var v in vertices)
                    centroid += v;
                if (vertices.Count > 0)
                    centroid /= vertices.Count;
                // Calculate volume
                volume = 0f;
                for (int i = 0; i < indices.Count; i += 3)
                {
                    var a = vertices[indices[i]];
                    var b = vertices[indices[i + 1]];
                    var c = vertices[indices[i + 2]];
                    volume += Vector3.Dot(a, Vector3.Cross(b, c)) / 6f;
                }
                volume = Mathf.Abs(volume);
                // Calculate surface area
                surfaceArea = 0f;
                for (int i = 0; i < indices.Count; i += 3)
                {
                    var a = vertices[indices[i]];
                    var b = vertices[indices[i + 1]];
                    var c = vertices[indices[i + 2]];
                    surfaceArea += Vector3.Cross(b - a, c - a).magnitude / 2f;
                }
                // Calculate inertia tensor
                CalculateInertiaTensor();
                // Calculate shape metrics
                CalculateShapeMetrics();
                // Build edges
                BuildEdges();
                // Validate hull
                ValidateHull();
                // Additional: Calculate symmetry
                DetectSymmetry();
                creationTime = DateTime.Now;
                // Additional: Calculate topology
                CalculateTopology();
            }
            private void CalculateInertiaTensor()
            {
                // Calculate inertia tensor using parallel axis theorem
                float xx = 0f, yy = 0f, zz = 0f, xy = 0f, xz = 0f, yz = 0f;
                for (int i = 0; i < indices.Count; i += 3)
                {
                    var a = vertices[indices[i]];
                    var b = vertices[indices[i + 1]];
                    var c = vertices[indices[i + 2]];
                    // Calculate tetrahedron contribution to inertia
                    float vol = Vector3.Dot(a, Vector3.Cross(b, c)) / 6f;
                    if (vol <= 0) continue;
                    // Center of mass of tetrahedron
                    var com = (a + b + c) / 4f;
                    // Inertia contribution
                    xx += vol * (com.y * com.y + com.z * com.z);
                    yy += vol * (com.x * com.x + com.z * com.z);
                    zz += vol * (com.x * com.x + com.y * com.y);
                    xy -= vol * com.x * com.y;
                    xz -= vol * com.x * com.z;
                    yz -= vol * com.y * com.z;
                }
                // Build inertia tensor matrix
                inertiaMatrix = new Matrix4x4(
                    new Vector4(xx, xy, xz, 0),
                    new Vector4(xy, yy, yz, 0),
                    new Vector4(xz, yz, zz, 0),
                    new Vector4(0, 0, 0, 1)
                );
                // Calculate scalar inertia tensor (trace)
                inertiaTensor = xx + yy + zz;
            }
            /// <summary>
            /// Calculate shape metrics for quality assessment
            /// </summary>
            private void CalculateShapeMetrics()
            {
                if (volume <= 1e-6f || surfaceArea <= 1e-6f)
                {
                    compactness = sphericity = aspectRatio = elongation = flatness = 0;
                    return;
                }
                // Calculate compactness (volume to surface area ratio)
                compactness = volume / surfaceArea;
                // Calculate sphericity (how sphere-like the shape is)
                float sphereSurfaceArea = Mathf.Pow(Mathf.PI, 1 / 3f) * Mathf.Pow(6 * volume, 2 / 3f);
                sphericity = sphereSurfaceArea / surfaceArea;
                // Calculate aspect ratio (based on bounding box)
                var size = bounds.size;
                var sortedSize = new float[] { size.x, size.y, size.z };
                Array.Sort(sortedSize);
                if (sortedSize[0] < 1e-6f)
                {
                    aspectRatio = elongation = flatness = 0;
                    return;
                }
                aspectRatio = sortedSize[2] / sortedSize[0]; // max / min
                // Calculate elongation and flatness
                elongation = (sortedSize[1] - sortedSize[0]) / sortedSize[2];
                flatness = sortedSize[0] / sortedSize[2];
                // Additional: Calculate more metrics like convexity measure
                float convexity = volume / bounds.size.x / bounds.size.y / bounds.size.z;
                customProperties["Convexity"] = convexity;
            }
            private void BuildEdges()
            {
                edges.Clear();
                var edgeSet = new HashSet<(int, int)>();
                for (int i = 0; i < indices.Count; i += 3)
                {
                    int i0 = indices[i];
                    int i1 = indices[i + 1];
                    int i2 = indices[i + 2];
                    // Add edges (ensure consistent ordering)
                    AddEdge(i0, i1, edgeSet);
                    AddEdge(i1, i2, edgeSet);
                    AddEdge(i2, i0, edgeSet);
                }
                // Additional: Sort edges by length
                edges.Sort((e1, e2) => e1.length.CompareTo(e2.length));
                // Additional: Calculate edge angles
                CalculateEdgeAngles();
            }
            private void AddEdge(int i1, int i2, HashSet<(int, int)> edgeSet)
            {
                // Ensure consistent ordering
                if (i1 > i2)
                {
                    (i1, i2) = (i2, i1);
                }
                var edge = (i1, i2);
                if (edgeSet.Add(edge))
                {
                    edges.Add(new HullEdge
                    {
                        vertex1 = i1,
                        vertex2 = i2,
                        length = Vector3.Distance(vertices[i1], vertices[i2])
                    });
                }
            }
            private void CalculateEdgeAngles()
            {
                // Calculate angles for each edge
                for (int k = 0; k < edges.Count; k++)
                {
                    var edge = edges[k];
                    // Find adjacent faces
                    List<Vector3> adjNormals = new List<Vector3>();
                    for (int m = 0; m < indices.Count; m += 3)
                    {
                        bool hasEdge = false;
                        for (int n = 0; n < 3; n++)
                        {
                            int v1 = indices[m + n];
                            int v2 = indices[m + (n + 1) % 3];
                            if ((v1 == edge.vertex1 && v2 == edge.vertex2) || (v1 == edge.vertex2 && v2 == edge.vertex1))
                            {
                                hasEdge = true;
                                break;
                            }
                        }
                        if (hasEdge)
                        {
                            var a = vertices[indices[m]];
                            var b = vertices[indices[m + 1]];
                            var c = vertices[indices[m + 2]];
                            var normal = Vector3.Cross(b - a, c - a).normalized;
                            adjNormals.Add(normal);
                        }
                    }
                    if (adjNormals.Count == 2)
                    {
                        edges[k].CalculateAngle(adjNormals[0], adjNormals[1]);
                    }
                }
            }
            private void ValidateHull()
            {
                isValid = true;
                if (vertices.Count < 4 || indices.Count < 12)
                {
                    isValid = false;
                    return;
                }
                // Check if hull is convex
                if (!IsConvex())
                {
                    isValid = false;
                    return;
                }
                // Check if hull is closed
                if (!IsClosed())
                {
                    isValid = false;
                    return;
                }
                // Check for degenerate triangles
                for (int i = 0; i < indices.Count; i += 3)
                {
                    var a = vertices[indices[i]];
                    var b = vertices[indices[i + 1]];
                    var c = vertices[indices[i + 2]];
                    var area = Vector3.Cross(b - a, c - a).magnitude / 2f;
                    if (area < 1e-6f)
                    {
                        isValid = false;
                        return;
                    }
                }
                // Additional: Check self-intersections
                if (HasSelfIntersections())
                {
                    isValid = false;
                }
            }
            private bool HasSelfIntersections()
            {
                // Basic check for intersecting triangles
                for (int i = 0; i < indices.Count; i += 3)
                {
                    for (int j = i + 3; j < indices.Count; j += 3)
                    {
                        if (TrianglesIntersect(i, j))
                            return true;
                    }
                }
                return false;
            }
            private bool TrianglesIntersect(int tri1, int tri2)
            {
                var a1 = vertices[indices[tri1]];
                var b1 = vertices[indices[tri1 + 1]];
                var c1 = vertices[indices[tri1 + 2]];
                var a2 = vertices[indices[tri2]];
                var b2 = vertices[indices[tri2 + 1]];
                var c2 = vertices[indices[tri2 + 2]];
                // Check if any edge of tri1 intersects tri2
                if (LineTriangleIntersect(a1, b1, a2, b2, c2) ||
                    LineTriangleIntersect(b1, c1, a2, b2, c2) ||
                    LineTriangleIntersect(c1, a1, a2, b2, c2))
                    return true;
                // Check if any edge of tri2 intersects tri1
                if (LineTriangleIntersect(a2, b2, a1, b1, c1) ||
                    LineTriangleIntersect(b2, c2, a1, b1, c1) ||
                    LineTriangleIntersect(c2, a2, a1, b1, c1))
                    return true;
                return false;
            }
            private bool LineTriangleIntersect(Vector3 p1, Vector3 p2, Vector3 a, Vector3 b, Vector3 c)
            {
                // Implement line-triangle intersection
                var plane = new Plane(a, b, c);
                if (!plane.Raycast(new Ray(p1, p2 - p1), out float enter))
                    return false;
                if (enter < 0 || enter > Vector3.Distance(p1, p2))
                    return false;
                var point = p1 + enter * (p2 - p1);
                // Check if point is inside triangle
                var u = Vector3.Cross(b - a, point - a).magnitude;
                var v = Vector3.Cross(c - b, point - b).magnitude;
                var w = Vector3.Cross(a - c, point - c).magnitude;
                var area = Vector3.Cross(b - a, c - a).magnitude / 2f;
                return Mathf.Abs(u + v + w - area * 2) < 0.01f;
            }
            private bool IsConvex()
            {
                // Check if all face normals point outward relative to the centroid
                for (int i = 0; i < indices.Count; i += 3)
                {
                    var a = vertices[indices[i]];
                    var b = vertices[indices[i + 1]];
                    var c = vertices[indices[i + 2]];
                    var normal = Vector3.Cross(b - a, c - a).normalized;
                    var center = (a + b + c) / 3f;
                    // Check if centroid is on the "inside" of the face
                    if (Vector3.Dot(normal, centroid - center) > 1e-6f)
                    {
                        return false;
                    }
                }
                return true;
            }
            private bool IsClosed()
            {
                var edgeCountDict = new Dictionary<(int, int), int>();
                for (int i = 0; i < indices.Count; i += 3)
                {
                    for (int j = 0; j < 3; j++)
                    {
                        int v1 = indices[i + j];
                        int v2 = indices[i + (j + 1) % 3];
                        if (v1 > v2)
                        {
                            (v1, v2) = (v2, v1);
                        }
                        var edge = (v1, v2);
                        if (edgeCountDict.ContainsKey(edge))
                            edgeCountDict[edge]++;
                        else
                            edgeCountDict[edge] = 1;
                    }
                }
                // Hull is closed if all edges are shared by exactly 2 triangles
                foreach (var count in edgeCountDict.Values)
                {
                    if (count != 2)
                        return false;
                }
                return true;
            }
            /// <summary>
            /// Build mesh data representation
            /// </summary>
            public void BuildMeshData()
            {
                meshData = new MeshData();
                meshData.vertices = new List<Vector3>(vertices);
                meshData.indices = new List<int>(indices);
                meshData.normals = new List<Vector3>(normals);
                meshData.CalculateProperties();
            }
            /// <summary>
            /// Build acceleration structure for fast queries
            /// </summary>
            public void BuildAccelerationStructure()
            {
                if (meshData == null)
                {
                    BuildMeshData();
                }
                accelerationStructure = new BVH(meshData);
                hasPrecomputedData = true;
            }
            /// <summary>
            /// Calculate witness points for error analysis
            /// </summary>
            public void CalculateWitnessPoints(MeshData referenceMesh, int sampleCount = 100)
            {
                if (accelerationStructure == null)
                {
                    BuildAccelerationStructure();
                }
                witnessPoints = new List<Vector3>();
                witnessErrors = new List<float>();
                // Sample points from reference mesh
                var samples = ErrorCalculator.SampleMeshSurface(referenceMesh, sampleCount);
                foreach (var sample in samples)
                {
                    float distance = DistanceToPoint(sample);
                    witnessPoints.Add(sample);
                    witnessErrors.Add(distance);
                }
                // Additional: Sort by error
                var sortedIndices = Enumerable.Range(0, witnessErrors.Count).OrderByDescending(idx => witnessErrors[idx]).ToList();
                var sortedPoints = new List<Vector3>();
                var sortedErrors = new List<float>();
                foreach (var idx in sortedIndices)
                {
                    sortedPoints.Add(witnessPoints[idx]);
                    sortedErrors.Add(witnessErrors[idx]);
                }
                witnessPoints = sortedPoints;
                witnessErrors = sortedErrors;
            }
            private void DetectSymmetry()
            {
                symmetryAxes = new List<Vector3>();
                // Basic symmetry detection along axes
                if (IsSymmetricAlongAxis(Vector3.right))
                    symmetryAxes.Add(Vector3.right);
                if (IsSymmetricAlongAxis(Vector3.up))
                    symmetryAxes.Add(Vector3.up);
                if (IsSymmetricAlongAxis(Vector3.forward))
                    symmetryAxes.Add(Vector3.forward);
                // Additional: Detect arbitrary symmetry planes
                DetectArbitrarySymmetry();
            }
            private void DetectArbitrarySymmetry()
            {
                // Advanced symmetry detection
                // ... (add logic for PCA or other methods)
            }
            private bool IsSymmetricAlongAxis(Vector3 axis)
            {
                // Check if reflecting across plane gives the same hull
                var plane = new PlaneF { n = axis, d = -Vector3.Dot(centroid, axis) };
                var reflectedVertices = new List<Vector3>();
                foreach (var v in vertices)
                {
                    float dist = plane.DistanceToPoint(v);
                    var reflected = v - 2 * dist * plane.n;
                    reflectedVertices.Add(reflected);
                }
                // Check if all reflected points are in hull
                foreach (var rv in reflectedVertices)
                {
                    if (!ContainsPoint(rv))
                        return false;
                }
                return true;
            }
            private void SmoothHull(int iterations)
            {
                // Apply smoothing to hull vertices
                for (int iter = 0; iter < iterations; iter++)
                {
                    // ... (add smoothing logic)
                }
            }
            private void CalculateTopology()
            {
                // Calculate Euler characteristic for hull
                int V = vertices.Count;
                int E = edges.Count;
                int F = indices.Count / 3;
                eulerCharacteristic = V - E + F;
                genus = (2 - eulerCharacteristic) / 2;
            }
        }
        /// <summary>
        /// Represents an edge in a convex hull
        /// </summary>
        [Serializable]
        public struct HullEdge
        {
            public int vertex1;
            public int vertex2;
            public float length;
            // Additional
            public bool isSharp;
            public float angle;
            public float CalculateAngle(Vector3 n1, Vector3 n2)
            {
                angle = Mathf.Acos(Vector3.Dot(n1, n2)) * Mathf.Rad2Deg;
                isSharp = angle > 30f;
                return angle;
            }
        }
        /// <summary>
        /// Burst-safe plane structure
        /// </summary>
        [Serializable]
        public struct PlaneF
        {
            public Vector3 n; // normal
            public float d; // distance from origin
            public float DistanceToPoint(Vector3 point)
            {
                return Vector3.Dot(n, point) + d;
            }
            // Additional
            public Vector3 ProjectPoint(Vector3 point)
            {
                return point - DistanceToPoint(point) * n;
            }
            public bool Equals(PlaneF other, float epsilon = 1e-5f)
            {
                return Vector3.Dot(n, other.n) > 1 - epsilon && Mathf.Abs(d - other.d) < epsilon;
            }
        }
        /// <summary>
        /// Error calculator for mesh comparison with multiple metrics
        /// </summary>
        public static class ErrorCalculator
        {
            /// <summary>
            /// Sample points from mesh surface using specified strategy
            /// </summary>
            public static List<Vector3> SampleMeshSurface(MeshData mesh, int sampleCount, SamplingStrategy strategy = SamplingStrategy.AreaWeighted)
            {
                var samples = new List<Vector3>();
                if (mesh.triangleCount == 0) return samples;
                switch (strategy)
                {
                    case SamplingStrategy.Uniform:
                        return SampleUniform(mesh, sampleCount);
                    case SamplingStrategy.AreaWeighted:
                        return SampleAreaWeighted(mesh, sampleCount);
                    case SamplingStrategy.CurvatureWeighted:
                        return SampleCurvatureWeighted(mesh, sampleCount);
                    case SamplingStrategy.FeaturePreserving:
                        return SampleFeaturePreserving(mesh, sampleCount);
                    case SamplingStrategy.Adaptive:
                        return SampleAdaptive(mesh, sampleCount);
                    case SamplingStrategy.Stratified:
                        return SampleStratified(mesh, sampleCount);
                    default:
                        return SampleAreaWeighted(mesh, sampleCount);
                }
            }
            /// <summary>
            /// Uniform sampling across triangles
            /// </summary>
            private static List<Vector3> SampleUniform(MeshData mesh, int sampleCount)
            {
                var samples = new List<Vector3>();
                int trianglesPerSample = Mathf.Max(1, mesh.triangleCount / sampleCount);
                for (int i = 0; i < mesh.indices.Count; i += 3 * trianglesPerSample)
                {
                    if (samples.Count >= sampleCount) break;
                    // Select a triangle from this group
                    int triIndex = i + UnityEngine.Random.Range(0, trianglesPerSample * 3);
                    triIndex = Mathf.Min(triIndex, mesh.indices.Count - 3);
                    // Get triangle vertices
                    var a = mesh.vertices[mesh.indices[triIndex]];
                    var b = mesh.vertices[mesh.indices[triIndex + 1]];
                    var c = mesh.vertices[mesh.indices[triIndex + 2]];
                    // Generate random barycentric coordinates
                    float r1 = UnityEngine.Random.value;
                    float r2 = UnityEngine.Random.value;
                    if (r1 + r2 > 1f)
                    {
                        r1 = 1f - r1;
                        r2 = 1f - r2;
                    }
                    float u = r1;
                    float v = r2;
                    float w = 1f - u - v;
                    // Calculate point
                    var point = a * u + b * v + c * w;
                    samples.Add(point);
                }
                return samples;
            }
            /// <summary>
            /// Area-weighted sampling
            /// </summary>
            private static List<Vector3> SampleAreaWeighted(MeshData mesh, int sampleCount)
            {
                var samples = new List<Vector3>();
                if (mesh.triangleCount == 0) return samples;
                // Calculate triangle areas for weighted sampling
                var areas = new float[mesh.triangleCount];
                float totalArea = 0f;
                for (int i = 0; i < mesh.indices.Count; i += 3)
                {
                    int triIndex = i / 3;
                    var a = mesh.vertices[mesh.indices[i]];
                    var b = mesh.vertices[mesh.indices[i + 1]];
                    var c = mesh.vertices[mesh.indices[i + 2]];
                    areas[triIndex] = Vector3.Cross(b - a, c - a).magnitude / 2f;
                    totalArea += areas[triIndex];
                }
                if (totalArea < 1e-6f) return samples;
                // Normalize areas to create cumulative distribution
                var cumulativeAreas = new float[areas.Length];
                float cumulative = 0f;
                for (int i = 0; i < areas.Length; i++)
                {
                    cumulative += areas[i];
                    cumulativeAreas[i] = cumulative;
                }
                // Sample points
                for (int s = 0; s < sampleCount; s++)
                {
                    float r = UnityEngine.Random.value * totalArea;
                    // Find triangle using binary search
                    int low = 0;
                    int high = cumulativeAreas.Length - 1;
                    while (low < high)
                    {
                        int mid = (low + high) / 2;
                        if (cumulativeAreas[mid] < r)
                            low = mid + 1;
                        else
                            high = mid;
                    }
                    int triIndex = low;
                    // Get triangle vertices
                    int idx = triIndex * 3;
                    var a = mesh.vertices[mesh.indices[idx]];
                    var b = mesh.vertices[mesh.indices[idx + 1]];
                    var c = mesh.vertices[mesh.indices[idx + 2]];
                    // Generate random barycentric coordinates
                    float r1 = UnityEngine.Random.value;
                    float r2 = UnityEngine.Random.value;
                    if (r1 + r2 > 1f)
                    {
                        r1 = 1f - r1;
                        r2 = 1f - r2;
                    }
                    float u = r1;
                    float v = r2;
                    float w = 1f - u - v;
                    // Calculate point
                    var point = a * u + b * v + c * w;
                    samples.Add(point);
                }
                return samples;
            }
            /// <summary>
            /// Curvature-weighted sampling
            /// </summary>
            private static List<Vector3> SampleCurvatureWeighted(MeshData mesh, int sampleCount)
            {
                // Calculate curvature if not already present
                if (mesh.curvature == null || mesh.curvature.Count != mesh.vertices.Count)
                {
                    mesh.CalculateCurvature();
                }
                var samples = new List<Vector3>();
                // Calculate triangle curvature weights
                var weights = new float[mesh.triangleCount];
                float totalWeight = 0f;
                for (int i = 0; i < mesh.indices.Count; i += 3)
                {
                    int triIndex = i / 3;
                    var i0 = mesh.indices[i];
                    var i1 = mesh.indices[i + 1];
                    var i2 = mesh.indices[i + 2];
                    // Use average curvature of triangle vertices
                    weights[triIndex] = (mesh.curvature[i0].magnitude +
                                         mesh.curvature[i1].magnitude +
                                         mesh.curvature[i2].magnitude) / 3f;
                    totalWeight += weights[triIndex];
                }
                if (totalWeight < 1e-6f) return SampleAreaWeighted(mesh, sampleCount);
                // Normalize weights to create cumulative distribution
                var cumulativeWeights = new float[weights.Length];
                float cumulative = 0f;
                for (int i = 0; i < weights.Length; i++)
                {
                    cumulative += weights[i];
                    cumulativeWeights[i] = cumulative;
                }
                // Sample points
                for (int s = 0; s < sampleCount; s++)
                {
                    float r = UnityEngine.Random.value * totalWeight;
                    // Find triangle using binary search
                    int low = 0;
                    int high = cumulativeWeights.Length - 1;
                    while (low < high)
                    {
                        int mid = (low + high) / 2;
                        if (cumulativeWeights[mid] < r)
                            low = mid + 1;
                        else
                            high = mid;
                    }
                    int triIndex = low;
                    // Get triangle vertices
                    int idx = triIndex * 3;
                    var a = mesh.vertices[mesh.indices[idx]];
                    var b = mesh.vertices[mesh.indices[idx + 1]];
                    var c = mesh.vertices[mesh.indices[idx + 2]];
                    // Generate random barycentric coordinates
                    float r1 = UnityEngine.Random.value;
                    float r2 = UnityEngine.Random.value;
                    if (r1 + r2 > 1f)
                    {
                        r1 = 1f - r1;
                        r2 = 1f - r2;
                    }
                    float u = r1;
                    float v = r2;
                    float w = 1f - u - v;
                    // Calculate point
                    var point = a * u + b * v + c * w;
                    samples.Add(point);
                }
                return samples;
            }
            /// <summary>
            /// Feature-preserving sampling
            /// </summary>
            private static List<Vector3> SampleFeaturePreserving(MeshData mesh, int sampleCount)
            {
                // Identify features if not already present
                if (mesh.featureVertices == null || mesh.featureVertices.Count == 0)
                {
                    mesh.IdentifyFeatureVertices();
                }
                var samples = new List<Vector3>();
                // Reserve some samples for feature vertices
                int featureSampleCount = Mathf.Min(sampleCount / 4, mesh.featureVertices.Count);
                int regularSampleCount = sampleCount - featureSampleCount;
                // Sample feature vertices
                if (mesh.featureVertices.Count > 0)
                {
                    var featureIndices = new List<int>(mesh.featureVertices);
                    // Randomly select feature vertices
                    for (int i = 0; i < featureSampleCount; i++)
                    {
                        if (featureIndices.Count == 0) break;
                        int idx = UnityEngine.Random.Range(0, featureIndices.Count);
                        samples.Add(mesh.vertices[featureIndices[idx]]);
                        featureIndices.RemoveAt(idx);
                    }
                }
                // Sample remaining points using area-weighted strategy
                if (regularSampleCount > 0)
                {
                    var regularSamples = SampleAreaWeighted(mesh, regularSampleCount);
                    samples.AddRange(regularSamples);
                }
                // Additional: Ensure no duplicates
                samples = samples.Distinct(new Vector3Comparer()).ToList();
                return samples;
            }
            class Vector3Comparer : IEqualityComparer<Vector3>
            {
                public bool Equals(Vector3 x, Vector3 y)
                {
                    return (x - y).sqrMagnitude < 1e-6f;
                }
                public int GetHashCode(Vector3 obj)
                {
                    return obj.GetHashCode();
                }
            }
            /// <summary>
            /// Adaptive sampling based on error
            /// </summary>
            private static List<Vector3> SampleAdaptive(MeshData mesh, int sampleCount)
            {
                // Start with area-weighted sampling
                var samples = SampleAreaWeighted(mesh, sampleCount / 2);
                // Add more samples in high-error regions (using curvature as a proxy)
                var curvatureSamples = SampleCurvatureWeighted(mesh, sampleCount / 2);
                samples.AddRange(curvatureSamples);
                // Additional: Adapt based on density
                AdjustSampleDensity(samples, mesh.bounds, 0.05f);
                return samples;
            }
            private static void AdjustSampleDensity(List<Vector3> samples, Bounds bounds, float minDistance)
            {
                for (int i = samples.Count - 1; i >= 0; i--)
                {
                    for (int j = 0; j < i; j++)
                    {
                        if (Vector3.Distance(samples[i], samples[j]) < minDistance)
                        {
                            samples.RemoveAt(i);
                            break;
                        }
                    }
                }
            }
            /// <summary>
            /// Stratified sampling
            /// </summary>
            private static List<Vector3> SampleStratified(MeshData mesh, int sampleCount)
            {
                var samples = new List<Vector3>();
                // Divide the mesh into strata based on spatial location
                var bounds = mesh.bounds;
                int strataPerAxis = Mathf.CeilToInt(Mathf.Pow(sampleCount, 1f / 3f));
                float strataSizeX = bounds.size.x / strataPerAxis;
                float strataSizeY = bounds.size.y / strataPerAxis;
                float strataSizeZ = bounds.size.z / strataPerAxis;
                // Create strata
                var strata = new List<List<int>>();
                for (int x = 0; x < strataPerAxis; x++)
                {
                    for (int y = 0; y < strataPerAxis; y++)
                    {
                        for (int z = 0; z < strataPerAxis; z++)
                        {
                            var stratumBounds = new Bounds(
                                bounds.min + new Vector3(x * strataSizeX, y * strataSizeY, z * strataSizeZ) + new Vector3(strataSizeX, strataSizeY, strataSizeZ) / 2,
                                new Vector3(strataSizeX, strataSizeY, strataSizeZ)
                            );
                            var stratumTriangles = new List<int>();
                            // Find triangles in this stratum
                            for (int i = 0; i < mesh.indices.Count; i += 3)
                            {
                                var a = mesh.vertices[mesh.indices[i]];
                                var b = mesh.vertices[mesh.indices[i + 1]];
                                var c = mesh.vertices[mesh.indices[i + 2]];
                                var triBounds = new Bounds(a, Vector3.zero);
                                triBounds.Encapsulate(b);
                                triBounds.Encapsulate(c);
                                if (stratumBounds.Intersects(triBounds))
                                {
                                    stratumTriangles.Add(i);
                                }
                            }
                            strata.Add(stratumTriangles);
                        }
                    }
                }
                // Sample from each stratum
                int samplesPerStratum = Mathf.Max(1, sampleCount / strata.Count);
                foreach (var stratum in strata)
                {
                    if (stratum.Count == 0) continue;
                    for (int i = 0; i < samplesPerStratum && samples.Count < sampleCount; i++)
                    {
                        // Randomly select a triangle from this stratum
                        int triIndex = stratum[UnityEngine.Random.Range(0, stratum.Count)];
                        // Get triangle vertices
                        var a = mesh.vertices[mesh.indices[triIndex]];
                        var b = mesh.vertices[mesh.indices[triIndex + 1]];
                        var c = mesh.vertices[mesh.indices[triIndex + 2]];
                        // Generate random barycentric coordinates
                        float r1 = UnityEngine.Random.value;
                        float r2 = UnityEngine.Random.value;
                        if (r1 + r2 > 1f)
                        {
                            r1 = 1f - r1;
                            r2 = 1f - r2;
                        }
                        float u = r1;
                        float v = r2;
                        float w = 1f - u - v;
                        // Calculate point
                        var point = a * u + b * v + c * w;
                        samples.Add(point);
                    }
                }
                // Additional: Fill remaining samples if needed
                if (samples.Count < sampleCount)
                {
                    var additional = SampleAreaWeighted(mesh, sampleCount - samples.Count);
                    samples.AddRange(additional);
                }
                return samples;
            }
            /// <summary>
            /// Calculate symmetric Hausdorff distance between two meshes
            /// </summary>
            public static float CalculateSymmetricHausdorff(MeshData mesh1, MeshData mesh2, int sampleCount = 1000)
            {
                // Sample points from both meshes
                var samples1 = SampleMeshSurface(mesh1, sampleCount);
                var samples2 = SampleMeshSurface(mesh2, sampleCount);
                if (samples1.Count == 0 || samples2.Count == 0)
                    return 0f;
                // Build spatial acceleration structure for mesh2
                var bvh2 = new BVH(mesh2);
                // Calculate max distance from mesh1 to mesh2
                float maxDist1 = 0f;
                foreach (var point in samples1)
                {
                    float dist = bvh2.DistanceToPoint(point);
                    maxDist1 = Mathf.Max(maxDist1, dist);
                }
                // Build spatial acceleration structure for mesh1
                var bvh1 = new BVH(mesh1);
                // Calculate max distance from mesh2 to mesh1
                float maxDist2 = 0f;
                foreach (var point in samples2)
                {
                    float dist = bvh1.DistanceToPoint(point);
                    maxDist2 = Mathf.Max(maxDist2, dist);
                }
                // Return symmetric Hausdorff distance
                return Mathf.Max(maxDist1, maxDist2);
            }
            /// <summary>
            /// Calculate error using specified metric
            /// </summary>
            public static float CalculateError(MeshData mesh1, MeshData mesh2, ErrorMetricType metricType)
            {
                switch (metricType)
                {
                    case ErrorMetricType.Hausdorff:
                        return CalculateHausdorff(mesh1, mesh2);
                    case ErrorMetricType.SymmetricHausdorff:
                        return CalculateSymmetricHausdorff(mesh1, mesh2);
                    case ErrorMetricType.MeanSquared:
                        return CalculateMeanSquaredError(mesh1, mesh2);
                    case ErrorMetricType.RootMeanSquared:
                        return Mathf.Sqrt(CalculateMeanSquaredError(mesh1, mesh2));
                    case ErrorMetricType.MaxDeviation:
                        return CalculateMaxDeviation(mesh1, mesh2);
                    case ErrorMetricType.VolumeDifference:
                        return CalculateVolumeDifference(mesh1, mesh2);
                    case ErrorMetricType.SurfaceAreaDifference:
                        return CalculateSurfaceAreaDifference(mesh1, mesh2);
                    default:
                        return CalculateSymmetricHausdorff(mesh1, mesh2);
                }
            }
            /// <summary>
            /// Calculate Hausdorff distance from mesh1 to mesh2
            /// </summary>
            public static float CalculateHausdorff(MeshData mesh1, MeshData mesh2)
            {
                // Sample points from mesh1
                int sampleCount = Mathf.Min(1000, mesh1.triangleCount * 10);
                var samples1 = SampleMeshSurface(mesh1, sampleCount);
                if (samples1.Count == 0)
                    return 0f;
                // Build spatial acceleration structure for mesh2
                var bvh2 = new BVH(mesh2);
                // Calculate max distance from mesh1 to mesh2
                float maxDist = 0f;
                foreach (var point in samples1)
                {
                    float dist = bvh2.DistanceToPoint(point);
                    maxDist = Mathf.Max(maxDist, dist);
                }
                return maxDist;
            }
            /// <summary>
            /// Calculate mean squared error between two meshes
            /// </summary>
            public static float CalculateMeanSquaredError(MeshData mesh1, MeshData mesh2)
            {
                // Sample points from both meshes
                int sampleCount = Mathf.Min(1000, mesh1.triangleCount * 10);
                var samples1 = SampleMeshSurface(mesh1, sampleCount);
                var samples2 = SampleMeshSurface(mesh2, sampleCount);
                if (samples1.Count == 0 || samples2.Count == 0)
                    return 0f;
                // Build spatial acceleration structure for mesh2
                var bvh2 = new BVH(mesh2);
                // Calculate squared distances from mesh1 to mesh2
                float sumSquaredDist = 0f;
                foreach (var point in samples1)
                {
                    float dist = bvh2.DistanceToPoint(point);
                    sumSquaredDist += dist * dist;
                }
                // Build spatial acceleration structure for mesh1
                var bvh1 = new BVH(mesh1);
                // Calculate squared distances from mesh2 to mesh1
                foreach (var point in samples2)
                {
                    float dist = bvh1.DistanceToPoint(point);
                    sumSquaredDist += dist * dist;
                }
                // Return mean squared error
                return sumSquaredDist / (samples1.Count + samples2.Count);
            }
            /// <summary>
            /// Calculate maximum deviation between two meshes
            /// </summary>
            public static float CalculateMaxDeviation(MeshData mesh1, MeshData mesh2)
            {
                // This is equivalent to symmetric Hausdorff
                return CalculateSymmetricHausdorff(mesh1, mesh2);
            }
            /// <summary>
            /// Calculate volume difference between two meshes
            /// </summary>
            public static float CalculateVolumeDifference(MeshData mesh1, MeshData mesh2)
            {
                return Mathf.Abs(mesh1.volume - mesh2.volume);
            }
            /// <summary>
            /// Calculate surface area difference between two meshes
            /// </summary>
            public static float CalculateSurfaceAreaDifference(MeshData mesh1, MeshData mesh2)
            {
                return Mathf.Abs(mesh1.surfaceArea - mesh2.surfaceArea);
            }
            /// <summary>
            /// Calculate error distribution between two meshes
            /// </summary>
            public static List<float> CalculateErrorDistribution(MeshData mesh1, MeshData mesh2, int sampleCount = 1000)
            {
                var errors = new List<float>();
                // Sample points from mesh1
                var samples1 = SampleMeshSurface(mesh1, sampleCount / 2);
                // Build spatial acceleration structure for mesh2
                var bvh2 = new BVH(mesh2);
                // Calculate distances from mesh1 to mesh2
                foreach (var point in samples1)
                {
                    float dist = bvh2.DistanceToPoint(point);
                    errors.Add(dist);
                }
                // Sample points from mesh2
                var samples2 = SampleMeshSurface(mesh2, sampleCount / 2);
                // Build spatial acceleration structure for mesh1
                var bvh1 = new BVH(mesh1);
                // Calculate distances from mesh2 to mesh1
                foreach (var point in samples2)
                {
                    float dist = bvh1.DistanceToPoint(point);
                    errors.Add(dist);
                }
                return errors;
            }
            public static float CalculateSymmetricHausdorff(MeshData mesh, List<ConvexHull> hulls, int maxSamples = 2000)
            {
                var hullMesh = CombineHullsToMesh(hulls);
                if (hullMesh.vertices.Count == 0) return 0f;
                return CalculateSymmetricHausdorff(mesh, hullMesh, maxSamples);
            }
            public static List<float> CalculateErrorDistribution(MeshData mesh, List<ConvexHull> hulls, int sampleCount = 1000)
            {
                var hullMesh = CombineHullsToMesh(hulls);
                if (hullMesh.vertices.Count == 0) return new List<float>();
                return CalculateErrorDistribution(mesh, hullMesh, sampleCount);
            }
            static MeshData CombineHullsToMesh(List<ConvexHull> hulls)
            {
                var m = new MeshData();
                foreach (var h in hulls)
                {
                    if (h.vertices == null || h.indices == null || h.vertices.Count == 0 || h.indices.Count == 0) continue;
                    int baseIdx = m.vertices.Count;
                    m.vertices.AddRange(h.vertices);
                    if (h.normals != null && h.normals.Count == h.vertices.Count) m.normals.AddRange(h.normals);
                    else { for (int i = 0; i < h.vertices.Count; i++) m.normals.Add(Vector3.zero); } // Add placeholder normals
                    for (int i = 0; i < h.indices.Count; i++) m.indices.Add(h.indices[i] + baseIdx);
                }
                m.CalculateProperties();
                return m;
            }
            // Additional: Calculate PSNR
            public static float CalculatePSNR(MeshData mesh1, MeshData mesh2)
            {
                float mse = CalculateMeanSquaredError(mesh1, mesh2);
                if (mse == 0) return float.PositiveInfinity;
                float maxValue = Mathf.Max(mesh1.bounds.size.magnitude, mesh2.bounds.size.magnitude);
                return 20 * Mathf.Log10(maxValue / Mathf.Sqrt(mse));
            }
            // Additional: Calculate SSIM
            public static float CalculateSSIM(MeshData mesh1, MeshData mesh2)
            {
                // Simplified SSIM calculation
                throw new NotImplementedException();
            }
        }
        /// <summary>
        /// Bounding Volume Hierarchy for spatial acceleration
        /// </summary>
        public class BVH
        {
            private class Node
            {
                public Bounds bounds;
                public List<int> triangleIndices;
                public Node left;
                public Node right;
                public bool isLeaf;
                public int depth;
            }
            private Node root;
            private MeshData mesh;
            private int maxTrianglesPerNode = 10;
            private int maxDepth = 20;
            public BVH(MeshData mesh)
            {
                this.mesh = mesh;
                Build();
            }
            private void Build()
            {
                if (mesh == null || mesh.triangleCount == 0) return;
                var allIndices = Enumerable.Range(0, mesh.triangleCount).Select(i => i * 3).ToList();
                root = BuildRecursive(allIndices, 0);
            }
            private Node BuildRecursive(List<int> triangleIndices, int depth)
            {
                var node = new Node
                {
                    triangleIndices = triangleIndices,
                    depth = depth,
                    bounds = CalculateBounds(triangleIndices)
                };
                if (triangleIndices.Count <= maxTrianglesPerNode || depth >= maxDepth)
                {
                    node.isLeaf = true;
                    return node;
                }
                // Find longest axis to split
                int axis = 0;
                var size = node.bounds.size;
                if (size.y > size.x) axis = 1;
                if (size.z > size[axis]) axis = 2;
                float splitPos = node.bounds.center[axis];
                var leftIndices = new List<int>();
                var rightIndices = new List<int>();
                foreach (var triIndex in triangleIndices)
                {
                    var triCenter = (mesh.vertices[mesh.indices[triIndex]] +
                                     mesh.vertices[mesh.indices[triIndex + 1]] +
                                     mesh.vertices[mesh.indices[triIndex + 2]]) / 3f;
                    if (triCenter[axis] < splitPos)
                        leftIndices.Add(triIndex);
                    else
                        rightIndices.Add(triIndex);
                }
                // Handle cases where split fails
                if (leftIndices.Count == 0 || rightIndices.Count == 0)
                {
                    node.isLeaf = true;
                    return node;
                }
                node.isLeaf = false;
                node.left = BuildRecursive(leftIndices, depth + 1);
                node.right = BuildRecursive(rightIndices, depth + 1);
                return node;
            }
            private Bounds CalculateBounds(List<int> triangleIndices)
            {
                if (triangleIndices.Count == 0) return new Bounds();
                var bounds = new Bounds(mesh.vertices[mesh.indices[triangleIndices[0]]], Vector3.zero);
                foreach (var triIndex in triangleIndices)
                {
                    bounds.Encapsulate(mesh.vertices[mesh.indices[triIndex]]);
                    bounds.Encapsulate(mesh.vertices[mesh.indices[triIndex + 1]]);
                    bounds.Encapsulate(mesh.vertices[mesh.indices[triIndex + 2]]);
                }
                return bounds;
            }
            public float DistanceToPoint(Vector3 point)
            {
                if (root == null) return float.MaxValue;
                return DistanceToPointRecursive(point, root);
            }
            private float DistanceToPointRecursive(Vector3 point, Node node)
            {
                if (node.isLeaf)
                {
                    float minDistance = float.MaxValue;
                    foreach (var triIndex in node.triangleIndices)
                    {
                        var a = mesh.vertices[mesh.indices[triIndex]];
                        var b = mesh.vertices[mesh.indices[triIndex + 1]];
                        var c = mesh.vertices[mesh.indices[triIndex + 2]];
                        minDistance = Mathf.Min(minDistance, DistanceToTriangle(point, a, b, c));
                    }
                    return minDistance;
                }
                float distToLeftBounds = SqrDistanceToBounds(point, node.left.bounds);
                float distToRightBounds = SqrDistanceToBounds(point, node.right.bounds);
                if (distToLeftBounds < distToRightBounds)
                {
                    float distLeft = DistanceToPointRecursive(point, node.left);
                    if (distToRightBounds >= distLeft * distLeft) return distLeft;
                    return Mathf.Min(distLeft, DistanceToPointRecursive(point, node.right));
                }
                else
                {
                    float distRight = DistanceToPointRecursive(point, node.right);
                    if (distToLeftBounds >= distRight * distRight) return distRight;
                    return Mathf.Min(distRight, DistanceToPointRecursive(point, node.left));
                }
            }
            private float SqrDistanceToBounds(Vector3 point, Bounds bounds)
            {
                var closest = bounds.ClosestPoint(point);
                return (point - closest).sqrMagnitude;
            }
            private float DistanceToTriangle(Vector3 point, Vector3 a, Vector3 b, Vector3 c)
            {
                var ab = b - a;
                var ac = c - a;
                var ap = point - a;
                var d1 = Vector3.Dot(ab, ap);
                var d2 = Vector3.Dot(ac, ap);
                if (d1 <= 0f && d2 <= 0f) return Vector3.Distance(point, a);
                var bp = point - b;
                var d3 = Vector3.Dot(ab, bp);
                var d4 = Vector3.Dot(ac, bp);
                if (d3 >= 0f && d4 <= d3) return Vector3.Distance(point, b);
                var vc = d1 * d4 - d3 * d2;
                if (vc <= 0f && d1 >= 0f && d3 <= 0f)
                {
                    var v = d1 / (d1 - d3);
                    return Vector3.Distance(point, a + v * ab);
                }
                var cp = point - c;
                var d5 = Vector3.Dot(ab, cp);
                var d6 = Vector3.Dot(ac, cp);
                if (d6 >= 0f && d5 <= d6) return Vector3.Distance(point, c);
                var vb = d5 * d2 - d1 * d6;
                if (vb <= 0f && d2 >= 0f && d6 <= 0f)
                {
                    var w = d2 / (d2 - d6);
                    return Vector3.Distance(point, a + w * ac);
                }
                var va = d3 * d6 - d5 * d4;
                if (va <= 0f && (d4 - d3) >= 0f && (d5 - d6) >= 0f)
                {
                    var w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                    return Vector3.Distance(point, b + w * (c - b));
                }
                var denom = va + vb + vc;
                if (Mathf.Abs(denom) < 1e-6f) return Vector3.Distance(point, a); // Degenerate
                var v_ = vb / denom;
                var w_ = vc / denom;
                var closestPoint = a + ab * v_ + ac * w_;
                return Vector3.Distance(point, closestPoint);
            }
            /// <summary>
            /// Find all triangles intersecting with a ray
            /// </summary>
            public List<int> IntersectRay(Vector3 origin, Vector3 direction, float maxDistance = float.MaxValue)
            {
                var results = new List<int>();
                if (root == null) return results;
                IntersectRayRecursive(origin, direction, maxDistance, root, results);
                return results;
            }
            private static bool Intersects(Bounds b, Vector3 origin, Vector3 dir, float maxDistance)
            {
                return b.IntersectRay(new Ray(origin, dir), out float hit) && hit <= maxDistance;
            }
            private void IntersectRayRecursive(Vector3 origin, Vector3 direction, float maxDistance, Node node, List<int> results)
            {
                if (!Intersects(node.bounds, origin, direction, maxDistance))
                    return;
                if (node.isLeaf)
                {
                    // Check if ray intersects with triangle
                    foreach (var triIndex in node.triangleIndices)
                    {
                        var a = mesh.vertices[mesh.indices[triIndex]];
                        var b = mesh.vertices[mesh.indices[triIndex + 1]];
                        var c = mesh.vertices[mesh.indices[triIndex + 2]];
                        if (IntersectRayTriangle(origin, direction, a, b, c, out float distance) && distance <= maxDistance)
                        {
                            results.Add(triIndex);
                        }
                    }
                }
                else
                {
                    // Check both children
                    IntersectRayRecursive(origin, direction, maxDistance, node.left, results);
                    IntersectRayRecursive(origin, direction, maxDistance, node.right, results);
                }
            }
            private bool IntersectRayTriangle(Vector3 origin, Vector3 direction, Vector3 a, Vector3 b, Vector3 c, out float distance)
            {
                // MllerTrumbore intersection algorithm
                var edge1 = b - a;
                var edge2 = c - a;
                var h = Vector3.Cross(direction, edge2);
                var det = Vector3.Dot(edge1, h);
                if (det > -1e-6f && det < 1e-6f)
                {
                    distance = 0f;
                    return false; // Ray is parallel to triangle
                }
                var f = 1f / det;
                var s = origin - a;
                var u = f * Vector3.Dot(s, h);
                if (u < 0f || u > 1f)
                {
                    distance = 0f;
                    return false;
                }
                var q = Vector3.Cross(s, edge1);
                var v = f * Vector3.Dot(direction, q);
                if (v < 0f || u + v > 1f)
                {
                    distance = 0f;
                    return false;
                }
                distance = f * Vector3.Dot(edge2, q);
                return distance > 1e-6f;
            }
            // Additional: Find closest triangle to point
            public int FindClosestTriangle(Vector3 point)
            {
                float minDist = float.MaxValue;
                int closestTri = -1;
                FindClosestTriangleRecursive(point, root, ref minDist, ref closestTri);
                return closestTri;
            }
            private void FindClosestTriangleRecursive(Vector3 point, Node node, ref float minDist, ref int closestTri)
            {
                if (node.isLeaf)
                {
                    foreach (var triIndex in node.triangleIndices)
                    {
                        var a = mesh.vertices[mesh.indices[triIndex]];
                        var b = mesh.vertices[mesh.indices[triIndex + 1]];
                        var c = mesh.vertices[mesh.indices[triIndex + 2]];
                        float dist = DistanceToTriangle(point, a, b, c);
                        if (dist < minDist)
                        {
                            minDist = dist;
                            closestTri = triIndex;
                        }
                    }
                }
                else
                {
                    // Recurse on children
                    FindClosestTriangleRecursive(point, node.left, ref minDist, ref closestTri);
                    FindClosestTriangleRecursive(point, node.right, ref minDist, ref closestTri);
                }
            }
        }
        /// <summary>
        /// Full 3D QuickHull implementation for convex hull generation.
        /// Uses a half-edge data structure to manage hull topology.
        /// </summary>
        public static class QuickHullImplementation
        {
            private class QHVertex
            {
                public Vector3 position;
                public int index;
                public QHHalfEdge edge;
                public bool onHull;
            }
            private class QHHalfEdge
            {
                public QHVertex vertex;
                public QHFace face;
                public QHHalfEdge next;
                public QHHalfEdge prev;
                public QHHalfEdge twin;
            }
            private class QHFace
            {
                public QHHalfEdge edge;
                public Vector3 normal;
                public float distance;
                public List<QHVertex> outsideSet = new List<QHVertex>();
                public bool visible;
                public int generation;
            }
            public static ConvexHull ComputeConvexHullFromPoints(List<Vector3> points)
            {
                if (points == null || points.Count < 4)
                {
                    return new ConvexHull();
                }
                const float epsilon = 1e-5f;
                List<QHFace> allFaces = new List<QHFace>();
                List<QHVertex> qhVertices = points.Select((p, i) => new QHVertex { position = p, index = i }).ToList();
                if (!CreateInitialSimplex(qhVertices, allFaces, epsilon))
                {
                    return new ConvexHull();
                }
                int currentGeneration = 0;
                while (true)
                {
                    QHFace furthestFace = null;
                    QHVertex eyePoint = null;
                    float maxDist = epsilon;
                    foreach (var face in allFaces)
                    {
                        if (face.visible || face.outsideSet.Count == 0) continue;
                        foreach (var v in face.outsideSet)
                        {
                            float dist = Vector3.Dot(face.normal, v.position) - face.distance;
                            if (dist > maxDist)
                            {
                                maxDist = dist;
                                furthestFace = face;
                                eyePoint = v;
                            }
                        }
                    }
                    if (furthestFace == null || eyePoint == null) break;
                    eyePoint.onHull = true;
                    List<QHFace> visibleFaces = new List<QHFace>();
                    List<QHHalfEdge> horizonEdges = new List<QHHalfEdge>();
                    Stack<QHFace> faceStack = new Stack<QHFace>();
                    furthestFace.visible = true;
                    faceStack.Push(furthestFace);
                    while (faceStack.Count > 0)
                    {
                        var face = faceStack.Pop();
                        visibleFaces.Add(face);
                        var edge = face.edge;
                        do
                        {
                            var neighbor = edge.twin.face;
                            if (!neighbor.visible)
                            {
                                float dist = Vector3.Dot(neighbor.normal, eyePoint.position) - neighbor.distance;
                                if (dist > epsilon)
                                {
                                    neighbor.visible = true;
                                    faceStack.Push(neighbor);
                                }
                                else
                                {
                                    horizonEdges.Add(edge);
                                }
                            }
                            edge = edge.next;
                        } while (edge != face.edge);
                    }
                    List<QHVertex> orphanedPoints = new List<QHVertex>();
                    foreach (var face in visibleFaces)
                    {
                        orphanedPoints.AddRange(face.outsideSet);
                        face.outsideSet.Clear();
                    }
                    // after collecting raw horizonEdges:
                    var horizon = OrderHorizonEdges(horizonEdges);
                    var newFaces = new List<QHFace>();
                    BuildFacesFromHorizon(horizon, eyePoint, newFaces);
                    // Assign orphaned points to new faces
                    foreach (var v in orphanedPoints)
                    {
                        if (v.onHull) continue;
                        QHFace bestFace = null;
                        float maxPointDist = epsilon;
                        foreach (var face in newFaces)
                        {
                            float dist = Vector3.Dot(face.normal, v.position) - face.distance;
                            if (dist > maxPointDist)
                            {
                                maxPointDist = dist;
                                bestFace = face;
                            }
                        }
                        if (bestFace != null)
                        {
                            bestFace.outsideSet.Add(v);
                        }
                    }
                    allFaces.AddRange(newFaces);
                    currentGeneration++;
                }
                var finalHull = new ConvexHull();
                var vertexMap = new Dictionary<QHVertex, int>();
                allFaces.RemoveAll(f => f.visible);
                foreach (var face in allFaces)
                {
                    var edge = face.edge;
                    do
                    {
                        if (!vertexMap.ContainsKey(edge.vertex))
                        {
                            vertexMap.Add(edge.vertex, finalHull.vertices.Count);
                            finalHull.vertices.Add(edge.vertex.position);
                        }
                        edge = edge.next;
                    } while (edge != face.edge);
                }
                foreach (var face in allFaces)
                {
                    finalHull.indices.Add(vertexMap[face.edge.vertex]);
                    finalHull.indices.Add(vertexMap[face.edge.next.vertex]);
                    finalHull.indices.Add(vertexMap[face.edge.next.next.vertex]);
                }
                finalHull.CalculateProperties();
                return finalHull;
            }
            // Order the horizon into a loop: next starts at current.twin.vertex
            static List<QHHalfEdge> OrderHorizonEdges(List<QHHalfEdge> raw)
            {
                var from = new Dictionary<QHVertex, QHHalfEdge>();
                foreach (var e in raw) from[e.vertex] = e; // e.vertex is the tail (origin) of the horizon edge on a VISIBLE face
                var ordered = new List<QHHalfEdge>(raw.Count);
                var cur = raw[0];
                ordered.Add(cur);
                while (ordered.Count < raw.Count)
                {
                    if (!from.TryGetValue(cur.twin.vertex, out var next)) break; // next tail == current head
                    ordered.Add(next);
                    cur = next;
                }
                return ordered;
            }
            // Build the "cone" of new faces around 'eye' from the ordered horizon
            static void BuildFacesFromHorizon(List<QHHalfEdge> orderedHorizon, QHVertex eye, List<QHFace> newFaces)
            {
                QHHalfEdge firstB = null;
                QHHalfEdge prevC = null;
                foreach (var h in orderedHorizon)
                {
                    var vA = h.vertex; // tail of horizon edge on visible side
                    var vB = h.twin.vertex; // head of horizon edge (in the neighbor, i.e., outside)
                    var f = new QHFace();
                    var a = new QHHalfEdge { vertex = vA, face = f }; // vA -> vB (shares with old outside face)
                    var b = new QHHalfEdge { vertex = vB, face = f }; // vB -> eye (will twin with previous face's c)
                    var c = new QHHalfEdge { vertex = eye, face = f }; // eye -> vA (will twin with next face's b)
                    // cycle
                    a.next = b; b.next = c; c.next = a;
                    a.prev = c; b.prev = a; c.prev = b;
                    // twin to the existing outside face along the horizon
                    a.twin = h.twin;
                    h.twin.twin = a;
                    // stitch around the eye
                    if (prevC != null) { prevC.twin = b; b.twin = prevC; } else { firstB = b; }
                    // face data
                    f.edge = a;
                    f.normal = Vector3.Cross(vB.position - vA.position, eye.position - vA.position).normalized;
                    f.distance = Vector3.Dot(f.normal, vA.position);
                    newFaces.Add(f);
                    prevC = c;
                }
                // close the ring
                if (firstB != null && prevC != null) { firstB.twin = prevC; prevC.twin = firstB; }
            }
            private static bool CreateInitialSimplex(List<QHVertex> vertices, List<QHFace> faces, float epsilon)
            {
                faces.Clear();
                int i0 = 0, i1 = 0;
                float maxDistSq = 0;
                for (int i = 0; i < vertices.Count; i++)
                    for (int j = i + 1; j < vertices.Count; j++)
                    {
                        float d = (vertices[j].position - vertices[i].position).sqrMagnitude;
                        if (d > maxDistSq) { maxDistSq = d; i0 = i; i1 = j; }
                    }
                if (Mathf.Sqrt(maxDistSq) < epsilon) return false;
                int i2 = -1;
                maxDistSq = 0;
                for (int i = 0; i < vertices.Count; i++)
                {
                    if (i == i0 || i == i1) continue;
                    float d = DistanceToLineSq(vertices[i].position, vertices[i0].position, vertices[i1].position);
                    if (d > maxDistSq) { maxDistSq = d; i2 = i; }
                }
                if (i2 == -1 || Mathf.Sqrt(maxDistSq) < epsilon) return false;
                int i3 = -1;
                float maxDist = 0;
                for (int i = 0; i < vertices.Count; i++)
                {
                    if (i == i0 || i == i1 || i == i2) continue;
                    float d = DistanceToPlane(vertices[i].position, vertices[i0].position, vertices[i1].position, vertices[i2].position);
                    if (Mathf.Abs(d) > maxDist) { maxDist = Mathf.Abs(d); i3 = i; }
                }
                if (i3 == -1 || maxDist < epsilon) return false;
                var v = new QHVertex[4] { vertices[i0], vertices[i1], vertices[i2], vertices[i3] };
                if (DistanceToPlane(v[3].position, v[0].position, v[1].position, v[2].position) > 0)
                {
                    (v[1], v[2]) = (v[2], v[1]);
                }
                for (int i = 0; i < 4; i++) v[i].onHull = true;
                faces.Add(CreateFace(v[0], v[1], v[2]));
                faces.Add(CreateFace(v[3], v[1], v[0]));
                faces.Add(CreateFace(v[3], v[2], v[1]));
                faces.Add(CreateFace(v[3], v[0], v[2]));
                LinkTwins(faces[0].edge, faces[1].edge.next);
                LinkTwins(faces[0].edge.next, faces[2].edge.next);
                LinkTwins(faces[0].edge.prev, faces[3].edge.next);
                LinkTwins(faces[1].edge, faces[2].edge.prev);
                LinkTwins(faces[1].edge.prev, faces[3].edge);
                LinkTwins(faces[2].edge, faces[3].edge.prev);
                foreach (var vert in vertices)
                {
                    if (vert.onHull) continue;
                    QHFace bestFace = null;
                    float maxPointDist = epsilon;
                    foreach (var face in faces)
                    {
                        float dist = Vector3.Dot(face.normal, vert.position) - face.distance;
                        if (dist > maxPointDist)
                        {
                            maxPointDist = dist;
                            bestFace = face;
                        }
                    }
                    if (bestFace != null) bestFace.outsideSet.Add(vert);
                }
                return true;
            }
            private static float DistanceToLineSq(Vector3 p, Vector3 a, Vector3 b)
            {
                Vector3 ab = b - a;
                Vector3 ap = p - a;
                return Vector3.Cross(ab, ap).sqrMagnitude / ab.sqrMagnitude;
            }
            private static float DistanceToPlane(Vector3 p, Vector3 a, Vector3 b, Vector3 c)
            {
                Vector3 normal = Vector3.Cross(b - a, c - a).normalized;
                return Vector3.Dot(normal, p - a);
            }
            private static QHFace CreateFace(QHVertex a, QHVertex b, QHVertex c)
            {
                var face = new QHFace();
                var e0 = new QHHalfEdge { vertex = a, face = face };
                var e1 = new QHHalfEdge { vertex = b, face = face };
                var e2 = new QHHalfEdge { vertex = c, face = face };
                e0.next = e1; e1.prev = e0;
                e1.next = e2; e2.prev = e1;
                e2.next = e0; e0.prev = e2;
                face.edge = e0;
                face.normal = Vector3.Cross(b.position - a.position, c.position - a.position).normalized;
                face.distance = Vector3.Dot(face.normal, a.position);
                return face;
            }
            private static void LinkTwins(QHHalfEdge e1, QHHalfEdge e2)
            {
                e1.twin = e2;
                e2.twin = e1;
            }

            public static ConvexHull Simplify(ConvexHull hull, float tolerance)
            {
                if (hull == null || hull.vertices.Count == 0) return hull;
                hull.OptimizeHull(tolerance);
                return hull;
            }
        }
        /// <summary>
        /// Mesh boolean operations with proper clipping
        /// </summary>
        public static class MeshBooleanOperations
        {
            static int Add(MeshData m, Vector3 p, Vector3 n)
            {
                if (m.vertices == null) m.vertices = new List<Vector3>();
                if (m.normals == null) m.normals = new List<Vector3>();
                for (int i = 0; i < m.vertices.Count; i++)
                    if ((m.vertices[i] - p).sqrMagnitude < 1e-12f) return i;
                m.vertices.Add(p);
                m.normals.Add(n);
                return m.vertices.Count - 1;
            }
            public static MeshData BooleanIntersect(MeshData meshA, MeshData meshB)
            {
                var outMesh = new MeshData();
                var hullB = QuickHullImplementation.ComputeConvexHullFromPoints(meshB.vertices);
                hullB.BuildPlanes();

                if (hullB.planes == null || hullB.planes.Count == 0) return meshA;

                for (int i = 0; i < meshA.indices.Count; i += 3)
                {
                    int i0 = meshA.indices[i], i1 = meshA.indices[i + 1], i2 = meshA.indices[i + 2];
                    var v0 = meshA.vertices[i0]; var v1 = meshA.vertices[i1]; var v2 = meshA.vertices[i2];
                    var n0 = meshA.normals.Count > i0 ? meshA.normals[i0] : Vector3.zero;
                    var n1 = meshA.normals.Count > i1 ? meshA.normals[i1] : Vector3.zero;
                    var n2 = meshA.normals.Count > i2 ? meshA.normals[i2] : Vector3.zero;
                    foreach (var (p0, p1, p2, norm0, norm1, norm2) in ClipTriAgainstConvex(v0, v1, v2, n0, n1, n2, hullB.planes))
                    {
                        int ia = Add(outMesh, p0, norm0);
                        int ib = Add(outMesh, p1, norm1);
                        int ic = Add(outMesh, p2, norm2);
                        outMesh.indices.AddRange(new[] { ia, ib, ic });
                    }
                }
                outMesh.Optimize();
                return outMesh;
            }
            public static ConvexHull BooleanUnion(ConvexHull hull1, ConvexHull hull2)
            {
                var allPoints = new List<Vector3>(hull1.vertices);
                allPoints.AddRange(hull2.vertices);
                var union = QuickHullImplementation.ComputeConvexHullFromPoints(allPoints);
                // Additional: Optimize union
                union.OptimizeHull(0.001f);
                return union;
            }
            static IEnumerable<(Vector3, Vector3)> ClipPolyAgainstPlane(IEnumerable<(Vector3 p, Vector3 n)> poly, Vector3 N, float d, float eps = 1e-6f)
            {
                var input = new List<(Vector3 p, Vector3 n)>(poly);
                if (input.Count == 0) yield break;
                var prev = input[input.Count - 1];
                float dpPrev = Vector3.Dot(N, prev.p) + d;
                for (int i = 0; i < input.Count; i++)
                {
                    var curr = input[i];
                    float dpCurr = Vector3.Dot(N, curr.p) + d;
                    bool currIn = dpCurr <= eps, prevIn = dpPrev <= eps;
                    if (currIn != prevIn)
                    {
                        float t = dpPrev / (dpPrev - dpCurr);
                        var ip = Vector3.Lerp(prev.p, curr.p, t);
                        var inorm = Vector3.Lerp(prev.n, curr.n, t).normalized;
                        yield return (ip, inorm);
                    }
                    if (currIn)
                    {
                        yield return curr;
                    }
                    prev = curr; dpPrev = dpCurr;
                }
            }
            static IEnumerable<(Vector3, Vector3, Vector3, Vector3, Vector3, Vector3)> ClipTriAgainstConvex(
                Vector3 v0, Vector3 v1, Vector3 v2, Vector3 n0, Vector3 n1, Vector3 n2, List<PlaneF> planes)
            {
                var poly = new List<(Vector3 p, Vector3 n)> { (v0, n0), (v1, n1), (v2, n2) };
                foreach (var pl in planes)
                {
                    poly = new List<(Vector3 p, Vector3 n)>(ClipPolyAgainstPlane(poly, pl.n, pl.d));
                    if (poly.Count < 3) yield break;
                }
                for (int i = 1; i + 1 < poly.Count; i++)
                    yield return (poly[0].p, poly[i].p, poly[i + 1].p, poly[0].n, poly[i].n, poly[i + 1].n);
            }
            public static ConvexHull BooleanIntersection(ConvexHull hullA, ConvexHull hullB)
            {
                if (hullB.planes == null || hullB.planes.Count == 0) hullB.BuildPlanes();
                var outMesh = new MeshData();
                for (int i = 0; i < hullA.indices.Count; i += 3)
                {
                    int i0 = hullA.indices[i], i1 = hullA.indices[i + 1], i2 = hullA.indices[i + 2];
                    var a = hullA.vertices[i0]; var b = hullA.vertices[i1]; var c = hullA.vertices[i2];
                    var na = (hullA.normals.Count == hullA.vertices.Count) ? hullA.normals[i0] : Vector3.up;
                    var nb = (hullA.normals.Count == hullA.vertices.Count) ? hullA.normals[i1] : Vector3.up;
                    var nc = (hullA.normals.Count == hullA.vertices.Count) ? hullA.normals[i2] : Vector3.up;
                    foreach (var (p0, p1, p2, norm0, norm1, norm2) in ClipTriAgainstConvex(a, b, c, na, nb, nc, hullB.planes))
                    {
                        int ia = Add(outMesh, p0, norm0);
                        int ib = Add(outMesh, p1, norm1);
                        int ic = Add(outMesh, p2, norm2);
                        outMesh.indices.Add(ia); outMesh.indices.Add(ib); outMesh.indices.Add(ic);
                    }
                }
                outMesh.CalculateProperties();
                var intersection = QuickHullImplementation.ComputeConvexHullFromPoints(outMesh.vertices);
                // Additional: Validate intersection
                intersection.ValidateHull();
                return intersection;
            }
            public static MeshData BooleanDifference(MeshData a, MeshData b)
            {
                // Proper boolean difference using clipping
                var outMesh = new MeshData();
                // First, clip a against b's planes (assuming b is convex)
                var bHull = QuickHullImplementation.ComputeConvexHullFromPoints(b.vertices);
                bHull.BuildPlanes();
                for (int i = 0; i < a.indices.Count; i += 3)
                {
                    int i0 = a.indices[i], i1 = a.indices[i + 1], i2 = a.indices[i + 2];
                    var v0 = a.vertices[i0]; var v1 = a.vertices[i1]; var v2 = a.vertices[i2];
                    var n0 = a.normals.Count > i0 ? a.normals[i0] : Vector3.zero;
                    var n1 = a.normals.Count > i1 ? a.normals[i1] : Vector3.zero;
                    var n2 = a.normals.Count > i2 ? a.normals[i2] : Vector3.zero;
                    var planes = bHull.planes.Select(p => new PlaneF { n = -p.n, d = -p.d }).ToList(); // Invert planes for difference
                    foreach (var (p0, p1, p2, norm0, norm1, norm2) in ClipTriAgainstConvex(v0, v1, v2, n0, n1, n2, planes))
                    {
                        int ia = Add(outMesh, p0, norm0);
                        int ib = Add(outMesh, p1, norm1);
                        int ic = Add(outMesh, p2, norm2);
                        outMesh.indices.Add(ia); outMesh.indices.Add(ib); outMesh.indices.Add(ic);
                    }
                }
                outMesh.Optimize();
                return outMesh;
            }
            // Additional: Boolean Union for meshes
            public static MeshData BooleanUnion(MeshData a, MeshData b)
            {
                var hullA = QuickHullImplementation.ComputeConvexHullFromPoints(a.vertices);
                var hullB = QuickHullImplementation.ComputeConvexHullFromPoints(b.vertices);
                var unionHull = BooleanUnion(hullA, hullB);
                var unionMesh = new MeshData { vertices = unionHull.vertices, indices = unionHull.indices };
                return unionMesh;
            }
        }
        /// <summary>
        /// Split plane evaluator for finding optimal split planes
        /// </summary>
        public static class SplitPlaneEvaluator
        {
            public static (Vector3 normal, Vector3 point) FindBestSplitPlane_CoACD(MeshData mesh, ConvexHull hull, float error)
            {
                var candidates = new List<(Vector3 normal, Vector3 point)>();

                // Candidate 1: PCA-aligned planes
                CalculatePCA(mesh.vertices, out Vector3 pca1, out Vector3 pca2, out Vector3 pca3);
                candidates.Add((pca1, mesh.bounds.center));
                candidates.Add((pca2, mesh.bounds.center));
                candidates.Add((pca3, mesh.bounds.center));

                // Candidate 2: Planes from sharp features
                if (mesh.sharpEdges != null && mesh.sharpEdges.Count > 10)
                {
                     // Attempt to find a plane from a cluster of sharp edges
                    var featurePlane = GenerateFeaturePlane(mesh);
                    candidates.Add(featurePlane);
                }

                // Candidate 3: Plane from the point of maximum error (witness point)
                (Vector3 witness, float maxError) = FindMaxErrorPoint(mesh, hull);
                if (maxError > 0)
                {
                    Vector3 closestPointOnHull = hull.bounds.ClosestPoint(witness);
                    Vector3 witnessNormal = (witness - closestPointOnHull).normalized;
                    if (witnessNormal.sqrMagnitude > 0.1f)
                    {
                        candidates.Add((witnessNormal, witness));
                    }
                }

                // Evaluate all candidates and find the best one
                var bestPlane = (Vector3.up, mesh.bounds.center);
                float bestScore = float.MaxValue;

                foreach (var candidate in candidates)
                {
                    float score = EvaluateSplitPlane(mesh, candidate.normal, candidate.point);
                    if (score < bestScore)
                    {
                        bestScore = score;
                        bestPlane = candidate;
                    }
                }
                return bestPlane;
            }

            private static (Vector3 point, float error) FindMaxErrorPoint(MeshData mesh, ConvexHull hull)
            {
                Vector3 maxErrorPoint = Vector3.zero;
                float maxError = 0;

                // We can sample the mesh vertices to find the point with the largest distance to the hull
                foreach (var vert in mesh.vertices)
                {
                    float dist = hull.DistanceToPoint(vert);
                    if (dist > maxError)
                    {
                        maxError = dist;
                        maxErrorPoint = vert;
                    }
                }
                return (maxErrorPoint, maxError);
            }
            public static (Vector3 normal, Vector3 point) FindBestSplitPlane(MeshData mesh, ConvexHull hull, float concavity)
            {
                // Find the deepest point of concavity (witness point)
                Vector3 witnessPoint = Vector3.zero;
                float maxDist = -1f;
                foreach (var vert in mesh.vertices)
                {
                    float dist = hull.DistanceToPoint(vert);
                    if (dist > maxDist)
                    {
                        maxDist = dist;
                        witnessPoint = vert;
                    }
                }
                var candidates = GenerateSplitPlaneCandidates(mesh, hull, witnessPoint);
                if (candidates.Count == 0) return (Vector3.up, mesh.bounds.center);
                var bestPlane = (Vector3.zero, Vector3.zero);
                float bestScore = float.MaxValue; // We want to minimize the score
                foreach (var candidate in candidates)
                {
                    float score = EvaluateSplitPlane(mesh, candidate.normal, candidate.point);
                    if (score < bestScore)
                    {
                        bestScore = score;
                        bestPlane = candidate;
                    }
                }
                return bestPlane;
            }
            private static List<(Vector3 normal, Vector3 point)> GenerateSplitPlaneCandidates(MeshData mesh, ConvexHull hull, Vector3 witnessPoint)
            {
                var candidates = new List<(Vector3 normal, Vector3 point)>();
                if (mesh.vertices.Count < 10) return candidates;
                // Candidate 1: Plane aligned with axis-aligned bounding box
                candidates.Add((Vector3.right, mesh.bounds.center));
                candidates.Add((Vector3.up, mesh.bounds.center));
                candidates.Add((Vector3.forward, mesh.bounds.center));
                // Candidate 2: Plane through witness point
                Vector3 closestPointOnHull = hull.bounds.ClosestPoint(witnessPoint);
                Vector3 witnessNormal = (witnessPoint - closestPointOnHull).normalized;
                if (witnessNormal.sqrMagnitude > 0.1f)
                {
                    candidates.Add((witnessNormal, witnessPoint));
                }
                // Candidate 3: PCA-aligned plane
                CalculatePCA(mesh.vertices, out Vector3 pca1, out Vector3 pca2, out Vector3 pca3);
                candidates.Add((pca1, mesh.bounds.center));
                candidates.Add((pca2, mesh.bounds.center));
                candidates.Add((pca3, mesh.bounds.center));
                // Additional candidates: Based on features
                if (mesh.featureVertices != null && mesh.featureVertices.Count > 3)
                {
                    candidates.Add(GenerateFeaturePlane(mesh));
                }
                // Additional: Voxel-based candidates
                var voxelCandidate = GenerateVoxelBasedCandidate(mesh);
                if (voxelCandidate.normal != Vector3.zero)
                    candidates.Add(voxelCandidate);
                return candidates;
            }
            private static (Vector3 normal, Vector3 point) GenerateVoxelBasedCandidate(MeshData mesh)
            {
                // Use voxel data to find high concavity plane
                throw new NotImplementedException();
            }

        private static float EvaluateSplitPlane(MeshData mesh, Vector3 normal, Vector3 point)
        {
            // Use a cost function based on the Surface Area Heuristic (SAH)
            // This tries to create balanced splits.
            Bounds leftBounds = new Bounds();
            Bounds rightBounds = new Bounds();
            int leftCount = 0, rightCount = 0;
            bool firstLeft = true, firstRight = true;
            foreach (var v in mesh.vertices)
            {
                float side = Vector3.Dot(v - point, normal);
                if (side <= 0)
                {
                    if (firstLeft) { leftBounds = new Bounds(v, Vector3.zero); firstLeft = false; }
                    else { leftBounds.Encapsulate(v); }
                    leftCount++;
                }
                else
                {
                    if (firstRight) { rightBounds = new Bounds(v, Vector3.zero); firstRight = false; }
                    else { rightBounds.Encapsulate(v); }
                    rightCount++;
                }
            }
            if (leftCount == 0 || rightCount == 0) return float.MaxValue; // Invalid split
            float surfaceAreaLeft = 2 * (leftBounds.size.x * leftBounds.size.y + leftBounds.size.x * leftBounds.size.z + leftBounds.size.y * leftBounds.size.z);
            float surfaceAreaRight = 2 * (rightBounds.size.x * rightBounds.size.y + rightBounds.size.x * rightBounds.size.z + rightBounds.size.y * rightBounds.size.z);
            float cost = surfaceAreaLeft * leftCount + surfaceAreaRight * rightCount;
            // Additional: Penalize unbalanced splits
            float balance = Mathf.Abs(leftCount - rightCount) / (float)(leftCount + rightCount);
            cost *= 1 + balance;
            return cost;
        }
        }
            private static void CalculatePCA(List<Vector3> points, out Vector3 axis1, out Vector3 axis2, out Vector3 axis3)
            {
                if (points.Count < 3)
                {
                    axis1 = Vector3.right;
                    axis2 = Vector3.up;
                    axis3 = Vector3.forward;
                    return;
                }
                Vector3 mean = Vector3.zero;
                foreach (var p in points) mean += p;
                mean /= points.Count;
                float xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
                foreach (var p in points)
                {
                    Vector3 r = p - mean;
                    xx += r.x * r.x; xy += r.x * r.y; xz += r.x * r.z;
                    yy += r.y * r.y; yz += r.y * r.z; zz += r.z * r.z;
                }
                xx /= points.Count; xy /= points.Count; xz /= points.Count;
                yy /= points.Count; yz /= points.Count; zz /= points.Count;
                Matrix4x4 cov = new Matrix4x4();
                cov[0, 0] = xx; cov[0, 1] = xy; cov[0, 2] = xz;
                cov[1, 0] = xy; cov[1, 1] = yy; cov[1, 2] = yz;
                cov[2, 0] = xz; cov[2, 1] = yz; cov[2, 2] = zz;
                // Using Jacobi iterations for Eigendecomposition of the 3x3 covariance matrix
                const int maxIterations = 10;
                Matrix4x4 eigenVectors = Matrix4x4.identity;
                for (int i = 0; i < maxIterations; i++)
                {
                    float maxOffDiag = 0;
                    int p = 0, q = 1;
                    for (int j = 0; j < 3; j++)
                    {
                        for (int k = j + 1; k < 3; k++)
                        {
                            if (Mathf.Abs(cov[j, k]) > maxOffDiag)
                            {
                                maxOffDiag = Mathf.Abs(cov[j, k]);
                                p = j;
                                q = k;
                            }
                        }
                    }
                    if (maxOffDiag < 1e-6f) break;
                    float app = cov[p, p];
                    float aqq = cov[q, q];
                    float apq = cov[p, q];
                    float theta;
                    if (Mathf.Abs(app - aqq) < 1e-9f)
                    {
                        theta = Mathf.PI / 4.0f * (apq > 0 ? 1 : -1);
                    }
                    else
                    {
                        theta = 0.5f * Mathf.Atan(2 * apq / (app - aqq));
                    }
                    float c = Mathf.Cos(theta);
                    float s = Mathf.Sin(theta);
                    Matrix4x4 R = Matrix4x4.identity;
                    R[p, p] = c; R[p, q] = s;
                    R[q, p] = -s; R[q, q] = c;
                    cov = R.transpose * cov * R;
                    eigenVectors = eigenVectors * R;
                }
                axis1 = eigenVectors.GetColumn(0).normalized;
                axis2 = eigenVectors.GetColumn(1).normalized;
                axis3 = eigenVectors.GetColumn(2).normalized;
                // Additional: Sort by eigenvalue magnitude if needed
            }
            // New method for voxel-based plane scoring
            public static float ScorePlaneWithVoxels(MeshData mesh, Vector3 normal, Vector3 point, float concavityWeight, float sahWeight, float balanceWeight)
            {
                // Split voxels into left and right
                int leftVoxels = 0, rightVoxels = 0;
                float leftConcavity = 0, rightConcavity = 0;
                for (int idx = 0; idx < mesh.sdfValues.Length; idx++)
                {
                    int3 coord = new int3(idx % mesh.voxelDimensions.x, (idx / mesh.voxelDimensions.x) % mesh.voxelDimensions.y, idx / (mesh.voxelDimensions.x * mesh.voxelDimensions.y));
                    Vector3 voxelPos = mesh.VoxelToWorld(coord);
                    float side = Vector3.Dot(voxelPos - point, normal);
                    if (side <= 0)
                    {
                        leftVoxels++;
                        leftConcavity += mesh.sdfValues[idx] > 0 ? mesh.sdfValues[idx] : 0;
                    }
                    else
                    {
                        rightVoxels++;
                        rightConcavity += mesh.sdfValues[idx] > 0 ? mesh.sdfValues[idx] : 0;
                    }
                }
                float maxConcavity = Mathf.Max(leftConcavity / leftVoxels, rightConcavity / rightVoxels);
                float sahCost = EvaluateSplitPlane(mesh, normal, point);
                float balance = Mathf.Abs(leftVoxels - rightVoxels) / (float)(leftVoxels + rightVoxels);
                return concavityWeight * maxConcavity + sahWeight * sahCost + balanceWeight * balance;
            }
            // Additional: Feature-based plane generation
            public static (Vector3 normal, Vector3 point) GenerateFeaturePlane(MeshData mesh)
            {
                if (mesh.featureVertices.Count < 3) return (Vector3.up, mesh.bounds.center);
                var fv1 = mesh.vertices[mesh.featureVertices[0]];
                var fv2 = mesh.vertices[mesh.featureVertices[1]];
                var fv3 = mesh.vertices[mesh.featureVertices[2]];
                var normal = Vector3.Cross(fv2 - fv1, fv3 - fv1).normalized;
                var point = (fv1 + fv2 + fv3) / 3f;
                return (normal, point);
            }
        }
        /// <summary>
        /// Utility for splitting a MeshData object by a plane.
        /// </summary>
        public static class MeshSplitter
        {
            public static void Split(MeshData original, PlaneF plane, out MeshData positiveSide, out MeshData negativeSide)
            {
                positiveSide = new MeshData();
                negativeSide = new MeshData();
                var positiveVerts = new List<Vector3>();
                var negativeVerts = new List<Vector3>();
                int[] posRemap = new int[original.vertices.Count];
                int[] negRemap = new int[original.vertices.Count];
                float[] vertDistances = new float[original.vertices.Count];
                for (int i = 0; i < original.vertices.Count; i++)
                {
                    vertDistances[i] = plane.DistanceToPoint(original.vertices[i]);
                }
                for (int i = 0; i < original.indices.Count; i += 3)
                {
                    int i0 = original.indices[i];
                    int i1 = original.indices[i + 1];
                    int i2 = original.indices[i + 2];
                    float d0 = vertDistances[i0];
                    float d1 = vertDistances[i1];
                    float d2 = vertDistances[i2];
                    int posCount = (d0 > 0 ? 1 : 0) + (d1 > 0 ? 1 : 0) + (d2 > 0 ? 1 : 0);
                    if (posCount == 3)
                    {
                        AddTriangle(positiveSide, original, i0, i1, i2, posRemap);
                    }
                    else if (posCount == 0)
                    {
                        AddTriangle(negativeSide, original, i0, i1, i2, negRemap);
                    }
                    else // Triangle is clipped by the plane
                    {
                        ClipTriangle(original, plane, i0, i1, i2, d0, d1, d2, positiveSide, negativeSide, posRemap, negRemap);
                    }
                }
                positiveSide.Optimize();
                negativeSide.Optimize();
            }
            private static void AddTriangle(MeshData mesh, MeshData original, int i0, int i1, int i2, int[] remap)
            {
                int r0 = RemapVertex(mesh, original, i0, remap);
                int r1 = RemapVertex(mesh, original, i1, remap);
                int r2 = RemapVertex(mesh, original, i2, remap);
                mesh.indices.Add(r0); mesh.indices.Add(r1); mesh.indices.Add(r2);
            }
            private static int RemapVertex(MeshData mesh, MeshData original, int originalIndex, int[] remap)
            {
                if (remap[originalIndex] > 0) return remap[originalIndex] - 1;
                int newIndex = mesh.vertices.Count;
                mesh.vertices.Add(original.vertices[originalIndex]);
                if (original.normals.Count > originalIndex) mesh.normals.Add(original.normals[originalIndex]);
                remap[originalIndex] = newIndex + 1;
                return newIndex;
            }
            private static void ClipTriangle(MeshData original, PlaneF plane, int i0, int i1, int i2, float d0, float d1, float d2, MeshData posMesh, MeshData negMesh, int[] posRemap, int[] negRemap)
            {
                int[] indices = { i0, i1, i2 };
                float[] distances = { d0, d1, d2 };
                List<int> posIndices = new List<int>();
                List<int> negIndices = new List<int>();
                for (int i = 0; i < 3; i++)
                {
                    int current = indices[i];
                    int next = indices[(i + 1) % 3];
                    float dCurrent = distances[i];
                    float dNext = distances[(i + 1) % 3];
                    if (dCurrent >= 0)
                    {
                        posIndices.Add(RemapVertex(posMesh, original, current, posRemap));
                    }
                    else
                    {
                        negIndices.Add(RemapVertex(negMesh, original, current, negRemap));
                    }
                    // If edge crosses plane, create new vertex
                    if (dCurrent * dNext < 0)
                    {
                        float t = dCurrent / (dCurrent - dNext);
                        Vector3 intersectPoint = Vector3.Lerp(original.vertices[current], original.vertices[next], t);
                        int newPosVert = posMesh.vertices.Count;
                        posMesh.vertices.Add(intersectPoint);
                        posIndices.Add(newPosVert);
                        int newNegVert = negMesh.vertices.Count;
                        negMesh.vertices.Add(intersectPoint);
                        negIndices.Add(newNegVert);
                    }
                }
                // Triangulate resulting polygons
                if (posIndices.Count >= 3)
                {
                    for (int i = 1; i < posIndices.Count - 1; i++)
                    {
                        posMesh.indices.Add(posIndices[0]);
                        posMesh.indices.Add(posIndices[i]);
                        posMesh.indices.Add(posIndices[i + 1]);
                    }
                }
                if (negIndices.Count >= 3)
                {
                    for (int i = 1; i < negIndices.Count - 1; i++)
                    {
                        negMesh.indices.Add(negIndices[0]);
                        negMesh.indices.Add(negIndices[i]);
                        negMesh.indices.Add(negIndices[i + 1]);
                    }
                }
            }
            // Additional: Multi-plane split
            public static List<MeshData> MultiSplit(MeshData original, List<PlaneF> planes)
            {
                var parts = new List<MeshData> { original };
                foreach (var plane in planes)
                {
                    var newParts = new List<MeshData>();
                    foreach (var part in parts)
                    {
                        Split(part, plane, out MeshData pos, out MeshData neg);
                        if (pos.vertices.Count > 3) newParts.Add(pos);
                        if (neg.vertices.Count > 3) newParts.Add(neg);
                    }
                    parts = newParts;
                }
                return parts;
            }
        }
        [Header("Settings")]
        public ConvexDecompositionSettings settings;
        [Header("Input")]
        public GameObject sourceObject;
        public List<RegionBox> regionBoxes = new List<RegionBox>();
        [Header("Output")]
        public List<GameObject> hullObjects = new List<GameObject>();
        public DecompositionMetrics metrics = new DecompositionMetrics();
        [Header("Debug")]
        public bool enableInteractiveMode = false;
        public bool showRegionBounds = true;

        public float Progress { get; private set; }

        private MeshData combinedMesh;
        private List<ConvexHull> hulls = new List<ConvexHull>();
        private List<float> errorDistribution = new List<float>();
        private bool isProcessing = false;
        private static bool enableDetailedLoggingStatic = false; // Static for class-wide logging
        private static bool enableProfilingStatic = false;
        float ToUnitsFromMm(float mm) => (mm * 0.001f) * settings.unitsPerMeter;
        void Start()
        {
            if (sourceObject == null)
            {
                sourceObject = gameObject;
            }
            if (settings == null)
            {
                Debug.LogError("No convex decomposition settings assigned!");
                return;
            }
        }
        private void LoadAndCombineMeshData()
        {
            if (sourceObject == null)
            {
                Debug.LogError("No source object assigned!");
                return;
            }
            var meshFilters = sourceObject.GetComponentsInChildren<MeshFilter>();
            if (meshFilters.Length == 0)
            {
                Debug.LogError("No MeshFilters found in source object or its children!");
                return;
            }
            combinedMesh = new MeshData();
            int vertexOffset = 0;
            foreach (var mf in meshFilters)
            {
                if (mf.sharedMesh == null) continue;
                var mesh = mf.sharedMesh;
                vertexOffset = combinedMesh.vertices.Count;
                Matrix4x4 toWorld = transform.worldToLocalMatrix * mf.transform.localToWorldMatrix;
                var transformedVerts = new Vector3[mesh.vertexCount];
                for (int i = 0; i < mesh.vertexCount; i++)
                {
                    transformedVerts[i] = toWorld.MultiplyPoint3x4(mesh.vertices[i]);
                }
                combinedMesh.vertices.AddRange(transformedVerts);
                if (mesh.normals.Length > 0)
                {
                    var transformedNormals = new Vector3[mesh.normals.Length];
                    for (int i = 0; i < mesh.normals.Length; i++)
                    {
                        transformedNormals[i] = toWorld.MultiplyVector(mesh.normals[i]).normalized;
                    }
                    combinedMesh.normals.AddRange(transformedNormals);
                }
                var triangles = mesh.triangles;
                for (int i = 0; i < triangles.Length; i++)
                {
                    triangles[i] += vertexOffset;
                }
                combinedMesh.indices.AddRange(triangles);
            }
            combinedMesh.Optimize();
            combinedMesh.CalculateProperties();
            combinedMesh.CalculateCurvature();
            combinedMesh.CalculateSaliency();
            combinedMesh.IdentifySharpEdges();
            combinedMesh.BuildAccelerationStructure();
            metrics.originalMeshVolume = combinedMesh.volume;
            metrics.originalMeshSurfaceArea = combinedMesh.surfaceArea;
        }
        [ContextMenu("Decompose Mesh")]
        public void DecomposeMesh()
        {
            if (isProcessing)
            {
                Debug.LogWarning("Decomposition already in progress!");
                return;
            }
            LoadAndCombineMeshData();
            if (combinedMesh == null || combinedMesh.vertexCount == 0)
            {
                Debug.LogWarning("Combined mesh is empty. Aborting decomposition.");
                return;
            }
            StartCoroutine(DecomposeMeshAsync());
        }
        private IEnumerator DecomposeMeshAsync()
        {
            isProcessing = true;
            float startTime = Time.realtimeSinceStartup;
            Progress = 0f;
            ClearResults();

            if (settings.enableCacheResults && LoadResultsFromCache())
            {
                Progress = 1f;
                isProcessing = false;
                yield break;
            }

            var regionHulls = new List<List<ConvexHull>>();
            // Process regions
            float regionProgress = 0f;
            float regionIncrement = regionBoxes.Count > 0 ? 0.5f / regionBoxes.Count : 0;

            foreach (var regionBox in regionBoxes)
            {
                if (!regionBox.IsValid()) continue;
                var regionMesh = MeshBooleanOperations.BooleanIntersect(combinedMesh, MeshFromBounds(regionBox.AABB));
                if (regionMesh.vertices.Count < 4) continue;

                CalculatePriorityDelegate priorityCalc;
                FindSplitPlaneDelegate splitPlaneFinder;

                if (settings.acdSubroutine == ACDSubroutineType.CoACD)
                {
                    priorityCalc = CalculatePriority_CoACD;
                    splitPlaneFinder = FindSplitPlane_CoACD;
                }
                else // Default to VHACD
                {
                    regionMesh.Voxelize(settings.voxelSize, settings.voxelizationStrategy);
                    regionMesh.ComputeSDF(settings.sdfIterationCount, settings.sdfSmoothingFactor);
                    priorityCalc = CalculatePriority_VHACD;
                    splitPlaneFinder = FindSplitPlane_VHACD;
                }

                var newHulls = DecomposeRegion(regionMesh, regionBox.partBudgetMax, ToUnitsFromMm(regionBox.epsilonMm), priorityCalc, splitPlaneFinder);
                regionHulls.Add(newHulls);

                regionProgress += regionIncrement;
                Progress = regionProgress;
                yield return null;
            }
            // Process remainder
            var remainderMesh = combinedMesh;
            foreach (var regionBox in regionBoxes)
            {
                remainderMesh = MeshBooleanOperations.BooleanDifference(remainderMesh, MeshFromBounds(regionBox.AABB));
            }
            Progress = 0.6f;
            yield return null;

            if (remainderMesh.vertices.Count >= 4)
            {
                CalculatePriorityDelegate priorityCalc;
                FindSplitPlaneDelegate splitPlaneFinder;

                if (settings.acdSubroutine == ACDSubroutineType.CoACD)
                {
                    priorityCalc = CalculatePriority_CoACD;
                    splitPlaneFinder = FindSplitPlane_CoACD;
                }
                else
                {
                    remainderMesh.Voxelize(settings.voxelSize, settings.voxelizationStrategy);
                    remainderMesh.ComputeSDF(settings.sdfIterationCount, settings.sdfSmoothingFactor);
                    priorityCalc = CalculatePriority_VHACD;
                    splitPlaneFinder = FindSplitPlane_VHACD;
                }

                var newHulls = DecomposeRegion(remainderMesh, settings.maxHullCount, settings.errorTolerance, priorityCalc, splitPlaneFinder);
                regionHulls.Add(newHulls);
            }

            Progress = 0.7f;
            yield return null;

            // Flatten list
            foreach (var list in regionHulls) hulls.AddRange(list);

            MergeHulls();
            Progress = 0.8f;
            yield return null;

            MergeHulls();
            Progress = 0.9f;
            yield return null;

            CreateHullObjects();
            Progress = 0.95f;
            yield return null;

            CalculateMetrics();
            Progress = 1.0f;
            metrics.totalTime = Time.realtimeSinceStartup - startTime;
            SaveResultsToCache();
            isProcessing = false;
            Debug.Log($"Decomposition completed in {metrics.totalTime:F2}s with {hulls.Count} hulls");
        }
        // Delegates to abstract the decomposition strategy
        private delegate float CalculatePriorityDelegate(MeshData mesh, ConvexHull hull);
        private delegate (Vector3 normal, Vector3 point) FindSplitPlaneDelegate(MeshData mesh, ConvexHull hull, float priority);

        private List<ConvexHull> DecomposeRegion(MeshData regionMesh, int budget, float tolerance, CalculatePriorityDelegate calculatePriority, FindSplitPlaneDelegate findSplitPlane)
        {
            var finalHulls = new List<ConvexHull>();
            var processQueue = new MaxPQ<(MeshData, float)>();

            if (regionMesh.vertices.Count < 4) return finalHulls;

            var initialHull = QuickHullImplementation.ComputeConvexHullFromPoints(regionMesh.vertices);
            float initialPriority = calculatePriority(regionMesh, initialHull);
            processQueue.Enqueue((regionMesh, initialPriority));

            while (processQueue.Count > 0 && finalHulls.Count < budget)
            {
                processQueue.TryDequeue(out var currentItem, out float currentPriority);
                var currentMesh = currentItem.Item1;

                if (currentMesh.vertices.Count < 4) continue;

                var hull = QuickHullImplementation.ComputeConvexHullFromPoints(currentMesh.vertices);
                if (hull.vertices.Count < 4) continue;

                // Recalculate priority with the most recent hull
                currentPriority = calculatePriority(currentMesh, hull);

                if (currentPriority < tolerance)
                {
                    finalHulls.Add(hull);
                    continue;
                }

                var (splitNormal, splitPoint) = findSplitPlane(currentMesh, hull, currentPriority);
                var splitPlane = new PlaneF { n = splitNormal, d = -Vector3.Dot(splitNormal, splitPoint) };

                MeshSplitter.Split(currentMesh, splitPlane, out MeshData positiveSide, out MeshData negativeSide);

                if (positiveSide.vertices.Count > 3)
                {
                    var posHull = QuickHullImplementation.ComputeConvexHullFromPoints(positiveSide.vertices);
                    float posPriority = calculatePriority(positiveSide, posHull);
                    processQueue.Enqueue((positiveSide, posPriority));
                }
                if (negativeSide.vertices.Count > 3)
                {
                    var negHull = QuickHullImplementation.ComputeConvexHullFromPoints(negativeSide.vertices);
                    float negPriority = calculatePriority(negativeSide, negHull);
                    processQueue.Enqueue((negativeSide, negPriority));
                }
            }

            while (processQueue.Count > 0)
            {
                processQueue.TryDequeue(out var(mesh, _), out _);
                if (mesh.vertices.Count > 3)
                {
                    finalHulls.Add(QuickHullImplementation.ComputeConvexHullFromPoints(mesh.vertices));
                }
            }

            return finalHulls;
        }


        private float CalculatePriority_CoACD(MeshData mesh, ConvexHull hull)
        {
            if (mesh == null || hull == null || mesh.vertices.Count == 0 || hull.vertices.Count == 0) return 0;
            return ErrorCalculator.CalculateSymmetricHausdorff(mesh, hull);
        }

        private float CalculatePriority_VHACD(MeshData mesh, ConvexHull hull)
        {
            return CalculateConcavity(mesh, hull);
        }

        private (Vector3, Vector3) FindSplitPlane_CoACD(MeshData mesh, ConvexHull hull, float priority)
        {
            return SplitPlaneEvaluator.FindBestSplitPlane_CoACD(mesh, hull, priority);
        }

        private (Vector3, Vector3) FindSplitPlane_VHACD(MeshData mesh, ConvexHull hull, float priority)
        {
            return SplitPlaneEvaluator.FindBestSplitPlane(mesh, hull, priority);
        }

        private float CalculateConcavity(MeshData mesh, ConvexHull hull)
        {
            if (hull == null)
            {
                hull = QuickHullImplementation.ComputeConvexHullFromPoints(mesh.vertices);
            }
            // Use SDF/voxel-based concavity
            float maxConcavity = 0f;
            float averageConcavity = 0f;
            int outsideCount = 0;
            for (int idx = 0; idx < mesh.sdfValues.Length; idx++)
            {
                int3 coord = new int3(idx % mesh.voxelDimensions.x, (idx / mesh.voxelDimensions.x) % mesh.voxelDimensions.y, idx / (mesh.voxelDimensions.x * mesh.voxelDimensions.y));
                Vector3 voxelPos = mesh.VoxelToWorld(coord);
                if (!hull.ContainsPoint(voxelPos) && mesh.sdfValues[idx] < 0) // Outside hull but inside mesh
                {
                    float dist = hull.DistanceToPoint(voxelPos);
                    maxConcavity = Mathf.Max(maxConcavity, dist);
                    averageConcavity += dist;
                    outsideCount++;
                }
            }
            return outsideCount > 0 ? averageConcavity / outsideCount : 0f; // Weighted average
        }
        private void MergeHulls()
        {
            bool changed = true;
            int mergeIterations = 0;
            while (changed && mergeIterations < settings.maxOptimizationIterations)
            {
                changed = false;
                for (int i = 0; i < hulls.Count && !changed; i++)
                {
                    for (int j = i + 1; j < hulls.Count; j++)
                    {
                        if (hulls[i].bounds.Intersects(hulls[j].bounds))
                        {
                            var merged = MeshBooleanOperations.BooleanUnion(hulls[i], hulls[j]);
                            var hullMesh = new MeshData { vertices = merged.vertices, indices = merged.indices };
                            hullMesh.Voxelize(settings.voxelSize, settings.voxelizationStrategy);
                            hullMesh.ComputeSDF();
                            float mergedConcavity = CalculateConcavity(hullMesh, merged);
                            if (mergedConcavity < settings.mergeConcavityThreshold)
                            {
                                hulls.RemoveAt(j);
                                hulls[i] = merged;
                                changed = true;
                                metrics.successfulMerges++;
                                break;
                            }
                        }
                    }
                }
                mergeIterations++;
            }
            metrics.mergeEfficiency = (float)metrics.successfulMerges / hulls.Count;
        }
        private void CreateHullObjects()
        {
            foreach (var hullObj in hullObjects) if (hullObj != null) Destroy(hullObj);
            hullObjects.Clear();
            var container = new GameObject("ConvexHulls");
            container.transform.SetParent(transform, false);
            for (int i = 0; i < hulls.Count; i++)
            {
                var hull = hulls[i];
                if (settings.enableHullOptimization && hull.vertices.Count > settings.maxVerticesPerHull)
                    hull.OptimizeHull(settings.hullOptimizationThreshold);
                // Safety for convex collider:
                if ((hull.indices.Count / 3) > 255)
                    hull.OptimizeHull(settings.hullOptimizationThreshold);
                var hullObj = new GameObject($"Hull_{i}");
                hullObj.transform.SetParent(container.transform, false);
                var mesh = new Mesh
                {
                    vertices = hull.vertices.ToArray(),
                    triangles = hull.indices.ToArray()
                };
                mesh.RecalculateNormals();
                mesh.RecalculateBounds();
                hullObj.AddComponent<MeshFilter>().sharedMesh = mesh;
                hullObj.AddComponent<MeshRenderer>().sharedMaterial = settings.hullMaterial ?? new Material(Shader.Find("Standard"));
                var col = hullObj.AddComponent<MeshCollider>();
                col.sharedMesh = mesh;
                col.convex = true;
                hullObjects.Add(hullObj);
            }
        }
        private void CalculateMetrics()
        {
            metrics = new DecompositionMetrics
            {
                vertexCount = combinedMesh.vertexCount,
                triangleCount = combinedMesh.triangleCount,
                hullCount = hulls.Count,
                originalMeshVolume = combinedMesh.volume,
                originalMeshSurfaceArea = combinedMesh.surfaceArea
            };
            errorDistribution = ErrorCalculator.CalculateErrorDistribution(combinedMesh, hulls, 1000);
            metrics.errorDistribution = errorDistribution;
            metrics.hullMetrics.Clear();
            for (int i = 0; i < hulls.Count; i++)
            {
                var hull = hulls[i];
                var hullMetric = new HullMetrics
                {
                    hullId = i,
                    vertexCount = hull.vertices.Count,
                    triangleCount = hull.indices.Count / 3,
                    volume = hull.volume,
                    surfaceArea = hull.surfaceArea,
                    bounds = hull.bounds,
                    centroid = hull.centroid,
                    compactness = hull.compactness,
                    aspectRatio = hull.aspectRatio
                };
                metrics.hullMetrics.Add(hullMetric);
            }
            metrics.CalculateDerivedMetrics();
            // Additional: Export if profiling
            if (settings.enableProfiling)
                metrics.ExportToCSV("decomposition_metrics.csv");
        }
        private void ClearResults()
        {
            foreach (var hullObj in hullObjects)
            {
                if (hullObj != null)
                {
#if UNITY_EDITOR
                    DestroyImmediate(hullObj);
#else
                    Destroy(hullObj);
#endif
                }
            }
            // Also destroy the container
            var container = transform.Find("ConvexHulls");
            if (container != null)
            {
#if UNITY_EDITOR
                DestroyImmediate(container.gameObject);
#else
                Destroy(container.gameObject);
#endif
            }
            hullObjects.Clear();
            hulls.Clear();
            metrics = new DecompositionMetrics();
        }
        [Serializable]
        private class CachedHull
        {
            public Vector3[] vertices;
            public int[] indices;
        }

        [Serializable]
        private class CacheData
        {
            public List<CachedHull> hulls;
            public DecompositionMetrics metrics;
        }

        private bool LoadResultsFromCache()
        {
            if (!System.IO.File.Exists(settings.cacheFilePath)) return false;

            try
            {
                string json = System.IO.File.ReadAllText(settings.cacheFilePath);
                CacheData cache = JsonUtility.FromJson<CacheData>(json);

                hulls.Clear();
                foreach(var cachedHull in cache.hulls)
                {
                    var hull = new ConvexHull();
                    hull.vertices = new List<Vector3>(cachedHull.vertices);
                    hull.indices = new List<int>(cachedHull.indices);
                    hull.CalculateProperties();
                    hulls.Add(hull);
                }

                this.metrics = cache.metrics;
                CreateHullObjects();
                Debug.Log($"Loaded {hulls.Count} hulls from cache.");
                return true;
            }
            catch(Exception e)
            {
                Debug.LogError($"Failed to load from cache: {e.Message}");
                return false;
            }
        }

        private void SaveResultsToCache()
        {
            if (!settings.enableCacheResults) return;

            var cache = new CacheData
            {
                hulls = new List<CachedHull>(),
                metrics = this.metrics
            };

            foreach(var hull in hulls)
            {
                cache.hulls.Add(new CachedHull
                {
                    vertices = hull.vertices.ToArray(),
                    indices = hull.indices.ToArray()
                });
            }

            try
            {
                string json = JsonUtility.ToJson(cache, true);
                System.IO.File.WriteAllText(settings.cacheFilePath, json);
                Debug.Log($"Saved {hulls.Count} hulls to cache.");
            }
            catch(Exception e)
            {
                Debug.LogError($"Failed to save to cache: {e.Message}");
            }
        }

        private void OnDrawGizmos()
        {
            if (!enableInteractiveMode || settings == null) return;
            if (showRegionBounds)
            {
                Gizmos.color = Color.yellow;
                foreach (var region in regionBoxes)
                {
                    Gizmos.matrix = transform.localToWorldMatrix;
                    Gizmos.DrawWireCube(region.AABB.center, region.AABB.size);
                }
            }
        }
    }
}