# 3D Geometry, Reconstruction, and SLAM Cheat Sheet

## I. 3D Concepts and Representations

### What is 3D?
- Understanding the difference between 2D (images as arrays of pixels) and 3D (geometry with depth)

### Representations of 3D Data

#### Point Clouds
- Linear array of 3D coordinates (x, y, z) with optional attributes like color and normals
- Sample the surface and are linear in storage complexity
- Advantages: simple structure, direct sensor output
- Disadvantages: no connectivity information, sparse

#### Voxel Grids
- Cubic storage, storing values everywhere in 3D space, not just on surfaces
- Advantages: regular structure, easy to implement algorithms
- Disadvantages: cubic storage complexity, limited resolution

#### Polygonal Meshes
- Piecewise linear surfaces composed of vertices and faces
- They can represent the same geometry with different topologies
- Common file formats: OBJ, STL, PLY, OFF
- Advantages: efficient for rendering, widely supported
- Disadvantages: can represent invalid surfaces

#### Parametric Surfaces
- Higher-order surfaces used in CAD modeling and animation
- Offering non-linear representation with continuous normals
- Often using cubic degrees for animation
- Advantages: smooth representation, compact
- Disadvantages: cannot be directly rendered on GPU

#### Implicit Surfaces
- Defined by a function that represents distance to the surface
- Storing a floating-point value per voxel indicating the proximity to the surface
- Advantages: easy to combine multiple surfaces, robust topology
- Disadvantages: harder to render directly

#### Explicit Surfaces
- Surfaces defined by an explicit function, such as height fields in game design
- Advantages: simple to work with for specific applications
- Disadvantages: limited in representable geometries

## II. Scanning Devices and Depth Sensors

### Depth Images
- Images where each pixel stores depth information
- Typically represents distance along the z-axis, not raw ray length

### Scanning Devices

#### Stereo
- **Type**: Passive
- **Advantages**: Works indoors and outdoors
- **Disadvantages**: Computationally expensive due to feature matching

#### Structured Light
- **Type**: Active
- **Advantages**: Excels in featureless regions
- **Disadvantages**: Fails outdoors due to reliance on IR spectrum
- Requires precise calibration between projector and sensor

#### Laser Scanning
- **Type**: Active
- **Advantages**: Precise
- **Disadvantages**: Often slow for high quality
- Trivial feature matching

#### Time of Flight
- **Type**: Active
- **Advantages**: Can handle featureless regions
- **Disadvantages**: Sensitive to scattering light, fails outdoors
- Uses phase differences to measure distance

### Camera Parameters

#### Intrinsic Parameters
- Focal length, principal point, skew, and distortion
- Projects from camera space to screen space

#### Extrinsic Parameters
- 6DoF pose (rotation and translation)
- Defines the transformation to global camera space

## III. 3D Reconstruction Methods

### Multi-View Stereo
- Using multiple images to reconstruct 3D structures

### Structure from Motion (SfM)
- Reconstructing 3D structures from a set of 2D images
- Widely used in practical applications
- Involves:
  - Feature extraction (SIFT, SURF)
  - Feature matching
  - Bundle adjustment

### Bundle Adjustment
- Optimizing camera poses and 3D point positions to minimize reprojection error

### Virtual Hull Carving
- Using multi-view setups to carve out free space in a voxel grid

## IV. Optimization Techniques

### Linear vs. Non-Linear Least Squares
- Linear: When the problem can be represented as a linear combination of parameters
- Non-Linear: When the problem involves non-linear functions of parameters

### Solvers for Linear Systems

#### Iterative Solvers
- Conjugate Gradient Descent (CG)
- Preconditioned Conjugate Gradient Descent (PCG)
- Good for large, sparse systems

#### Direct Approaches
- QR Decomposition
- LU Decomposition
- Cholesky Decomposition
- SVD (Singular Value Decomposition)
- Good for smaller, dense systems

### Numerical Derivatives
- Central differences
- Forward differences

### Automatic Differentiation
- Using tools like Ceres to automatically compute derivatives
- Dual numbers can be implemented using C++ templates

### Bundle Adjustment Optimization

#### Reprojection Error
- Minimizing the distance between projected 3D points and their corresponding image locations

#### Photometric Alignment
- Aligning images based on color and intensity values

## V. Rigid and Non-Rigid Reconstruction

### Iterative Closest Point (ICP)
- Aligning two point clouds by iteratively finding correspondences and minimizing distance
- Key steps:
  - Correspondence finding using nearest neighbor search
  - Error metric computation and minimization
  - Transformation application and iteration

### Variations of ICP
- Point-to-Point ICP: Minimizes distance between corresponding points
- Point-to-Plane ICP: Minimizes distance from point to tangent plane
- Symmetric ICP: Uses an average normal of both points for improved alignment

### Signed Distance Field (SDF) Integration
- Integrating depth data into a volumetric grid
- The sign indicates in front of (positive) or behind (negative) the surface

### Non-Rigid ICP
- Extending ICP to handle deformations
- Widely used in character animation

### As-Rigid-As-Possible (ARAP)
- A method for mesh deformation that preserves local rigidity
- Uses cotangent weights in discrete settings
- Solved with alternating optimization ("flip-flop"):
  - Fix vertices, solve rotations
  - Fix rotations, solve vertices

### Non-Rigid Tracking
- Aligning a template mesh to a sequence of depth images in a non-rigid fashion

## VI. Advanced Reconstruction Techniques

### Marching Cubes
- Algorithm to extract a polygonal mesh from a volume
- Identifies triangle intersections within a voxel based on sign changes
- Can be parallelized by independently processing each voxel

### Ray Marching
- A rendering technique used with distance fields
- Steps along a ray and checks for sign changes to detect surface crossings
- Step size can be dynamically adapted using sphere tracing

### Poisson Surface Reconstruction
- Converts point clouds into surfaces by solving a Poisson equation
- Requires a point cloud with normals as input
- The output is an implicit indicator function from which a mesh can be extracted

### Volumetric Fusion
- Combines frame-to-model and frame-to-frame camera tracking
- Uses SIFT features for global pose estimation
- Can perform loop closures and relocalization

### Elastic Fusion
- Uses point-based representations based on surfels (surface elements)
- Updates surfaces and corrects drift

## VII. Feature Extraction and Matching

### Feature Descriptors
- Methods to describe local image features (SIFT, SURF, ORB, FREAK)
- Good feature descriptors should be scaling, view, and lighting invariant

### Feature Detection
- FAST (Features from Accelerated Segment Test): Looks for points where a circle of pixels has contiguous pixels brighter or darker than center
- Harris Corner Detector: Based on local gradient changes, analyzing eigenvalues

### Feature Matching
- Finding corresponding features in different images

### KD-Trees
- Data structure for efficient nearest neighbor search
- Used for feature matching and correspondence finding in ICP

## VIII. SLAM (Simultaneous Localization and Mapping)

### SLAM Overview
- Building a map of an environment while simultaneously localizing within it

### Keyframe Extraction
- Selecting keyframes to reduce computational cost

### Loop Closure
- Identifying previously visited locations to correct drift

### RGB-D SLAM
- SLAM using RGB-D cameras, providing both color and depth information

### PoseNet
- CNN for camera pose regression from a single RGB image
- Estimates 6DOF camera pose

## IX. Advanced Optimization Methods

### Gradient Descent
- Iterative optimization moving in steepest descent direction
- Requires determining appropriate step size

### Newton's Method
- Second-order optimization using the Hessian matrix
- Faster convergence but computationally expensive

### Gauss-Newton Method
- Approximates Hessian using the Jacobian
- For non-linear least squares problems

### Levenberg-Marquardt Algorithm
- Combines gradient descent and Gauss-Newton
- Uses damping factor to interpolate between methods
- Typically the preferred solver ("LM" in practice)

### BFGS and L-BFGS
- Quasi-Newton methods approximating Hessian
- L-BFGS (Limited-memory BFGS) is memory-efficient for large problems
- "L-BFGS is the way to go actually in practice"

### Iteratively Reweighted Least Squares (IRLS)
- Method for solving optimization problems with robust norms
- Iteratively solves weighted least squares problems

## X. Non-Rigid Deformation and Tracking

### Deformation Proxies
- Simplified representations used to control mesh deformations
- Examples include cages and harmonic coordinates

### Template Mesh Tracking
- Aligning a template mesh to a sequence of depth images

### Surface Registration
- Aligning two surfaces non-rigidly

## XI. Sensor Fusion

### IMU Integration
- Combining IMU data with visual data to improve tracking accuracy

### Visual-Inertial Odometry (VIO)
- Estimating pose by fusing IMU and camera data

### Kalman Filters and Extended Kalman Filters (EKF)
- Algorithms for fusing data from multiple sensors

---

## Interview Questions and Answers

### 3D Concepts and Representations

**Q: Explain the differences between point clouds and voxel grids in terms of storage and information.**  
**A:** Voxel grids require cubic storage, storing values throughout 3D space, whereas point clouds linearly store sampled surface points.

**Q: How would you represent a complex, curved surface? What are the advantages and disadvantages of your choice?**  
**A:** Parametric surfaces can represent complex, curved surfaces with higher-order continuity. However, they cannot be directly rendered on a GPU and must be converted to a triangle mesh.

**Q: Describe a scenario where an implicit surface representation would be preferred over a polygonal mesh.**  
**A:** Implicit surfaces are useful when you need to know how close you are to a surface, as they store a floating-point value per voxel indicating the distance to the surface.

**Q: Discuss how different triangulations of a surface would affect the solution you find during reconstruction.**  
**A:** Different triangulations mean multiple solutions, requiring careful handling in the reconstruction pipeline.

### Scanning Devices and Depth Sensors

**Q: Compare and contrast stereo and structured light techniques for depth sensing.**  
**A:** Stereo is passive and works in various environments but is computationally intensive. Structured light is active, good for featureless regions, but limited outdoors.

**Q: What are intrinsic and extrinsic camera parameters, and how are they used in 3D reconstruction?**  
**A:** Intrinsic parameters define the camera's internal characteristics, while extrinsic parameters define its position and orientation in the world.

**Q: How do you correct lens distortion?**  
**A:** Polynomial approximations are used, optimized with camera calibration toolchains using checkerboard patterns.

### 3D Reconstruction Methods

**Q: Explain the structure from motion pipeline.**  
**A:** It starts with unstructured images, extracts features, matches features across images, performs bundle adjustment, and generates a sparse or dense reconstruction.

**Q: What is bundle adjustment, and why is it important in 3D reconstruction?**  
**A:** Bundle adjustment refines camera poses and 3D point positions by minimizing the reprojection error, leading to more accurate reconstructions.

**Q: What are the limitations of virtual hull carving?**  
**A:** It requires expensive, calibrated multi-view setups, is sensitive to noise, and may not produce high-quality results.

### Optimization Techniques

**Q: When should you use iterative solvers like PCG over direct methods like Cholesky decomposition?**  
**A:** Iterative solvers are preferred for large, sparse systems, while direct methods are suitable for smaller, dense systems.

**Q: How does automatic differentiation simplify the optimization process?**  
**A:** It automates the computation of derivatives, reducing the chance of human error and speeding up development.

**Q: Describe the Gauss-Newton method and its application in bundle adjustment.**  
**A:** The Gauss-Newton method linearizes the non-linear least squares problem and iteratively refines the solution, widely used in bundle adjustment for camera pose and structure estimation.

**Q: Discuss a scenario when different data terms might be more difficult to optimize than others.**  
**A:** Photometric alignment, due to its complex and non-convex energy landscape with many local minima, is often more challenging to optimize for compared to reprojection errors or depth terms.

### Rigid and Non-Rigid Reconstruction

**Q: Explain the ICP algorithm and its variations (point-to-point, point-to-plane).**  
**A:** ICP iteratively finds closest points between two datasets, computes a transformation, and applies it, with variations differing in error metrics and constraints.

**Q: How does non-rigid ICP differ from standard ICP?**  
**A:** Non-rigid ICP accounts for deformations by solving for deformation parameters in addition to rigid transformations.

**Q: What is ARAP, and how does it preserve local rigidity during mesh deformation?**  
**A:** ARAP minimizes the difference between local rotations of mesh elements, preserving the original shape as much as possible while allowing deformation.

**Q: Describe the key challenges in non-rigid 3D reconstruction compared to rigid reconstruction.**  
**A:** Key challenges include resolving ambiguities between surface motion and deformation and needing effective priors to guide the reconstruction.

### Advanced Reconstruction Techniques

**Q: Explain the marching cubes algorithm and its applications.**  
**A:** Marching Cubes is used to create a triangle mesh from a volume by interpolating the intersections based on whether cube vertices are inside or outside the surface.

**Q: How does ray marching work, and what are its advantages?**  
**A:** Ray marching steps along a ray to detect surface crossings by checking for sign changes. The step size can be dynamically adapted using sphere tracing to accelerate the process.

**Q: What is Poisson surface reconstruction, and when is it useful?**  
**A:** Poisson surface reconstruction converts point clouds into surfaces by solving a Poisson equation, requiring a point cloud with normals as input.

**Q: Describe the key components of Bundle Fusion and how it achieves real-time performance.**  
**A:** Bundle Fusion combines frame-to-model and frame-to-frame camera tracking with global RGB bundling, using SIFT features for global pose estimation.

### Feature Extraction and Matching

**Q: What makes a good feature descriptor?**  
**A:** A good feature descriptor should be scaling, view, and lighting invariant.

**Q: Explain how KD-trees are used in feature matching.**  
**A:** KD-trees are used to efficiently find the nearest neighbors in feature space, which speeds up the matching process.

### SLAM

**Q: Explain the concept of loop closure in SLAM.**  
**A:** Loop closure involves identifying previously visited locations to correct accumulated drift in the map.

**Q: How does RGB-D SLAM differ from traditional SLAM?**  
**A:** RGB-D SLAM uses RGB-D cameras, providing both color and depth information, which can improve the accuracy and robustness of the SLAM system.

### Optimization Methods

**Q: Explain the difference between gradient descent and Newton's method.**  
**A:** Gradient descent uses the first derivative (gradient) to update parameters, while Newton's method uses the second derivative (Hessian) for faster convergence.

**Q: What is the Levenberg-Marquardt algorithm, and why is it useful?**  
**A:** The Levenberg-Marquardt algorithm combines gradient descent and the Gauss-Newton method, using a damping factor to interpolate between the two methods.

**Q: Describe the BFGS and L-BFGS algorithms.**  
**A:** BFGS is a quasi-Newton method that approximates the Hessian matrix, while L-BFGS is a memory-efficient version for large problems.

**Q: What is iteratively reweighted least squares (IRLS), and when is it used?**  
**A:** IRLS is a method for solving optimization problems with robust norms, iteratively solving a weighted least squares problem.

### Non-Rigid Deformation and Tracking

**Q: What are deformation proxies, and how are they used in non-rigid deformation?**  
**A:** Deformation proxies are simplified representations used to control mesh deformations, such as cages and harmonic coordinates.

**Q: Explain how ARAP regularization preserves local rigidity during deformation.**  
**A:** ARAP regularization minimizes the difference between local rotations of mesh elements, preserving the original shape as much as possible while allowing deformation.

**Q: How does non-rigid ICP work?**  
**A:** Non-rigid ICP extends the standard ICP algorithm to account for deformations by solving for deformation parameters in addition to rigid transformations.

**Q: What are the challenges of tracking deformable objects?**  
**A:** Challenges include resolving ambiguities between surface motion and deformation, and the need for effective priors to guide the reconstruction.

### Sensor Fusion

**Q: How can IMU data be used to improve tracking accuracy in visual SLAM?**  
**A:** IMU data provides high-frequency measurements of acceleration and angular velocity, which can be fused with visual data to improve tracking accuracy.

**Q: Explain the concept of visual-inertial odometry (VIO).**  
**A:** Visual-inertial odometry (VIO) estimates the pose of a device by fusing IMU and camera data.

### Implementation Questions

**Q: Implement a basic ICP algorithm.**  
**A:** This involves writing code for nearest neighbor search, transformation estimation, and iterative refinement.

**Q: Design a system for real-time 3D reconstruction using Kinect.**  
**A:** Select appropriate algorithms for tracking, reconstruction, and rendering, and optimize performance for real-time operation.

**Q: Write code to estimate normals from a point cloud using PCA.**  
**A:** Implement PCA on the neighborhood of each point and extract the normal from the eigenvector corresponding to the smallest eigenvalue.

**Q: How would you parallelize conjugate gradient updates on GPU?**  
**A:** Think through mapping the algorithm to GPU architecture, considering memory access patterns and thread synchronization. 