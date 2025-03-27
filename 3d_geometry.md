# 3D Geometry & Visualization Cheatsheet for Robotics

## 1. Coordinate Systems & Transformations

### Coordinate Systems
```
    Z                 Z
    |                 |
    |                 |    
    |____X       Y____|
   /               /
  /               /
 Y               X
ROS Standard      Camera Standard
```

- **Right-hand rule**: Curl fingers from x to y, thumb points to z
- **Common conventions**:
  - **ROS standard**: x forward, y left, z up
  - **Camera standard**: z forward, x right, y down
  - **OpenGL standard**: x right, y up, z backward

### Transformation Representation

#### Homogeneous Transformation Matrix
```
┌                  ┐
│ r11  r12  r13  tx │
│ r21  r22  r23  ty │
│ r31  r32  r33  tz │
│  0    0    0   1  │
└                  ┘
  Rotation    Translation
```

#### Euler Angles Visualization
```
      Z                 Z'                  Z''
      |                 |                   |
      |                 |                   |
Roll  |     Pitch       |      Yaw          |
→     |     →           |      →            |
      O----> X         O'----> X'          O''----> X''
     /                 /                   /
    /                 /                   /
   Y                 Y'                  Y''
```

#### Quaternion Parameters
```
q = w + xi + yj + zk

Unit quaternion: w² + x² + y² + z² = 1
```

## 2. Point Cloud Visualization & Processing

### Point Cloud Visualization Types
```
Raw Points           Colored by Height      Surface Normals
   .  .                  .  .                  .  .
 .      .  ↑           .      .  ↑           .  ↑   .  ↑
.    .    .  |        .    .    .  |        . /    . /
 .     .     | Z       .     .     | Z       /  .    /
   .  .      |           .  .      |        ↑     ↑
```

### Point Cloud Organization
```
KD-Tree                  Voxel Grid               Octree
    │                   ┌─┬─┬─┬─┐              ┌───┬───┐
    ▼                   ├─┼─┼─┼─┤              │   │   │
   / \                  ├─┼─┼─┼─┤              ├───┼───┤
  /   \                 ├─┼─┼─┼─┤              │   │   │
 /     \                └─┴─┴─┴─┘              └───┴───┘
```

### Common Visualization Color Schemes
| Type | Use Case | Example |
|------|----------|---------|
| **Height map** | Terrain visualization | Blue (low) → Green → Yellow → Red (high) |
| **Intensity** | LiDAR reflectivity | Grayscale (0-255) |
| **RGB** | Camera-colored points | Natural color |
| **Segmentation** | Object classification | Unique color per class |
| **Normal** | Surface orientation | RGB = XYZ components of normal |

### Segmentation Visualization
```
Raw Point Cloud       Segmented Objects        Instance Masks
   .  .  .  .            .  .  .  .             .  .  .  .
 .  .  .  .  .          .  .  .  .  .           .  .  .  .  .
.  .  .  .  .  .     →  1  1  2  2  3  3    →   🟥 🟥 🟦 🟦 🟨 🟨
 .  .  .  .  .          .  .  .  .  .           .  .  .  .  .
   .  .  .  .            .  .  .  .             .  .  .  .
```

## 3. SLAM Visualization Techniques

### SLAM Components Visualization
```
              ┌───────────────┐
              │ Input Sensors │
              └───────┬───────┘
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
┌───────────────┐           ┌───────────────┐
│Feature Extract│◄──────────►│Pose Estimation│
└───────┬───────┘           └───────┬───────┘
        │                           │
        ▼                           ▼
┌───────────────┐           ┌───────────────┐
│   Mapping     │◄──────────►│ Loop Closure  │
└───────────────┘           └───────────────┘
```

### Feature Visualization
```
Raw Image                 Features Detected          Feature Matching
┌──────────────┐         ┌──────────────┐         ┌─────────┬─────────┐
│              │         │   •    •     │         │ •    •  │  •    • │
│              │    →    │       •   •  │    →    │    •    │     •   │
│              │         │  •  •        │         │  •   •  │   •   • │
└──────────────┘         └──────────────┘         └─────────┴─────────┘
                            FAST corners           Correspondences (lines)
```

### Trajectory & Map Visualization
```
                   Estimated Path
                   .
                  / \
                 /   \
                /     \
Ground Truth   /       \    Loop
─────────────>•         \   Closure
               \         \    |
                \         \   ↓
                 \         \ /
                  \_________•
```

### Factor Graph Visualization
```
    ┌───┐     ┌───┐     ┌───┐     ┌───┐
    │ X₁│─────│ X₂│─────│ X₃│─────│ X₄│  ← Robot poses
    └─┬─┘     └─┬─┘     └─┬─┘     └─┬─┘
      │         │         │         │
      │         │         │         │     ← Observations
      ▼         ▼         ▼         ▼
    ┌───┐     ┌───┐     ┌───┐     ┌───┐
    │ L₁│     │ L₂│     │ L₃│     │ L₄│  ← Landmarks
    └───┘     └───┘     └───┘     └───┘
      ▲                             ▲
      │                             │
      └─────────────────────────────┘     ← Loop closure
```

## 4. Sensor Fusion Visualization

### Multi-Sensor Fusion Diagram
```
┌──────────┐    ┌──────────┐    ┌──────────┐
│  Camera  │    │   LiDAR  │    │   IMU    │
└────┬─────┘    └────┬─────┘    └────┬─────┘
     │               │               │
     ▼               ▼               ▼
┌────┴─────┐    ┌────┴─────┐    ┌────┴─────┐
│  Feature │    │   Point  │    │  Motion  │
│Extraction│    │  Cloud   │    │Estimation│
└────┬─────┘    └────┬─────┘    └────┬─────┘
     │               │               │
     └───────────────┼───────────────┘
                     ▼
              ┌──────────────┐
              │ Fusion Filter│
              │   (EKF/UKF)  │
              └──────┬───────┘
                     ▼
              ┌──────────────┐
              │  Fused State │
              │  Estimation  │
              └──────────────┘
```

### Kalman Filter Visualization
```
          Prior                 Measurement              Posterior
       Distribution             Distribution             Distribution
          /│\                      /│\                      /│\
         / │ \                    / │ \                    / │ \
        /  │  \                  /  │  \                  /  │  \
       /   │   \                /   │   \                /   │   \
      /    │    \              /    │    \              /    │    \
     /     │     \            /     │     \            /     │     \
    /      │      \          /      │      \          /      │      \
───┴───────┴───────┴────  ───┴───────┴───────┴────  ───┴───────┴───────┴────
          x₁                       x₂                   x₃ (Combined)
```

### Particle Filter Visualization
```
1. Initial Particles     2. Motion Update      3. Measurement Update
    • • • • • • •           ↗ ↗ ↗               • • • •
    • • • • • • •          ↗ ↗ ↗ ↗            • • • • • •
    • • • ⊙ • • •    →    ↗ ↗ ⊙ ↗ ↗      →   • • • ⊙ • •
    • • • • • • •          ↗ ↗ ↗ ↗               • • •
    • • • • • • •           ↗ ↗ ↗                  •

⊙ = True position        Arrows = Motion      Size = Weight
```

## 5. Camera-to-World Transformation Pipeline

### Camera Model Visualization
```
       3D World               Image Plane
          P(X,Y,Z)               p(u,v)
             •                     •
             │                     │
             │                     │
             │      Optical       │
             │       Axis         │
             │         │          │
             ▼         ▼          ▼
          Z  │         │          │
           ╲ │         │          │
            ╲│         │          │
      Camera ⊙─────────┼──────────┼─────► X
      Center           │          │
                  focal length    │
                                  │
                                  ▼
                                  Y
```

### Projection Matrix Visualization
```
┌   ┐   ┌                 ┐ ┌   ┐
│ u │   │ fx   0   cx   0 │ │ X │
│ v │ = │ 0   fy   cy   0 │ │ Y │
│ 1 │   │ 0    0    1   0 │ │ Z │
└   ┘   └                 ┘ │ 1 │
                            └   ┘

Intrinsics K            3D Point
```

### Transformation Chain Visualization
```
   Object           Camera           Robot           World
Coordinate        Coordinate       Coordinate      Coordinate
   System           System           System          System
     ↓                ↓                ↓               ↓
  [X_obj]    →     [X_cam]    →     [X_rob]    →    [X_world]
     ↑                ↑                ↑               ↑
     └────────┐      └────────┐       └───────┐       │
              │               │                │       │
         T_obj_cam        T_cam_rob       T_rob_world

```

## 6. 3D Perception Debugging Visualization

### Pipeline Component Visualization
```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Raw Sensor  │→ │  Preprocess  │→ │   Feature    │→ │   Object     │
│    Data      │  │  & Filter    │  │  Extraction  │  │  Detection   │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │                 │
       ▼                 ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Raw PCL    │  │  Filtered    │  │  Extracted   │  │  Bounding    │
│  Visualizer  │  │    PCL       │  │  Features    │  │   Boxes      │
└──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘
```

### Error Visualization Methods
```
Ground Truth         Estimated           Error Heatmap
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│              │   │              │   │▓▓▒▒▒▒        │
│     ┌───┐    │   │    ┌───┐     │   │▓▓▒▒▒▒        │
│     │   │    │   │    │   │     │   │▓▓▒▒          │
│     └───┘    │ - │    └───┘     │ = │▒▒            │
│              │   │              │   │              │
└──────────────┘   └──────────────┘   └──────────────┘
```

### Common Debug Visualizations

| Visualization Type | Purpose | Display Method |
|-------------------|---------|---------------|
| **Confidence heatmap** | Show detection certainty | Color gradient overlay |
| **Wireframe overlay** | Show 3D model alignment | Edges on image |
| **Confusion matrix** | Evaluate classification | Color-coded grid |
| **Occlusion visualization** | Show hidden areas | Transparency/hatching |
| **Trajectory comparison** | Evaluate localization | Multiple path lines |

## 7. System Integration & Visualization Architecture

### Data Flow Visualization
```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Sensor  │───►│ Perception│───►│ Planning │───►│ Control  │
│  Fusion  │    │  Stack   │    │  Stack   │    │  Stack   │
└────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘
     │               │               │               │
     ▼               ▼               ▼               ▼
┌────┴─────┐    ┌────┴─────┐    ┌────┴─────┐    ┌────┴─────┐
│  Sensor  │    │   3D     │    │   Path   │    │ Actuator │
│ Visual.  │    │  Scene   │    │ Visual.  │    │ Visual.  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
```

### Visualization Framework Architecture
```
┌─────────────────────────────────────────────┐
│              Application Layer               │
└───────────────────┬─────────────────────────┘
                    │
┌───────────────────▼─────────────────────────┐
│              Visualization Core              │
├─────────────┬─────────────┬─────────────────┤
│  Renderers  │  Layouts    │ Interaction     │
├─────────────┼─────────────┼─────────────────┤
│  2D         │  Timeline   │ Selection       │
│  3D         │  Split View │ Query           │
│  Plots      │  Multiview  │ Annotation      │
└─────────────┴─────────────┴─────────────────┘
                    │
┌───────────────────▼─────────────────────────┐
│              Data Adapters                   │
├─────────────┬─────────────┬─────────────────┤
│  Points     │  Images     │ Trajectories    │
│  Meshes     │  Video      │ Transforms      │
│  Volumes    │  Text       │ Semantic Data   │
└─────────────┴─────────────┴─────────────────┘
```

### Multi-view Visualization Layout
```
┌───────────────┬───────────────┐
│               │               │
│  3D View      │  Raw Camera   │
│               │  Feed         │
│               │               │
├───────────────┼───────────────┤
│               │               │
│  LiDAR Top    │  System       │
│  View         │  State        │
│               │  Dashboard    │
└───────────────┴───────────────┘
```

### Debugging Through Visualization
```
┌────────────────────────────────────────────────┐
│ Raw Data    Processed    Semantic    Decision  │
├────────┬────────┬─────────┬────────┬──────────┤
│        │        │         │        │          │
│Camera 1│Camera 2│ Detects │Tracking│ Planning │
│        │        │         │        │          │
├────────┴────────┴─────────┴────────┴──────────┤
│                                                │
│                 Timeline                       │
│  ◄─────────────────────────────────────────►  │
│                                                │
└────────────────────────────────────────────────┘
```


### Temporal Data Visualization
```
     T=0         T=1         T=2         T=3    
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│    •     │ │          │ │          │ │    •     │
│   / \    │ │    •     │ │    •     │ │   / \    │
│  /   \   │ │   / \    │ │   / \    │ │  •   •   │
│ •     •  │ │  •   •   │ │  •   •   │ │ /     \  │
└──────────┘ └──────────┘ └──────────┘ └──────────┘
    State       Action      Prediction   Outcome
```

### Robotic System Monitoring Dashboard
```
┌─────────────────────────────────────────┐
│ System Health                 ▲ CPU: 42% │
├──────────────┬──────────────┬───────────┤
│ 3D Map View  │ Camera View  │ Diagnostics│
│              │              │ ▓▓▓▓▓▓▒▒▒▒ │
│              │              │ ▓▓▓▓▓▓▒▒▒▒ │
│              │              │ Bat: 78%   │
├──────────────┴──────────────┼───────────┤
│ Console Output              │ Parameters │
│ [INFO] Localizing...        │ Max vel: 2m│
│ [INFO] Object detected      │ Map res: 5c│
│ [WARN] Low confidence       │ Conf th: .7│
└─────────────────────────────┴───────────┘
```

## 9. Drawing Practice - Key Diagrams for the Interview

Practice sketching these key diagrams that commonly appear in robotics visualization interviews:

1. **Camera projection model with distortion effects**
2. **Point cloud registration before/after with correspondence lines**
3. **SLAM factor graph with loop closure constraints**
4. **Complete coordinate transformation chain from sensor to world**
5. **Sensor fusion architecture with different fusion levels**
6. **Multi-modal perception system with processing stages**
7. **Visual-inertial odometry pipeline with data flow**
8. **Uncertainty visualization in 3D (ellipsoids, particle sets)**
9. **Robotics system architecture showing visualization components**
10. **Timeline-based multi-sensor debugging interface**