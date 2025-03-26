# Modern Robotics Interview Preparation Cheatsheet

## Interview Questions

### Fundamentals
1. Explain the difference between forward and inverse kinematics. When would you use each?
2. What are the advantages and disadvantages of different robot coordinate representations (Euler angles vs. quaternions)?
3. Describe how the Jacobian matrix is used in robotics and its significance.
4. How do you handle kinematic singularities in practice?
5. Explain the mathematical relationship between joint space and task space.

### Dynamics & Control
6. What is the difference between dynamic and kinematic control of a robot?
7. Explain how gravity compensation works in robot control.
8. How would you implement force control vs. position control?
9. Describe the components of the robot equation of motion and what each term represents.
10. Explain the concept of operational space control.

### Advanced Topics
11. How do you approach the multi-task control problem when tasks have conflicting objectives?
12. Explain how contact forces are modeled in legged robotics.
13. What is the difference between soft and hard constraints in robotics?
14. How do hierarchical optimization techniques work for controlling redundant robots?
15. Describe strategies for dealing with joint limits in inverse kinematics.

## Core Concepts

### Coordinate Systems & Transformations

#### Coordinate Representations
- **Cartesian**: x⃗ₚₖ = [x y z]ᵀ
- **Cylindrical**: x⃗ₚₖ = [r θ z]ᵀ, r = [r cos θ, r sin θ, z]ᵀ
- **Spherical**: x⃗ₚₖ = [r θ φ]ᵀ, r = [r cos θ sin φ, r sin θ sin φ, r cos φ]ᵀ

#### Rotation Representations
- **Rotation Matrices**:
  - C₁(z) = [[cos z, -sin z, 0], [sin z, cos z, 0], [0, 0, 1]]
  - C₂(y) = [[cos y, 0, sin y], [0, 1, 0], [-sin y, 0, cos y]]
  - C₃(x) = [[1, 0, 0], [0, cos x, -sin x], [0, sin x, cos x]]

- **Euler Angles**:
  - ZYZ, ZXZ, XYZ (roll-pitch-yaw), etc.
  - Conversion formulas between different representations

- **Euler Parameters (Quaternions)**:
  - Unit quaternion: q = [ξ₀, ξ]ᵀ, Σᵢ₌₀³ ξᵢ² = 1
  - ξ₀ = cos(θ/2), ξ = sin(θ/2)·n
  - Conversion to rotation matrix

- **Angle-Axis**:
  - x⃗ₖₐₙₖₗₑₐₓᵢₛ = (n, θ), ||n|| = 1
  - Euler-Rodrigues formula: φ = θ·n ∈ ℝ³

#### Homogeneous Transformations
- **T matrix**: T = [[R, p], [0, 1]] where R ∈ SO(3), p ∈ ℝ³
- **Inverse**: T⁻¹ = [[Rᵀ, -Rᵀp], [0, 1]]
- **Composition**: Tₐc = Tₐb·Tbc

### Kinematics

#### Forward Kinematics
- **Definition**: Mapping from joint space to task space
- **DH Parameters**: Systematic approach to assign coordinate frames
- **Product of Exponentials**: TₛE(q) = T₀·∏ᵏ₌₁ⁿ Tₖ₋₁,ₖ(qₖ)·T₀,E

#### Inverse Kinematics
- **Analytical Solutions**: Closed-form solutions for specific robot geometries
- **Numerical Solutions**:
  1. q ← q⁰
  2. while ||x* - x₊(q)|| > ε do
     - (a) Δx₊ ← x* - x₊(q) where x*∈E = rot(C*Eᵀ,C₊ᵀ)
     - (b) q ← q + αJ⁺₊(q)·Δx₊
  
- **Trajectory Control**:
  - Position: q* = J⁺₀,ₚ(q*)·(x*₊(t) + kₚΔx₊)
  - Orientation: q* = J⁺₀,ₒ(q*)·(ω*₊(t) + kₚΔφ)

### Jacobians

#### Geometric Jacobian: J₀(q) ∈ ℝ⁶ˣⁿ
- **Definition**: Maps joint velocities to end-effector velocities
  - v₊ = J₀(q)·q̇
- **Components**: J₀ = [Jv; Jω]
- **Addition Formula**: J₀ = J₀ₐ + J₀ᵦ

#### Analytical Jacobian: J₀ₐ(q)
- **Definition**: E₊(x) = C₊·J₀ₐ(q)·q̇
- **Relation**: J₀ₐ(q) = E₊(x)·J₀(q)

#### Inverse Differential Kinematics
- **Joint Velocity**: q̇ = J₀⁺w*₊
- **Pseudo-inverse**: J⁺ = Jᵀ(JJᵀ)⁻¹
- **Null Space Projection**: q̇ = J⁺w*₊ + (I - J⁺J)q̇₀

### Dynamics

#### Equation of Motion
- **General Form**: M(q)q̈ + b(q,q̇) + g(q) = S⊺τ + J₊(q)⊺F₊
- **Components**:
  - M(q) ∈ ℝⁿˣⁿ: Generalized mass matrix
  - b(q,q̇) ∈ ℝⁿ: Coriolis and centrifugal terms
  - g(q) ∈ ℝⁿ: Gravitational terms
  - τ ∈ ℝᵐ: Joint torques
  - F₊ ∈ ℝ⁶: External forces/torques

#### Operational Space Dynamics
- **End-effector Dynamics**: Λ₊ẅ₊ + μ + p = F₊
  - Λ₊ = (J₊M⁻¹J₊ᵀ)⁻¹: End-effector inertia
  - μ = Λ₊J₊M⁻¹b - Λ₊J̇₊q̇: Centrifugal/Coriolis
  - p = Λ₊J₊M⁻¹g: Gravitational terms

### Control

#### Joint Space Control
- **Gravity Compensation**: τ* = kₚ(q* - q) + kᵥ(q̇* - q̇) + g(q)
- **Inverse Dynamics**: q* = kₚ(q* - q) + kᵥ(q̇* - q̇), ω = √kₚ

#### Task Space Control
- **Operational Space Control**: τ* = J⊺(Λₛw*₊ + S₊F₊ + μ + p)
- **Selection Matrices**: 
  - Position: Σₚ = diag(σ₁, ..., σ₆)
  - σᵢ = 1 if axis is free of motion, 0 otherwise

#### Multi-task Control
- **Task Prioritization**: q̇ = Σᵢ₌₁ᵐ Nᵢq̇ᵢ
  - q̇ᵢ = (JᵢNᵢ₋₁)⁺(w*ᵢ - Jᵢq̇ - Jᵢ∑ⱼ₌₁ᶦ⁻¹ Nⱼq̇ⱼ)
  - Nᵢ = I - J⁺ᵢJᵢ: Null space projector

### Legged Robots

#### Optimization Targets
- **Equation of Motion**: [M(q) -J₊ -S⊺][q̈; F₊; τ] = -b(q̇,q) - g(q)
- **End-Effector Velocity**: [J₊ 0 0][q̈; F₊; τ] = ẇ₊ - J̇₊q̇
- **Torque Minimization**: [0 0 I][q̈; F₊; τ] = 0
- **Torque Limits**: [0 0 ±I][q̈; F₊; τ] ≤ ±1·τₘₐₓ
- **Contact Force Minimization**: [0 I 0][q̈; F₊; τ] = 0

#### Friction Cone
- For 2D x-z problem:
  [[0, -1; 1-μ, -1-μ], 0; 0, [0, -1; 1-μ, -1-μ]]·F₊ ≤ 0

### Floating Base Systems

#### Generalized Velocity
- u = [v₀ᴮ; ω₀ᴮ; q̇ₐ] ∈ ℝ⁶⁺ⁿ
- v₀ᴮ, ω₀ᴮ: Base linear/angular velocity
- q̇ₐ: Actuated joint velocities

#### Differential Kinematics
- ₀J₁(q) = [Jᵥ; Jω] = [∂v₀/∂v₀ᴮ, ∂v₀/∂ω₀ᴮ, ∂v₀/∂q̇ₐ; ∂ω₀/∂v₀ᴮ, ∂ω₀/∂ω₀ᴮ, ∂ω₀/∂q̇ₐ]

#### Contact Forces and Constraints
- r₊ = const, ṙ₊ = ṙ₊ = 0
- ₁J₊·u = 0, ₁J̇₊·u + ₁J₊·u̇ = 0

## Advanced Topics

### Constraint-Consistent Dynamics
- N₊ = I - M⁻¹J⊺₊(J₊M⁻¹J⊺₊)⁻¹J₊
- N₊⊺(Mu̇ + b + g) - N₊⊺S⊺τ = 0

### Impulse Transfer
- End-effector inertia: Λ₊ = (J₊M⁻¹J⊺₊)⁻¹
- Instantaneous change: Δu = u⁺ - u⁻ = -S₊⊺J̄⊺₊(J₊M⁻¹J̄⊺₊)⁻¹J₊u⁻
- Post-impact velocity: u⁺ = N₊u⁻
- Energy loss: Eₗₒₛₛ = -½Δu⊺Λ₊Δu

### Hierarchical Optimization
- **Hierarchical Least Squares [HO]**
- Tasks with different priority levels
- Null space projection for lower-priority tasks