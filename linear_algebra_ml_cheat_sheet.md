# Machine Learning & Robotics: Linear Algebra Cheat Sheet

## Core Concepts

### Vectors
- A vector is an ordered list of numbers representing magnitude and direction
- $\mathbf{v} = [v_1, v_2, \ldots, v_n]^T$ in $\mathbb{R}^n$
- **Unit vector**: $\hat{\mathbf{v}} = \frac{\mathbf{v}}{||\mathbf{v}||}$ has length 1
- **Basis vectors**: A set of linearly independent vectors that span the space
- **Standard basis**: $\mathbf{e}_1 = [1,0,0,\ldots]^T$, $\mathbf{e}_2 = [0,1,0,\ldots]^T$, etc.

### Vector Spaces
- **Span**: All possible linear combinations of a set of vectors
- **Linear Independence**: No vector can be expressed as a linear combination of others
- **Basis**: Minimum set of linearly independent vectors that span the space
- **Dimension**: Number of vectors in a basis

### Vector Operations
- **Dot Product**: $\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = ||\mathbf{a}||\cdot||\mathbf{b}||\cos\theta$
  - Returns a scalar
  - Measures similarity between vectors
  - Zero if vectors are orthogonal
- **Cross Product** (in $\mathbb{R}^3$): $\mathbf{a} \times \mathbf{b}$ 
  - Returns a vector perpendicular to both $\mathbf{a}$ and $\mathbf{b}$
  - Magnitude: $||\mathbf{a} \times \mathbf{b}|| = ||\mathbf{a}||\cdot||\mathbf{b}||\sin\theta$
  - Direction: Right-hand rule
- **Outer Product**: $\mathbf{a} \otimes \mathbf{b} = \mathbf{a}\mathbf{b}^T$
  - If $\mathbf{a} \in \mathbb{R}^m$ and $\mathbf{b} \in \mathbb{R}^n$, result is $m \times n$ matrix
  - **Key property**: rank($\mathbf{a} \otimes \mathbf{b}$) = 1 (unless either vector is zero)
  - Each element $(i,j)$ equals $a_i \cdot b_j$
  - Applications: Low-rank approximations, attention mechanisms in transformers

## Matrices

### Basics
- A matrix is a rectangular array of numbers: $A \in \mathbb{R}^{m \times n}$ has $m$ rows, $n$ columns
- $A_{ij}$ or $a_{ij}$ refers to element in row $i$, column $j$
- **Identity matrix** $I$: Has 1's on diagonal, 0's elsewhere
- **Diagonal matrix**: Non-zero elements only on diagonal
- **Symmetric matrix**: $A = A^T$
- **Skew-symmetric matrix**: $A = -A^T$
- **Orthogonal matrix**: $A^T A = A A^T = I$ (columns/rows form orthonormal basis)

### Matrix Operations
- **Transpose**: $A^T$ flips rows and columns
- **Trace**: $\text{tr}(A) = \sum_{i=1}^{n} a_{ii}$ (sum of diagonal elements)
- **Determinant**: $\det(A)$ or $|A|$ (scalar representing volume scaling)
- **Inverse**: $A^{-1}$ such that $A A^{-1} = A^{-1} A = I$ (exists only if $\det(A) \neq 0$)
- **Pseudo-inverse**: $A^+ = (A^T A)^{-1} A^T$ (for full-rank, thin matrices)
- **Matrix multiplication**: $(AB)_{ij} = \sum_{k=1}^{p} a_{ik} b_{kj}$ (if $A$ is $m \times p$ and $B$ is $p \times n$)

### Special Products
- **Hadamard (element-wise) product**: $(A \odot B)_{ij} = a_{ij} b_{ij}$
- **Kronecker product**: $A \otimes B$ creates block matrix by multiplying each element of $A$ by matrix $B$

### Matrix Properties

#### Rank
- The dimension of the space spanned by rows or columns
- Maximum number of linearly independent rows or columns
- $\text{rank}(A) \leq \min(m,n)$ for $A \in \mathbb{R}^{m \times n}$
- **Full rank**: $\text{rank}(A) = \min(m,n)$
- **Rank-deficient**: $\text{rank}(A) < \min(m,n)$
- **Rank of outer product**: $\text{rank}(\mathbf{a}\mathbf{b}^T) = 1$ (if neither vector is zero)
- **Rank of sum**: $\text{rank}(A+B) \leq \text{rank}(A) + \text{rank}(B)$
- **Rank of product**: $\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$

#### Determinant
- Exists only for square matrices
- Geometric interpretation: Scaling factor of volume transformation
- $\det(AB) = \det(A) \cdot \det(B)$
- $\det(A^T) = \det(A)$
- $\det(A^{-1}) = \frac{1}{\det(A)}$ (if $A$ is invertible)
- $\det(A) \neq 0 \iff A \text{ is invertible}$

## Eigendecomposition & Eigenvalues

### Definitions
- **Eigenvalue equation**: $A\mathbf{v} = \lambda \mathbf{v}$
- **Eigenvalue** $\lambda$: Scalar by which eigenvector is scaled
- **Eigenvector** $\mathbf{v}$: Non-zero vector whose direction is unchanged by transformation

### Properties
- **Characteristic polynomial**: $\det(A - \lambda I) = 0$
- **Trace**: $\text{tr}(A) = \sum_{i=1}^{n} \lambda_i$
- **Determinant**: $\det(A) = \prod_{i=1}^{n} \lambda_i$
- **Eigenvalues of $A^T$**: Same as eigenvalues of $A$
- **Eigenvalues of $A^{-1}$**: $\frac{1}{\lambda_i}$ where $\lambda_i$ are eigenvalues of $A$
- **Eigenvalues of $A^n$**: $\lambda_i^n$ where $\lambda_i$ are eigenvalues of $A$

### Eigendecomposition
- **Diagonalization**: If $A$ has $n$ linearly independent eigenvectors, then $A = PDP^{-1}$
  - $P$: Matrix of eigenvectors
  - $D$: Diagonal matrix of eigenvalues
- **Symmetric matrices**: Always diagonalizable with real eigenvalues and orthogonal eigenvectors

## Singular Value Decomposition (SVD)

### Definition
- Any matrix $A \in \mathbb{R}^{m \times n}$ can be factored as $A = U\Sigma V^T$
  - $U \in \mathbb{R}^{m \times m}$: Orthogonal matrix of left singular vectors
  - $\Sigma \in \mathbb{R}^{m \times n}$: Diagonal matrix of singular values
  - $V \in \mathbb{R}^{n \times n}$: Orthogonal matrix of right singular vectors

### Properties
- SVD always exists for any matrix
- Singular values $\sigma_i$ are non-negative and conventionally arranged in descending order
- Number of non-zero singular values equals rank of the matrix
- **Relationship with eigendecomposition**:
  - $U$: Eigenvectors of $AA^T$
  - $V$: Eigenvectors of $A^T A$
  - $\sigma_i^2$: Eigenvalues of both $AA^T$ and $A^T A$

### Applications
- **Low-rank approximation**: Keep top $k$ singular values/vectors
- **Principal Component Analysis** (PCA)
- **Image compression**
- **Pseudoinverse**: $A^+ = V\Sigma^+ U^T$ where $\Sigma^+$ inverts non-zero singular values

## Matrix Calculus

### Derivatives
- **Scalar by Scalar**: $\frac{dy}{dx}$ (standard derivative)
- **Vector by Scalar**: $\frac{d\mathbf{y}}{dx}$ (column vector of derivatives)
- **Scalar by Vector**: $\frac{dy}{d\mathbf{x}}$ (gradient, row vector)
- **Vector by Vector**: $\frac{d\mathbf{y}}{d\mathbf{x}}$ (Jacobian matrix)

### Common Matrix Derivatives
- $\frac{d}{dx}(x^T a) = \frac{d}{dx}(a^T x) = a$
- $\frac{d}{dx}(x^T A x) = (A + A^T)x$
- $\frac{d}{dX}(\text{tr}(AX)) = A^T$
- $\frac{d}{dX}(\text{tr}(XAX^T)) = XA^T + XA$

## Linear Transformations

### Basic Transformations
- **Scaling**: Diagonal matrix with scale factors
- **Rotation**:
  - 2D rotation by angle $\theta$: $R = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$
  - 3D rotations: $R_x(\theta)$, $R_y(\theta)$, $R_z(\theta)$ for rotations around respective axes
- **Shear**: Off-diagonal non-zero elements
- **Projection**: Project onto subspace (e.g., $P = vv^T/(v^T v)$ for projection onto vector $v$)

### Properties of Transformations
- **Linearity**: $T(a\mathbf{x} + b\mathbf{y}) = aT(\mathbf{x}) + bT(\mathbf{y})$
- **Composition**: $(T_2 \circ T_1)(\mathbf{x}) = T_2(T_1(\mathbf{x}))$ corresponds to matrix product $A_2 A_1$
- **Kernel/Null Space**: Set of vectors mapped to zero
- **Range/Column Space**: Set of all possible outputs of transformation

## Machine Learning Applications

### Data Representation
- **Data matrix**: $X \in \mathbb{R}^{n \times d}$ ($n$ samples, $d$ features)
- **Covariance matrix**: $C = \frac{1}{n-1} X^T X$ (after centering)
- **Gram matrix**: $G = X X^T$ (inner products between samples)

### Dimensionality Reduction
- **Principal Component Analysis** (PCA):
  - Find directions of maximum variance using eigenvectors of covariance matrix
  - Project data onto top $k$ eigenvectors
  - Implementation via SVD: $X = U\Sigma V^T$, principal components are columns of $V$
- **t-SNE**:
  - Non-linear dimensionality reduction for visualization
  - Preserves local structure and separates clusters
- **UMAP**:
  - Faster alternative to t-SNE with better global structure preservation

### Linear Regression
- **Ordinary Least Squares**: $\hat{\beta} = (X^T X)^{-1} X^T y$
- **Ridge Regression**: $\hat{\beta} = (X^T X + \lambda I)^{-1} X^T y$
- **Normal Equations**: $X^T X \hat{\beta} = X^T y$

### Neural Networks
- **Linear layer**: $y = Wx + b$
- **Attention mechanism**: Uses outer products of query and key vectors
- **Backpropagation**: Applies chain rule of matrix calculus

## Robotics Applications

### Coordinate Transformations
- **Homogeneous coordinates**: Represent position and orientation using 4×4 matrices
- **Translation matrix**: $T = \begin{bmatrix} I & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix}$
- **Rotation matrix**: $R = \begin{bmatrix} R_{3 \times 3} & \mathbf{0} \\ \mathbf{0}^T & 1 \end{bmatrix}$
- **Transformation composition**: $T_2 T_1$ (applied right to left)

### Rotation Representations
- **Rotation matrices**: 3×3, orthogonal, determinant = 1
- **Euler angles**: (roll, pitch, yaw) or (α, β, γ) - suffer from gimbal lock
- **Axis-angle**: Rotation by angle θ around axis $\mathbf{k}$
- **Quaternions**: 4D extension of complex numbers, avoids gimbal lock
  - $q = [w, x, y, z] = w + xi + yj + zk$ where $w = \cos(\theta/2)$, $[x,y,z] = \sin(\theta/2)\mathbf{k}$

### Kinematics
- **Forward kinematics**: Joint angles → end effector pose
- **Inverse kinematics**: End effector pose → joint angles
- **Jacobian matrix**: Maps joint velocities to end effector velocities
  - $\mathbf{v} = J(\mathbf{q}) \dot{\mathbf{q}}$
  - $J(\mathbf{q}) = \frac{\partial \mathbf{x}}{\partial \mathbf{q}}$

### SLAM (Simultaneous Localization and Mapping)
- **Extended Kalman Filter** (EKF): Linear approximation of non-linear dynamics
- **Particle Filter**: Monte Carlo sampling for robust state estimation
- **Factor Graphs**: Represent SLAM as optimization over graph of constraints

## Common Interview Questions & Answers

### Vector Operations and Properties

1. **Q: What is the geometric interpretation of the dot product?**
   - **A:** The dot product $\mathbf{a} \cdot \mathbf{b} = ||\mathbf{a}||\cdot||\mathbf{b}||\cos\theta$ represents:
     - The projection of one vector onto another, scaled by length
     - A measure of similarity between vectors (maximum when parallel, zero when perpendicular)
     - Work done when a force moves an object along a path

2. **Q: Given vectors $\mathbf{u}$ and $\mathbf{v}$, what's the unit vector that maximizes $\mathbf{u} \cdot \mathbf{v}$?**
   - **A:** The unit vector $\hat{\mathbf{v}} = \frac{\mathbf{u}}{||\mathbf{u}||}$ maximizes the dot product. This follows from the Cauchy-Schwarz inequality, with equality when vectors are parallel.

3. **Q: How would you test if a set of vectors is linearly independent?**
   - **A:** Form a matrix with the vectors as columns and:
     - Check if determinant is non-zero (for square matrices)
     - Check if rank equals the number of vectors
     - Reduce to row echelon form and check for pivot in each column

4. **Q: What are the differences between the L1, L2, and L-infinity norms?**
   - **A:** 
     - **L1 (Manhattan)**: Sum of absolute values, $||\mathbf{x}||_1 = \sum_i |x_i|$
       - Promotes sparsity, less sensitive to outliers
     - **L2 (Euclidean)**: Square root of sum of squares, $||\mathbf{x}||_2 = \sqrt{\sum_i x_i^2}$
       - Penalizes large values more, rotationally invariant
     - **L-infinity**: Maximum absolute value, $||\mathbf{x}||_\infty = \max_i |x_i|$
       - Only influenced by the largest component

5. **Q: What are the properties of the outer product and why is its rank always 1?**
   - **A:** For $\mathbf{a} \otimes \mathbf{b} = \mathbf{a}\mathbf{b}^T$:
     - Every column is a scalar multiple of $\mathbf{a}$, making columns linearly dependent
     - It has exactly one non-zero eigenvalue equal to $\mathbf{a}^T\mathbf{b}$
     - It can be used for low-rank approximations of matrices
     - Applications: Hebbian learning, attention mechanisms, dyadic representation

### Matrices and Determinants

6. **Q: How does changing a row/column affect the determinant of a matrix?**
   - **A:**
     - Multiplying a row/column by scalar $k$ multiplies determinant by $k$
     - Swapping two rows/columns multiplies determinant by -1
     - Adding a multiple of one row/column to another leaves determinant unchanged
     - If any row/column is all zeros, determinant is zero

7. **Q: What is the relationship between eigenvalues, trace, and determinant?**
   - **A:** For a matrix with eigenvalues $\lambda_1, \lambda_2, ..., \lambda_n$:
     - Trace: $\text{tr}(A) = \sum_{i=1}^{n} \lambda_i$
     - Determinant: $\det(A) = \prod_{i=1}^{n} \lambda_i$
     - Characteristic polynomial: $p(\lambda) = \det(A - \lambda I)$

8. **Q: How would you compute the inverse of a 3x3 matrix?**
   - **A:** The inverse can be computed as $A^{-1} = \frac{1}{\det(A)}\text{adj}(A)$ where:
     - $\text{adj}(A)$ is the adjugate matrix (transpose of cofactor matrix)
     - Each entry $(i,j)$ in the cofactor matrix is $(-1)^{i+j}\det(M_{ij})$
     - $M_{ij}$ is the minor obtained by removing row $i$ and column $j$

9. **Q: What is the difference between singular and non-singular matrices?**
   - **A:**
     - **Singular matrix**: $\det(A) = 0$, not invertible, maps some non-zero vector to zero
     - **Non-singular matrix**: $\det(A) \neq 0$, invertible, preserves dimensions
     - Singular matrices have at least one eigenvalue equal to zero
     - Computationally, avoid inverting matrices close to being singular (ill-conditioned)

10. **Q: What is the difference between $A^TA$ and $AA^T$?**
    - **A:**
      - $A^TA$: Square matrix of size $n \times n$ if $A$ is $m \times n$ (covariance matrix in data)
      - $AA^T$: Square matrix of size $m \times m$ (Gram matrix for kernel methods)
      - Share same non-zero eigenvalues but different eigenvectors
      - If $A = U\Sigma V^T$ (SVD), then $A^TA = V\Sigma^2V^T$ and $AA^T = U\Sigma^2U^T$

### Linear Systems and Transformations

11. **Q: How do you solve the system $A\mathbf{x} = \mathbf{b}$ if $A$ is not invertible?**
    - **A:** Use the pseudoinverse: $\mathbf{x} = A^+\mathbf{b}$ where:
      - $A^+ = (A^TA)^{-1}A^T$ if $A$ has full column rank
      - $A^+ = A^T(AA^T)^{-1}$ if $A$ has full row rank
      - For general case, compute via SVD: $A^+ = V\Sigma^+U^T$
      - This gives the minimum norm solution if multiple solutions exist

12. **Q: When can a system of linear equations have no solution, unique solution, or infinite solutions?**
    - **A:** For $A\mathbf{x} = \mathbf{b}$ with $A$ being $m \times n$:
      - **No solution**: When $\mathbf{b}$ is not in the column space of $A$ (rank($A$) < rank($[A|\mathbf{b}]$))
      - **Unique solution**: When $A$ has full column rank ($n$) and rank($A$) = rank($[A|\mathbf{b}]$)
      - **Infinite solutions**: When $A$ has rank $r < n$ and $\mathbf{b}$ is in the column space of $A$

13. **Q: How would you geometrically interpret a matrix with eigenvalues 1, 2, and -0.5?**
    - **A:** The transformation:
      - Preserves direction of first eigenvector
      - Stretches along second eigenvector by factor of 2
      - Shrinks along third eigenvector by factor of 0.5 and reflects its direction
      - The sign change indicates reflection, and overall volume scales by |1 × 2 × (-0.5)| = 1

14. **Q: What is a rotation matrix and what properties does it have?**
    - **A:** A rotation matrix $R$ represents rotation in space without scaling or reflection:
      - Orthogonal: $R^TR = RR^T = I$ (columns/rows form orthonormal basis)
      - Determinant = 1 (preserves orientation and volume)
      - Eigenvalues have magnitude 1 (complex for non-trivial rotations)
      - 2D rotation by angle θ: $R = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$

15. **Q: What is the difference between a rotation matrix and a general orthogonal matrix?**
    - **A:**
      - All rotation matrices are orthogonal, but not all orthogonal matrices are rotations
      - Rotation matrices have determinant = 1
      - Orthogonal matrices with determinant = -1 include reflections
      - Both preserve distances and angles between vectors

### Eigenvalues, SVD, and Decompositions

16. **Q: How do eigenvalues change when we square a matrix?**
    - **A:** If $\lambda_1, \lambda_2, ..., \lambda_n$ are eigenvalues of $A$, then:
      - Eigenvalues of $A^2$ are $\lambda_1^2, \lambda_2^2, ..., \lambda_n^2$
      - Eigenvectors remain the same
      - Negative eigenvalues become positive when squared
      - Zero eigenvalues remain zero

17. **Q: For what types of matrices are all eigenvalues real?**
    - **A:**
      - Symmetric matrices ($A = A^T$)
      - Hermitian matrices ($A = A^*$, complex conjugate transpose)
      - Real-valued tridiagonal matrices
      - Some structured matrices like real symmetric positive definite matrices

18. **Q: What is the difference between eigendecomposition and SVD?**
    - **A:**
      - **Eigendecomposition**: $A = PDP^{-1}$, only for square, diagonalizable matrices
      - **SVD**: $A = U\Sigma V^T$, works for any matrix (rectangular or square)
      - Eigendecomposition uses same basis for input and output spaces
      - SVD uses different orthogonal bases for input and output spaces
      - SVD singular values are always real and non-negative

19. **Q: How does condition number relate to solving linear systems?**
    - **A:** Condition number $\kappa(A) = \frac{\sigma_{max}}{\sigma_{min}}$ (ratio of largest to smallest singular value):
      - Measures sensitivity of solution to changes in input
      - High condition number (ill-conditioned) means small input errors cause large output errors
      - For numerical stability, prefer algorithms that don't explicitly form $A^TA$
      - Rule of thumb: Lose approximately $\log_{10}(\kappa(A))$ digits of precision

20. **Q: When would you use QR decomposition versus LU decomposition?**
    - **A:**
      - **QR decomposition**: Numerically stable for least squares problems, eigenvalue algorithms
      - **LU decomposition**: Faster for solving systems of linear equations
      - QR is preferable for ill-conditioned matrices
      - LU requires pivoting for numerical stability but is more efficient

### Machine Learning Applications

21. **Q: How is SVD used in Principal Component Analysis (PCA)?**
    - **A:** For data matrix $X$ (centered):
      - SVD: $X = U\Sigma V^T$
      - Principal components are the columns of $V$
      - Variance explained by each component proportional to $\sigma_i^2$ (eigenvalues)
      - Projected data: $Z = XV = U\Sigma$ (scores)
      - Advantage: More numerically stable than computing covariance matrix eigenvectors

22. **Q: Why does standardizing features matter before applying PCA?**
    - **A:**
      - Features with larger scales will dominate variance calculations
      - Standardization (subtract mean, divide by std) ensures fair contribution
      - Without standardization, PCA becomes sensitive to units of measurement
      - Standardized PCA is equivalent to correlation matrix eigenvectors rather than covariance

23. **Q: How would you use linear algebra to implement ridge regression?**
    - **A:** Ridge regression minimizes $||X\beta - y||^2 + \lambda||\beta||^2$:
      - Closed form: $\hat{\beta} = (X^TX + \lambda I)^{-1}X^Ty$
      - Computation via SVD: $\hat{\beta} = V \text{diag}\left(\frac{\sigma_i}{\sigma_i^2 + \lambda}\right)U^Ty$
      - Regularization parameter $\lambda$ shrinks coefficients toward zero
      - Effect: Reduces variance, increases bias, helps with multicollinearity

24. **Q: Explain the relationship between the covariance matrix and PCA.**
    - **A:**
      - Covariance matrix $C = \frac{1}{n-1}X^TX$ (for centered $X$)
      - Eigenvectors of $C$ are the principal components (directions of maximum variance)
      - Eigenvalues of $C$ represent variance explained by each component
      - PCA rotation matrix $W$ consists of eigenvectors of $C$ sorted by eigenvalue magnitude
      - Transformed data $Z = XW$ has diagonal covariance matrix

25. **Q: How is linear algebra used in deep learning?**
    - **A:**
      - Neural network layers: $y = \sigma(Wx + b)$ where $W$ is weight matrix
      - Backpropagation uses chain rule of matrix calculus
      - Attention mechanisms: Transformers use scaled dot-product attention $\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
      - Convolutional layers implement cross-correlation as matrix multiplication
      - Batch processing leverages efficient matrix-matrix operations

### Robotics Applications

26. **Q: How are homogeneous transformation matrices used in robotics?**
    - **A:** 4×4 matrices represent both rotation and translation:
      - $T = \begin{bmatrix} R_{3×3} & \mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix}$
      - Chain of transformations calculated via matrix multiplication
      - Forward kinematics: Multiply joint transformations to get end-effector pose
      - Inverse of transformation: $T^{-1} = \begin{bmatrix} R^T & -R^T\mathbf{t} \\ \mathbf{0}^T & 1 \end{bmatrix}$
      - Advantages: Unified representation, computationally efficient

27. **Q: What is the Jacobian matrix in robotics and how is it used?**
    - **A:** The Jacobian $J(\mathbf{q})$ maps joint velocities to end-effector velocities:
      - $\mathbf{v} = J(\mathbf{q})\dot{\mathbf{q}}$ where $\mathbf{v}$ contains linear and angular velocities
      - $J(\mathbf{q}) = \frac{\partial \mathbf{x}}{\partial \mathbf{q}}$ where $\mathbf{x}$ is end-effector pose
      - Used for: Inverse velocity kinematics, singularity analysis, force transmission
      - Pseudo-inverse $J^+$ used when $J$ is not square: $\dot{\mathbf{q}} = J^+\mathbf{v}$
      - Singularities occur when $\det(JJ^T) = 0$

28. **Q: Compare rotation matrices, Euler angles, and quaternions for representing orientation.**
    - **A:**
      - **Rotation matrices**: 9 elements (redundant), no singularities, easy to apply, composition by multiplication
      - **Euler angles**: 3 elements, intuitive, suffer from gimbal lock at certain angles
      - **Quaternions**: 4 elements, no gimbal lock, efficient composition, better numerical properties
      - For interpolation, quaternions provide the shortest path on a sphere (SLERP)

29. **Q: Explain how linear algebra is used in SLAM (Simultaneous Localization and Mapping).**
    - **A:**
      - State estimation via Extended Kalman Filter uses covariance matrices
      - Bundle adjustment optimizes matrices of camera poses and 3D points
      - Graph-based SLAM represents problem as sparse system of linear equations
      - SVD used for robust estimation in presence of outliers
      - Matrix factorization techniques improve computational efficiency

30. **Q: How would you implement the Iterative Closest Point (ICP) algorithm using linear algebra?**
    - **A:** ICP aligns point clouds:
      - Compute centroids: $\mu_X = \frac{1}{N}\sum_{i=1}^{N}X_i$, $\mu_Y = \frac{1}{N}\sum_{i=1}^{N}Y_i$
      - Form correlation matrix: $H = \sum_{i=1}^{N}(X_i-\mu_X)(Y_i-\mu_Y)^T$
      - SVD: $H = USV^T$
      - Optimal rotation: $R = VU^T$
      - Optimal translation: $t = \mu_Y - R\mu_X$
      - Iterate until convergence

### Matrix Decompositions Comparison
| Decomposition | Formula | Requirements | Applications |
|---------------|---------|--------------|--------------|
| Eigendecomposition | $A = PDP^{-1}$ | Square, diagonalizable | Spectral analysis, PCA |
| SVD | $A = U\Sigma V^T$ | Any matrix | PCA, pseudoinverse, compression |
| QR | $A = QR$ | Any matrix | Least squares, orthogonalization |
| LU | $A = LU$ | Square matrix | Efficient linear system solving |
| Cholesky | $A = LL^T$ | SPD matrix | Numerical stability, sampling |

### Numerical Stability Tips
- Use **QR decomposition** rather than normal equations for least squares
- Apply **SVD** for pseudoinverse calculation instead of direct formula
- Use **Cholesky decomposition** when working with symmetric positive definite matrices
- Consider **condition number** when evaluating stability of linear system
- Avoid explicit matrix inverses when possible (solve linear systems instead)
- Use orthogonal matrices when possible (preserve lengths and angles)
- For high-dimensional problems, consider iterative methods instead of direct ones

### Practical Tips for Interviews
- Practice calculating determinants, matrix multiplication, eigenvectors by hand
- Understand how to interpret eigenvalues/eigenvectors geometrically
- Be able to explain SVD and its applications clearly
- Connect linear algebra concepts to practical ML and robotics problems
- Prepare to work with small examples of transformations, projections, etc.
- Know how to identify and define spaces associated with matrices (null space, column space, etc.)
- Practice implementing algorithms like PCA, linear regression from scratch
- Be comfortable with mathematical notation but also able to explain concepts in plain language

## Advanced Practical Applications & Coding Questions

### Implementation Problems

31. **Q: Implement a function to determine if a matrix is positive definite without using library functions.**
    - **A:** Check if all eigenvalues are positive:
      ```python
      def is_positive_definite(A):
          # Check if symmetric/Hermitian
          if not np.allclose(A, A.T):
              return False
          
          # Try Cholesky decomposition (faster than eigenvalues)
          try:
              # Manually implement Cholesky
              n = A.shape[0]
              L = np.zeros_like(A)
              
              for i in range(n):
                  for j in range(i+1):
                      if i == j:  # Diagonal elements
                          s = A[i,i] - np.sum(L[i,:j]**2)
                          if s <= 0:  # Not positive definite
                              return False
                          L[i,i] = np.sqrt(s)
                      else:
                          L[i,j] = (A[i,j] - np.sum(L[i,:j]*L[j,:j])) / L[j,j]
              return True
          except:
              return False
      ```
      - Alternative: Check that all leading principal minors have positive determinants
      - Applications: Covariance matrices, kernel matrices, optimization problems

32. **Q: Write a function to solve a linear system Ax = b using LU decomposition.**
    - **A:**
      ```python
      def lu_solve(A, b):
          n = A.shape[0]
          # Initialize L as identity and U as zero
          L = np.identity(n)
          U = np.zeros((n, n))
          
          # LU Decomposition
          for i in range(n):
              # Upper triangular elements
              for j in range(i, n):
                  U[i,j] = A[i,j] - np.sum(L[i,:i] * U[:i,j])
              
              # Lower triangular elements
              for j in range(i+1, n):
                  L[j,i] = (A[j,i] - np.sum(L[j,:i] * U[:i,i])) / U[i,i]
          
          # Forward substitution: Solve Ly = b
          y = np.zeros(n)
          for i in range(n):
              y[i] = b[i] - np.sum(L[i,:i] * y[:i])
          
          # Back substitution: Solve Ux = y
          x = np.zeros(n)
          for i in range(n-1, -1, -1):
              x[i] = (y[i] - np.sum(U[i,i+1:] * x[i+1:])) / U[i,i]
              
          return x
      ```
      - In practice, add pivoting for numerical stability
      - Advantages: O(n³) for decomposition, but only O(n²) for each new right-hand side

33. **Q: Implement the power iteration method to find the dominant eigenvector of a matrix.**
    - **A:**
      ```python
      def power_iteration(A, max_iter=100, tol=1e-10):
          n = A.shape[0]
          # Start with random vector
          v = np.random.rand(n)
          v = v / np.linalg.norm(v)
          
          for i in range(max_iter):
              # Power iteration step
              Av = A @ v
              v_new = Av / np.linalg.norm(Av)
              
              # Check convergence
              if np.linalg.norm(v_new - v) < tol:
                  break
              v = v_new
              
          # Rayleigh quotient for eigenvalue
          eigenvalue = (v.T @ A @ v) / (v.T @ v)
          return eigenvalue, v
      ```
      - Applications: PageRank algorithm, PCA first component, dominant mode
      - Variations: Inverse power method for smallest eigenvalue, shifted power method

34. **Q: Implement Principal Component Analysis (PCA) from scratch.**
    - **A:**
      ```python
      def pca(X, n_components):
          # Center the data
          X_centered = X - np.mean(X, axis=0)
          
          # Compute covariance matrix
          cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)
          
          # Compute eigenvalues and eigenvectors
          eigenvalues, eigenvectors = np.linalg.eigh(cov)
          
          # Sort by eigenvalues in descending order
          idx = np.argsort(eigenvalues)[::-1]
          eigenvalues = eigenvalues[idx]
          eigenvectors = eigenvectors[:, idx]
          
          # Select top n_components
          components = eigenvectors[:, :n_components]
          
          # Project data
          X_transformed = X_centered @ components
          
          # Calculate explained variance
          explained_variance = eigenvalues[:n_components] / np.sum(eigenvalues)
          
          return X_transformed, components, explained_variance
      ```
      - Alternative implementation using SVD: More numerically stable
      - Optimizations: For high-dimensional data, use randomized SVD

35. **Q: Implement a function to compute the pseudoinverse of a matrix using SVD.**
    - **A:**
      ```python
      def pseudoinverse(A, tol=1e-15):
          # SVD decomposition
          U, s, Vh = np.linalg.svd(A, full_matrices=False)
          
          # Reciprocal of singular values with threshold
          s_inv = np.zeros_like(s)
          s_inv[s > tol] = 1.0 / s[s > tol]
          
          # Construct pseudoinverse
          return Vh.T @ np.diag(s_inv) @ U.T
      ```
      - Applications: Least squares solutions, projection matrices, regularization
      - Numerical considerations: Small singular values can cause instability

### Machine Learning & Robotics Applications

36. **Q: How would you optimize matrix multiplication for large sparse matrices in a deep learning context?**
    - **A:** Several approaches:
      - Use specialized sparse matrix formats (CSR, CSC, COO) that only store non-zero values
      - Apply blocking techniques to improve cache efficiency
      - Implement specialized algorithms like Strassen's algorithm for large matrices
      - Use libraries optimized for sparse operations (SciPy sparse, PyTorch sparse)
      - Consider hardware acceleration (GPU, TPU)
      - For neural networks: Pruning techniques to increase sparsity without accuracy loss

37. **Q: Given a large dataset for classification, explain how you would implement and optimize a linear SVM.**
    - **A:**
      - Formulate as convex optimization problem: min(½||w||² + C∑ξ)
      - For large datasets:
        - Use stochastic gradient descent with hinge loss
        - Implement kernel trick for non-linear boundaries without explicit transformation
        - Use coordinate descent for dual formulation
      - Optimizations:
        - Random projections to reduce dimensions
        - Incremental learning for streaming data
        - Parallelization across features or samples

38. **Q: In a robotics context, how would you handle singularities in the Jacobian matrix when computing inverse kinematics?**
    - **A:** Several approaches:
      - Damped Least Squares (DLS): J⁺ = J^T(JJ^T + λI)^(-1)
        - λ adjusts dynamically based on proximity to singularity
      - SVD-based method: Zero out or limit small singular values
      - Task prioritization: Primary tasks use null space of secondary tasks
      - Redundant manipulators: Use extra DOF to avoid singularities
      - Path planning in joint space rather than Cartesian space
      - For implementation: Monitor condition number as singularity indicator

39. **Q: How would you implement a Kalman filter for robot localization?**
    - **A:**
      ```python
      def kalman_filter_update(x, P, z, H, R, F, Q):
          # Prediction step
          x_pred = F @ x
          P_pred = F @ P @ F.T + Q
          
          # Update step
          y = z - H @ x_pred  # Innovation
          S = H @ P_pred @ H.T + R  # Innovation covariance
          K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain
          
          # Updated state and covariance
          x_new = x_pred + K @ y
          P_new = (np.eye(len(x)) - K @ H) @ P_pred
          
          return x_new, P_new
      ```
      - F: State transition matrix, H: Observation matrix
      - Q: Process noise covariance, R: Measurement noise covariance
      - Optimizations: Use matrix structure (symmetry, sparsity)
      - For nonlinear systems: Extended Kalman Filter linearizes around current estimate

40. **Q: Write a function to estimate a homography matrix between two sets of corresponding points.**
    - **A:**
      ```python
      def estimate_homography(src_points, dst_points):
          n = len(src_points)
          if n < 4:
              raise ValueError("At least 4 corresponding points needed")
              
          # Build matrix A for homogeneous system
          A = np.zeros((2*n, 9))
          for i in range(n):
              x, y = src_points[i]
              u, v = dst_points[i]
              A[2*i] = [x, y, 1, 0, 0, 0, -u*x, -u*y, -u]
              A[2*i+1] = [0, 0, 0, x, y, 1, -v*x, -v*y, -v]
          
          # SVD solution (h is nullspace of A)
          _, _, Vh = np.linalg.svd(A)
          h = Vh[-1]
          
          # Reshape to 3x3 homography matrix
          H = h.reshape(3, 3)
          
          # Normalize
          return H / H[2, 2]
      ```
      - Applications: Image stitching, perspective correction, augmented reality
      - Optimizations: RANSAC for outlier rejection, normalized coordinates for stability

### Numerical Optimization & Computational Considerations

41. **Q: Given a large symmetric positive definite matrix, what is the most efficient way to solve linear systems repeatedly?**
    - **A:**
      - Cholesky factorization: A = LL^T (compute once, reuse for multiple right-hand sides)
      - For extremely large matrices: Incomplete Cholesky + conjugate gradient
      - For specific structures (e.g., banded matrices), use specialized solvers
      - Implementation considerations:
        - Precompute factorization
        - Exploit matrix structure (sparsity pattern)
        - Consider blocked algorithms for cache efficiency

42. **Q: How would you optimize matrix operations for a real-time robotics application?**
    - **A:**
      - Use fixed-size matrices when dimensions are known at compile time
      - Exploit special matrix structures (diagonal, symmetric, triangular)
      - Consider single-precision arithmetic if accuracy requirements permit
      - Precompute and cache frequently used quantities (e.g., inverse dynamics)
      - Use adaptive algorithms that trade precision for speed
      - Hardware-specific optimizations (SIMD, GPU, specialized processors)
      - Incremental updates rather than full recomputation when possible

43. **Q: Implement a function to compute the matrix exponential for small matrices.**
    - **A:**
      ```python
      def matrix_exponential(A, tol=1e-14, max_iter=100):
          n = A.shape[0]
          result = np.identity(n)
          term = np.identity(n)
          for i in range(1, max_iter):
              term = term @ A / i
              result += term
              if np.max(np.abs(term)) < tol:
                  break
          return result
      ```
      - Applications: Linear differential equations, continuous-time dynamics
      - For larger matrices: Use scaling and squaring method or Padé approximation
      - Specialized methods for specific matrix structures (diagonal, triangular)

44. **Q: Explain how to implement an efficient algorithm for linear least squares regularization (ridge regression).**
    - **A:**
      ```python
      def ridge_regression(X, y, alpha):
          n, p = X.shape
          
          if n > p:  # More samples than features
              # Normal equations with regularization
              XTX = X.T @ X
              reg_term = alpha * np.eye(p)
              beta = np.linalg.solve(XTX + reg_term, X.T @ y)
          else:  # More features than samples
              # Dual form (more efficient when p >> n)
              I = np.eye(n)
              beta = X.T @ np.linalg.solve(X @ X.T + alpha * I, y)
              
          return beta
      ```
      - SVD implementation for numerical stability:
        - Decompose X = UΣV^T
        - Compute β = V diag(σᵢ/(σᵢ² + α)) U^T y
      - For very high dimensions: Iterative methods (conjugate gradient)

45. **Q: How would you detect and handle numerical instability in matrix operations?**
    - **A:**
      - Monitor condition number: κ(A) = σₘₐₓ/σₘᵢₙ
      - Use pivoting strategies in decompositions
      - Apply scaling to improve conditioning
      - When inverting matrices, prefer solving systems directly
      - For least squares, use QR or SVD rather than normal equations
      - For testing: Check residuals, perform backward error analysis
      - Use higher precision arithmetic for critical calculations
      - Implementation example (condition number monitoring):
        ```python
        def safe_solve(A, b, cond_threshold=1e14):
            # Check condition number
            s = np.linalg.svd(A, compute_uv=False)
            cond = s[0]/s[-1]
            
            if cond > cond_threshold:
                # Use regularized solution
                return regularized_solve(A, b, alpha=1e-6)
            else:
                # Standard solution
                return np.linalg.solve(A, b)
        ```

### Real-World Application Scenarios

46. **Q: In a SLAM system, the pose graph optimization becomes numerically unstable. How would you diagnose and fix the issue?**
    - **A:**
      - Check for poorly constrained or disconnected subgraphs
      - Examine loop closure constraints for inconsistency
      - Implement robust cost functions (Huber, Cauchy) to minimize outlier impact
      - For large graphs: Use sparse matrix structures and incremental updates
      - Implement covariance scaling based on constraint uncertainty
      - Consider Schur complement for efficient block matrix operations
      - Add regularization terms to improve conditioning
      - Use iterative solvers (Conjugate Gradient, Gauss-Newton) with good initialization

47. **Q: You're implementing a neural network from scratch. How would you optimize the backpropagation for matrix operations?**
    - **A:**
      - Use matrix operations rather than loops: X @ W instead of element-wise
      - Batch processing to leverage efficient matrix-matrix multiplication
      - Cache intermediate results during forward pass for reuse in backprop
      - Exploit sparsity in gradients when using regularization
      - In code:
        ```python
        # Forward pass with caching
        z1 = X @ W1 + b1
        a1 = relu(z1)  # Cache z1
        z2 = a1 @ W2 + b2
        
        # Backward pass
        dz2 = softmax_derivative(z2, y)
        dW2 = a1.T @ dz2 / batch_size
        db2 = np.mean(dz2, axis=0)
        
        da1 = dz2 @ W2.T
        dz1 = da1 * relu_derivative(z1)  # Use cached z1
        dW1 = X.T @ dz1 / batch_size
        db1 = np.mean(dz1, axis=0)
        ```
      - Consider mixed precision training for deep networks

48. **Q: You're working on a real-time control system for a robotic arm. How would you optimize the inverse kinematics computation?**
    - **A:**
      - Analytical solutions for specific robot geometries (faster than numerical)
      - Jacobian transpose method instead of full inversion for approximate solutions
      - Incremental updates for small changes in target position
      - Parallelization across multiple DOF or multiple targets
      - Pre-compute lookup tables for common positions
      - Implement warm-starting from previous solution
      - Code optimization:
        ```python
        # Instead of recomputing full Jacobian matrix:
        def incremental_ik_update(q_current, x_target, x_current, learning_rate=0.1):
            # Compute error
            dx = x_target - x_current
            
            # Approximate update with Jacobian transpose
            J = compute_jacobian(q_current)
            dq = learning_rate * J.T @ dx
            
            # Update joint angles
            q_new = q_current + dq
            
            return q_new
        ```

49. **Q: How would you implement an efficient collaborative filtering algorithm with matrix factorization?**
    - **A:**
      - Optimize matrix factorization: User × Item ≈ P × Q^T
      - Implementation with SGD:
        ```python
        def matrix_factorization(R, P, Q, K, steps=100, alpha=0.0002, beta=0.02):
            Q = Q.T
            for step in range(steps):
                for i in range(len(R)):
                    for j in range(len(R[i])):
                        if R[i][j] > 0:  # Only observed ratings
                            # Error
                            eij = R[i][j] - np.dot(P[i,:], Q[:,j])
                            
                            # Update P and Q
                            for k in range(K):
                                P[i][k] += alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                                Q[k][j] += alpha * (2 * eij * P[i][k] - beta * Q[k][j])
            
            return P, Q.T
        ```
      - Optimizations:
        - Use alternate least squares (ALS) for parallelization
        - Implement early stopping based on validation error
        - Employ factorized models like SVD++ or neural matrix factorization

50. **Q: In computer vision, how would you optimize the bundle adjustment step in Structure from Motion?**
    - **A:**
      - Exploit sparsity structure of the Jacobian and Hessian matrices
      - Schur complement trick to eliminate structure parameters
      - Use iterative methods (PCG, LM) instead of direct solvers
      - Efficient implementation:
        ```python
        def bundle_adjustment_sparse(points_3d, points_2d, camera_params, point_indices, camera_indices):
            # Define residual function
            def fun(params):
                camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
                points_3d = params[n_cameras * 9:].reshape((n_points, 3))
                
                projected = project(points_3d[point_indices], camera_params[camera_indices])
                error = (projected - points_2d).ravel()
                return error
            
            # Compute sparse Jacobian structure
            camera_indices = camera_indices.astype(np.int32)
            point_indices = point_indices.astype(np.int32)
            
            params = np.hstack((camera_params.ravel(), points_3d.ravel()))
            
            # Use sparse bundle adjustment optimizer
            res = least_squares(fun, params, jac_sparsity=sparsity, method='trf')
            return res
        ```
      - Additional optimizations:
        - Incremental reconstruction to provide good initialization
        - Local bundle adjustment for efficiency in large reconstructions
        - GPU acceleration for matrix operations