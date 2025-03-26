# Comprehensive Linear Algebra Study Guide

## Table of Contents
1. [Introduction to Matrices](#introduction-to-matrices)
2. [Determinants: Geometric Interpretation and Properties](#determinants-geometric-interpretation-and-properties)
3. [Rotation Matrices](#rotation-matrices)
4. [Matrix Operations and Properties](#matrix-operations-and-properties)
5. [Eigenvalues and Eigenvectors](#eigenvalues-and-eigenvectors)
6. [Special Matrix Decompositions](#special-matrix-decompositions)
7. [Matrix Norms and Metrics](#matrix-norms-and-metrics)
8. [Answers to Interview Questions](#answers-to-interview-questions)

## Introduction to Matrices

### What Are Matrices?
A matrix is a rectangular array of numbers, symbols, or expressions arranged in rows and columns. Matrices are used to represent linear transformations, systems of linear equations, and various mathematical structures.

### Why Matrices Are Linear Transformations
Matrices represent linear transformations because they satisfy the properties of linearity:
1. **Additivity**: T(u + v) = T(u) + T(v)
2. **Homogeneity**: T(αv) = αT(v)

When a matrix A multiplies a vector v, it transforms v into a new vector Av while preserving these linear properties. Every linear transformation between finite-dimensional vector spaces can be represented by a matrix, and every matrix represents a linear transformation.

## Determinants: Geometric Interpretation and Properties

### What Is a Determinant?
The determinant is a scalar value that can be computed from the elements of a square matrix. It provides important information about the matrix.

### Geometric Interpretation
The determinant of a matrix has several geometric interpretations:

1. **Area/Volume Scaling Factor**: For a 2×2 matrix, the determinant represents the area scaling factor when the transformation is applied to a unit square. For a 3×3 matrix, it represents the volume scaling factor of a unit cube.

2. **Orientation**: 
   - If det(A) > 0: The transformation preserves orientation
   - If det(A) < 0: The transformation reverses orientation
   - If det(A) = 0: The transformation collapses space into a lower dimension

3. **Example**: For a 2×2 matrix A = [[a, b], [c, d]], the determinant is ad - bc. This value tells us how much the area changes when we apply the transformation represented by A.

4. **Signed Volume**: For an n×n matrix, the determinant represents the signed n-dimensional volume of the parallelepiped formed by the columns (or rows) of the matrix.

### Key Properties of Determinants

1. **Multiplicativity**: det(AB) = det(A) × det(B)
2. **Transposition**: det(A) = det(A^T)
3. **Row/Column Operations**:
   - Multiplying a row/column by a scalar t multiplies the determinant by t
   - Swapping two rows/columns changes the sign of the determinant
   - Adding a multiple of one row/column to another doesn't change the determinant
4. **Singularity Test**: A matrix is invertible if and only if its determinant is non-zero
5. **For Triangular Matrices**: The determinant equals the product of the diagonal entries

### Calculating Determinants

#### For 2×2 Matrix:
For A = [[a, b], [c, d]], det(A) = ad - bc

#### For 3×3 Matrix:
For A = [[a, b, c], [d, e, f], [g, h, i]]:
det(A) = a(ei - fh) - b(di - fg) + c(dh - eg)

#### For Larger Matrices:
- **Cofactor Expansion**: Expand along a row or column using cofactors
- **Gaussian Elimination**: Transform to triangular form and multiply diagonal elements

## Rotation Matrices

### 2D Rotation Matrix
A counterclockwise rotation by angle θ in 2D space is represented by:
```
R = [cos(θ), -sin(θ)]
    [sin(θ),  cos(θ)]
```

The determinant of any rotation matrix is 1, indicating that rotations preserve area/volume and orientation.

### 3D Rotation Matrices

#### Rotation around X-axis:
```
Rx(θ) = [1,      0,       0]
        [0, cos(θ), -sin(θ)]
        [0, sin(θ),  cos(θ)]
```

#### Rotation around Y-axis:
```
Ry(θ) = [ cos(θ), 0, sin(θ)]
        [      0, 1,      0]
        [-sin(θ), 0, cos(θ)]
```

#### Rotation around Z-axis:
```
Rz(θ) = [cos(θ), -sin(θ), 0]
        [sin(θ),  cos(θ), 0]
        [     0,       0, 1]
```

### Properties of Rotation Matrices
1. **Orthogonality**: R^T R = I (transpose equals inverse)
2. **Determinant = 1**: All proper rotation matrices have determinant 1
3. **Preservation of Length**: ||Rv|| = ||v|| for any vector v
4. **Composition**: Consecutive rotations can be combined through multiplication

## Matrix Operations and Properties

### Matrix Inverse
- The inverse of matrix A is denoted A^(-1)
- A matrix has an inverse if and only if its determinant is non-zero
- For an invertible matrix: A × A^(-1) = A^(-1) × A = I (identity matrix)
- The inverse is unique when it exists

### Rank of a Matrix
- The rank is the dimension of the space spanned by the rows or columns
- It equals the number of linearly independent rows (or columns)
- rank(A) ≤ min(number of rows, number of columns)
- For an m×n matrix with rank r:
  - If r < n: The system Ax = 0 has non-trivial solutions
  - If r = m = n: A is invertible
  - If r < min(m,n): A is rank-deficient

### Trace of a Matrix
- The trace is the sum of the diagonal elements
- trace(A) = Σa_ii
- Properties:
  - trace(A + B) = trace(A) + trace(B)
  - trace(cA) = c·trace(A)
  - trace(AB) = trace(BA)
  - trace(A) = sum of eigenvalues of A

## Eigenvalues and Eigenvectors

### Definition
- An eigenvector v of matrix A is a non-zero vector such that Av = λv for some scalar λ
- The scalar λ is the corresponding eigenvalue
- Geometrically: An eigenvector is only scaled (not rotated) when transformed by A

### Properties
1. The determinant of A equals the product of its eigenvalues
2. The trace of A equals the sum of its eigenvalues
3. A matrix is diagonalizable if and only if it has n linearly independent eigenvectors
4. Similar matrices have the same eigenvalues

### Finding Eigenvalues and Eigenvectors
1. For eigenvalues: Solve det(A - λI) = 0
2. For each eigenvalue λ, find vectors v such that (A - λI)v = 0

### Applications
1. **Principal Component Analysis (PCA)**: Uses eigenvectors of the covariance matrix
2. **Spectral Clustering**: Uses eigenvectors of the graph Laplacian
3. **Markov Processes**: The steady-state of a Markov process is an eigenvector
4. **Google's PageRank**: Uses the principal eigenvector of the web graph

## Special Matrix Decompositions

### Singular Value Decomposition (SVD)
- Any m×n matrix A can be factored as A = UΣV^T where:
  - U is an m×m orthogonal matrix
  - Σ is an m×n diagonal matrix with non-negative singular values
  - V^T is the transpose of an n×n orthogonal matrix
- Applications: Image compression, noise reduction, pseudoinverse calculation

### Eigendecomposition
- For a diagonalizable matrix A, A = PDP^(-1) where:
  - P is a matrix of eigenvectors of A
  - D is a diagonal matrix of eigenvalues
- Not all matrices are diagonalizable

### QR Decomposition
- Any matrix A can be decomposed as A = QR where:
  - Q is an orthogonal matrix
  - R is an upper triangular matrix
- Applications: Solving linear systems, least squares problems

### LU Decomposition
- A square matrix A can be factored as A = LU where:
  - L is a lower triangular matrix
  - U is an upper triangular matrix
- Applications: Efficient solving of linear systems

## Matrix Norms and Metrics

### Matrix Norms
1. **Frobenius Norm**: ||A||_F = sqrt(sum of squares of all elements)
2. **Operator Norm (Induced 2-norm)**: ||A||_2 = largest singular value of A
3. **Nuclear Norm**: Sum of singular values of A

### Distance Between Matrices
- The distance between matrices A and B can be defined using norms:
  - d(A,B) = ||A - B||

### Relationship Between Norms and Metrics
- A norm induces a metric via d(x,y) = ||x - y||
- A metric does not always induce a norm

## Answers to Interview Questions

### Vectors

1. **Geometric interpretation of dot product**:
   The dot product a·b = |a||b|cos(θ) represents the product of the length of one vector and the length of the projection of the other vector onto it. It also represents the amount of "overlap" between two vectors.

2. **Vector v of unit length with maximum dot product with u**:
   The unit vector v = u/||u|| maximizes the dot product, as by Cauchy-Schwarz, u·v ≤ ||u||·||v|| with equality when vectors are parallel.

3. **Outer product of a = [3, 2, 1] and b = [-1, 0, 1]**:
   a^T b = [
     [3(-1), 3(0), 3(1)],
     [2(-1), 2(0), 2(1)],
     [1(-1), 1(0), 1(1)]
   ] = [
     [-3, 0, 3],
     [-2, 0, 2],
     [-1, 0, 1]
   ]

4. **Usefulness of outer product in ML**:
   - Constructing covariance matrices
   - Low-rank matrix approximations
   - Neural network weight updates (e.g., in Hebbian learning)
   - Creating attention mechanisms in transformers (queries ⊗ keys)

5. **Linearly independent vectors**:
   Two vectors are linearly independent if neither can be expressed as a scalar multiple of the other. Geometrically, they don't lie on the same line through the origin.

6. **Checking if two sets of vectors share the same basis**:
   First compute the span of each set (via row reduction to find basis). Then check if span(A) = span(B) by verifying each basis vector of A can be expressed as a linear combination of basis vectors of B and vice versa.

7. **Dimension of span of n vectors in d dimensions**:
   The dimension is at most min(n, d) and equals the rank of the matrix formed by these vectors as columns.

8. **Norms and different types**:
   A norm is a function that assigns a non-negative length to vectors satisfying:
   - L0 "norm": Count of non-zero elements (not actually a norm)
   - L1 norm: Sum of absolute values (Manhattan distance)
   - L2 norm: Euclidean distance (sqrt of sum of squares)
   - L∞ norm: Maximum absolute value among all elements

9. **Norm vs metric difference**:
   - A norm measures vector length from the origin
   - A metric measures distance between two points
   - Given a norm ||·||, we can make a metric d(x,y) = ||x-y||
   - Given a metric, we can make a norm only if the metric is translation-invariant

### Matrices

1. **Matrices as linear transformations**:
   Matrices represent linear transformations because matrix multiplication preserves vector addition and scalar multiplication: A(αu + βv) = αAu + βAv.

2. **Matrix inverse**:
   - The inverse A^(-1) is such that A·A^(-1) = A^(-1)·A = I
   - Only square matrices with non-zero determinant have inverses
   - When it exists, the inverse is always unique

3. **Determinant representation**:
   The determinant represents the scaling factor of the volume transformation induced by the matrix. It also indicates whether the orientation is preserved (positive) or flipped (negative).

4. **Effect of row multiplication on determinant**:
   If we multiply a row by scalar t, the determinant is multiplied by t.

5. **Trace and determinant from eigenvalues**:
   For a matrix with eigenvalues 3, 3, 2, -1:
   - Trace = sum of eigenvalues = 3 + 3 + 2 + (-1) = 7
   - Determinant = product of eigenvalues = 3 × 3 × 2 × (-1) = -18

6. **Determinant of the given matrix**:
   ```
   [1  4  -2]
   [-1 3   2]
   [3  5  -6]
   ```
   Without explicitly calculating, we can use properties of determinants. One approach is to eliminate some entries through row operations, which might simplify the calculation. However, a direct calculation would be more straightforward for a 3×3 matrix.

7. **Difference between A^TA and AA^T**:
   - A^TA (Covariance matrix): Size m×m, represents relationships between features
   - AA^T (Gram matrix): Size n×n, represents relationships between samples
   - They have the same non-zero eigenvalues, but different eigenvectors

8. **Finding x such that Ax = b**:
   - If A is full rank and square: x = A^(-1)b
   - If A is fat (more columns than rows): x = A^T(AA^T)^(-1)b (minimum norm solution)
   - If A is thin (more rows than columns): x = (A^TA)^(-1)A^Tb (least squares solution)

9. **Unique solution conditions**:
   A unique solution exists when the matrix A is square and has full rank (or equivalently, when det(A) ≠ 0).

10. **Multiple solutions with more columns than rows**:
    When A has more columns than rows, the system is underdetermined. The equation Ax = b represents fewer constraints than unknowns, leading to an infinite number of solutions that form an affine subspace.

11. **Solving Ax = b with non-invertible A**:
    Use the Moore-Penrose pseudoinverse: x = A^+b
    The pseudoinverse A^+ = (A^TA)^(-1)A^T (when A has full row rank) or A^+ = A^T(AA^T)^(-1) (when A has full column rank)
    For general cases, computed via SVD: A^+ = VΣ^+U^T where Σ^+ inverts non-zero singular values and zeros out the rest

12. **Derivative representation**:
    A derivative represents the rate of change of a function with respect to its variables. Geometrically, it's the slope of the tangent line to the function at a given point.

13. **Derivative vs gradient vs Jacobian**:
    - Derivative: Rate of change for a scalar function of a single variable (scalar → scalar)
    - Gradient: Vector of partial derivatives for a scalar function of multiple variables (vector → scalar)
    - Jacobian: Matrix of all first-order partial derivatives for a vector function (vector → vector)

14. **Dimension of Jacobian ∂y/∂x**:
    Given y = f(x; w) = xw, where x ∈ R^(n×d) and w ∈ R^(d×m), then y ∈ R^(n×m)
    The Jacobian ∂y/∂x has dimensions (n×m) × (n×d) which can be structured as a tensor of shape (n, m, n, d)

15. **Finding unit vector x to minimize x^TAx**:
    The solution is the eigenvector corresponding to the smallest eigenvalue of A.
    With the given constraints, you could use power iteration with the inverse of A, or by using the function f(x) = Ax to implement a variant of the power method.

### Dimensionality Reduction

1. **Need for dimensionality reduction**:
   - Reduces computational complexity
   - Mitigates the curse of dimensionality
   - Removes noise and redundant features
   - Enables visualization in 2D or 3D
   - Addresses multicollinearity in data

2. **Uniqueness of eigendecomposition**:
   The eigendecomposition of a matrix is generally not unique:
   - If eigenvalues are distinct, eigenvectors are unique up to scaling
   - If eigenvalues have multiplicity > 1, eigenvectors are not unique

3. **Applications of eigenvalues and eigenvectors**:
   - Principal Component Analysis (PCA)
   - Spectral clustering
   - Google's PageRank algorithm
   - Vibration analysis in engineering
   - Quantum mechanics (energy levels)
   - Stability analysis in dynamical systems

4. **PCA on features with different ranges**:
   PCA will not work well directly on this dataset because it's sensitive to the scale of the features. The features with larger ranges will dominate the covariance matrix. The solution is to standardize the data (subtract mean, divide by standard deviation) before applying PCA.

5. **Conditions for eigendecomposition vs SVD**:
   - Eigendecomposition: Applicable to square matrices, and matrix must be diagonalizable
   - SVD: Applicable to any matrix (rectangular or square), always exists

6. **Relationship between SVD and eigendecomposition**:
   - For a matrix A:
     * The right singular vectors of A are eigenvectors of A^TA
     * The left singular vectors of A are eigenvectors of AA^T
     * The singular values of A are square roots of eigenvalues of A^TA or AA^T
   - If A is symmetric positive definite, SVD and eigendecomposition are equivalent

7. **Relationship between PCA and SVD**:
   - PCA finds principal components by computing eigenvectors of the covariance matrix X^TX
   - These eigenvectors are the same as the right singular vectors of X
   - PCA can be implemented via SVD for better numerical stability

8. **t-SNE working principle and need**:
   - t-SNE converts high-dimensional similarities to probability distributions
   - It creates a similar distribution in low dimensions by minimizing KL divergence
   - It uses Student t-distribution in low dimensions to address "crowding problem"
   - We need t-SNE because linear methods like PCA fail to preserve local structure in nonlinear data
   - It's particularly useful for visualization of high-dimensional data clusters # Machine Learning & Robotics: Linear Algebra Cheat Sheet

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

## Interview Focus Points

### Outer Product Properties (Common Interview Question)
- **Definition**: $\mathbf{a} \otimes \mathbf{b} = \mathbf{a}\mathbf{b}^T$
- **Key property**: rank($\mathbf{a} \otimes \mathbf{b}$) = 1 (unless either vector is zero)
  - Proof: Every column is a scalar multiple of $\mathbf{a}$
  - Practical implication: Outer product matrices are always rank-1
- **Eigenvalues**: One non-zero eigenvalue equal to $\mathbf{a}^T\mathbf{b}$, rest are zero
- **Applications in ML**:
  - Weight updates in Hebbian learning
  - Attention mechanisms in transformers
  - Low-rank matrix approximations
  - Constructing covariance matrices

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

### Practical Tips for Interviews
- Practice calculating determinants, matrix multiplication, eigenvectors by hand
- Understand how to interpret eigenvalues/eigenvectors geometrically
- Be able to explain SVD and its applications clearly
- Demonstrate understanding of how linear algebra connects to ML and robotics
- Prepare to work with small examples of transformations, projections, etc.
- Know how to identify and define spaces associated with matrices (null space, column space, etc.)