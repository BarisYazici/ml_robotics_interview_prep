# SLAM (Simultaneous Localization and Mapping) Algorithm Cheatsheet

## Introduction to SLAM

SLAM (Simultaneous Localization and Mapping) is an algorithm with the goal of estimating the trajectory of a robot or sensor while simultaneously building a map of the environment.

- It addresses the problem that estimating a map and estimating the robot's pose are interdependent.
- SLAM is often called a "chicken or egg" problem, since knowing the poses makes mapping easy, and knowing the map makes localization easy.

> **Q: What is SLAM and why is it considered a "chicken or egg" problem?**
> 
> A: SLAM stands for Simultaneous Localization and Mapping. It is the process of estimating the trajectory of a robot or sensor and building a map of the environment at the same time. It's a "chicken or egg" problem because estimating the map is easier if the poses are known, and estimating the poses is easier if the map is known. Both quantities are unknown and must be estimated simultaneously, which makes the problem more complex.

### SLAM Components

#### Inputs:
- A sequence of control commands, denoted as U from time step 1 to time step T. These can be raw velocity commands or odometry readings.
- Sensor observations, denoted as Z, from sensors like laser range finders or cameras, providing information about the environment, such as bearing and range information to landmarks.

#### Outputs:
- A map of the environment (m), typically a set of landmark locations. These landmarks are identifiable objects that the robot can detect with its sensors.
- The path of the robot (x), representing the robot's position (x, y) and orientation at different points in time.

> **Q: What are the key components of a SLAM system?**
> 
> A: The key components of a SLAM system include:
> - A motion model that estimates the new pose of the robot given the old pose and control inputs or odometry commands.
> - An observation model that describes the likelihood of an observation given the robot's pose and the map.
> - A backend that optimizes the map and trajectory based on the sensor data and models.
> - A frontend which processes the sensor data and identifies features or landmarks, and determines the constraints between nodes.

> **Q: What is the difference between a motion model and an observation model in SLAM?**
> 
> A:
> - A motion model describes how the robot moves through its environment. It estimates the new pose of the robot, given its previous pose and control inputs such as odometry commands.
> - An observation model describes how the robot perceives its environment through its sensors. It provides the likelihood of an observation, given the robot's pose and the map of the environment. It relates what the robot is expected to measure based on its location.

### SLAM Types

- **Online SLAM** focuses on estimating the robot's current pose, ignoring previous locations.
- **Full SLAM** involves estimating the entire trajectory of the robot and the map of the environment.

> **Q: Explain the difference between full SLAM and online SLAM.**
> 
> A:
> - Full SLAM involves estimating the entire trajectory of the robot and the map of the environment.
> - Online SLAM only estimates the current pose of the robot and the map built up to the current point in time. Online SLAM is crucial for robots making real-time decisions based on their current location and surroundings.

- **Passive SLAM** deals with incoming data streams from sensors without the robot making decisions about where to go.
- **Active SLAM** involves the robot making its own decisions about where to move to build a better map.

> **Q: Can you explain the concepts of 'passive SLAM' versus 'active SLAM'?**
> 
> A:
> - Passive SLAM deals with incoming data streams from sensors without the robot making decisions about where to go. The robot is driven, perhaps with a joystick, and the SLAM algorithm processes the data.
> - Active SLAM, on the other hand, involves the robot making its own decisions about where to move in order to build a better map of the environment. This includes exploration and deciding where to go to improve the map.

## SLAM Approaches

SLAM approaches can be categorized into three paradigms:

### 1. Kalman Filter (EKF) Based Methods

- Utilize the Extended Kalman Filter algorithm to estimate the robot pose and landmark locations.
- EKF is a recursive filter consisting of two steps: prediction and correction.
  - **Prediction**: Uses control commands to predict the new state (pose) of the robot.
  - **Correction**: Uses sensor observations to correct the predicted state.
- The Kalman Gain (K) is computed to weigh the predicted belief and sensor properties. A higher Kalman Gain means more trust is given to sensor observation.
- The state space includes the robot's pose (x, y, theta) and the landmark locations.
- Requires defining a non-linear function G that describes the transition from a given state and control command to a new state and computing its Jacobian.
- Needs an observation function H and its Jacobian.
- Initialization requires defining the mean vector and covariance Matrix.
- The update step involves computing the difference between expected and obtained observations.

#### EKF SLAM Details:
- The prediction step in EKF SLAM only affects a subset of the variables, as motion commands primarily change the robot's position and not the landmark positions.
- Implementing the prediction step requires defining a non-linear function G to move from a given state and control command to the new state, as well as computing its Jacobian.
- The correction step involves computing the Kalman gain K to determine a weighted sum based on the robot's certainty about its predicted belief versus the sensor properties.
- R and Q matrices represent covariance change for the whole state space and measurement covariance, respectively.
- Loop closing in EKF SLAM involves revisiting a known area, which can be challenging due to ambiguities and symmetric environments.
- EKF-based SLAM can face computational problems for large-scale maps, which has led to sub-mapping techniques.

> **Q: Describe the Extended Kalman Filter (EKF) approach to SLAM. What are its limitations?**
> 
> A: The Extended Kalman Filter (EKF) is a commonly used approach to SLAM that uses the Kalman filter framework to estimate the robot's state recursively. It involves a prediction step, where the next state is predicted based on the previous state and control inputs, and a correction step, where the prediction is corrected based on new sensor observations. Because the models are non-linear the EKF uses Jacobian matrices to linearize the functions.
> 
> Limitations of EKF-based SLAM are that it can have computational issues with large-scale maps.

### 2. Particle Filter Based Methods

- Represent the state space using a set of samples or particles.
- Each sample represents a possible state of the system, and the probability distribution is approximated by the distribution of these samples.
- Key steps: proposal distribution, correction (weighting), and resampling.
- Monaco Localization (MCL) uses particles to represent the pose of the robot (x, y, theta) and uses the motion model as a proposal distribution.

#### Particle Filter SLAM Details:
- The particle filter uses samples to represent the posterior and the important sampling principle to update a belief.
- The prediction step involves drawing samples from a proposal distribution using motion odometry, while the correction step uses observations.
- The more samples, the better the estimate.
- The algorithm involves sampling from the proposal distribution, correction to account for the difference between proposal and target distribution, and a resampling step.

> **Q: How do particle filters work, and how can they be applied to the SLAM problem?**
> 
> A: Particle filters use a sample-based representation to approximate probability distributions with a set of samples, or particles. These filters draw samples from a proposal distribution to advance to the next state. The particle weights are adjusted to account for the difference between the proposal and target distributions. Resampling eliminates particles with low weights and duplicates those with high weights to focus on high-probability areas.
> 
> In SLAM, particle filters, like FastSLAM, split up the estimate about the trajectory of the robot from the map of the environment in order to perform state estimation efficiently.

#### FastSLAM:
- A particle filter approach to SLAM that separates the estimation of the robot's trajectory from the estimation of the map. Each particle maintains its own map.
- Uses a sample-based representation to represent the trajectory of the robot.
- The proposal distribution is typically the motion model.
- Importance weights are computed based on the difference between the expected and obtained observations.
- Low variance resampling is used.
- In FastSLAM, each particle maintains only the current pose of the robot and landmark locations, as past trajectories are not revised.
- The importance weight in FastSLAM is computed as the target distribution divided by the proposal distribution.
- A Gaussian distribution is used, taking into account the current observation minus the expected observation.
- Data association can be done on a per-particle basis, which helps in solving the data association problem.
- A limitation of some particle filter approaches is the assumption of a static environment.

#### Grid-based SLAM:
- Uses a variant of FastSLAM to build grid maps.
- Each particle represents a possible trajectory of the robot and maintains its own map.
- Scan matching can be used as a pre-correction step to improve the pose estimate.
- Grid-based SLAM with Rao-Blackwellized Particle Filters uses scan matching to improve pose estimates before applying the particle filter.
- An improved proposal distribution uses the current observation in the proposal distribution of the particle filter.

### 3. Graph-based Approaches

- More modern systems use graph-based approaches due to their flexibility.
- Involves creating a graph where nodes represent robot poses and landmarks, and edges represent constraints between them.
- Aim to reduce the sum of squared errors.
- A front end is used to determine if a constraint is likely to be correct.

> **Q: In the context of graph-based SLAM, what are some advantages of graph-based approaches?**
> 
> A: Graph-based approaches are popular because of their flexibility in terms of linearization and relinearization. The graph-based framework is motivated by reducing the sum of squared errors, which is equivalent to finding the mode of a Gaussian distribution.

## Loop Closure

Loop closure is critical in SLAM for correcting drift and maintaining map accuracy when a robot revisits a known area.

- **Definition**: When the robot revisits a known area after a long traversal.
- **Challenge**: Requires careful data association to avoid ambiguities, especially in symmetric environments.

> **Q: What is loop closure and why is it important in SLAM?**
> 
> A: Loop closure occurs when the robot revisits a known area after a long traversal. It's important for correcting drift and maintaining map accuracy. Loop closure requires careful data association to avoid ambiguities, especially in symmetric environments.

## Frontends and Ambiguity Management

- **Frontends**: Used to determine if a constraint is likely to be correct, helping to manage ambiguities in the environment.
- **Ambiguity**: Addresses the challenge of building accurate maps in ambiguous environments.

> **Q: What is the role of a front end in SLAM? Describe one approach to managing ambiguities in SLAM.**
> 
> A: A SLAM front end is used to determine if a constraint is likely to be correct, helping to manage ambiguities in the environment.
> 
> One approach to managing ambiguities involves performing topological grouping of nearby poses and identifying consistent constraints within these groups. This includes testing for local unambiguousness and global sufficiency.
> 
> - Local unambiguousness ensures that there are no overlapping matches that could lead to a "picket fence" problem.
> - Global sufficiency ensures that there is no possible disjoint match in the uncertainty ellipse.

## ICP (Iterative Closest Point)

ICP is an algorithm used to minimize the difference between two point clouds. It's commonly used in SLAM frontends.

### Core ICP Algorithm
- ICP generally involves two steps: data association and transformation computation.
- Data association typically uses a nearest neighbor approach to match points between two point clouds.
- Transformation computation determines the transformation needed to minimize the distances between corresponding points.
- The process is repeated iteratively until convergence is achieved.

### ICP Variants and Enhancements
- **Point-to-plane ICP**: Assumes points originate from surfaces, minimizing the distance between a point and the tangent plane of the closest point.
- **Generalized ICP**: Combines point-to-point and point-to-plane metrics, incorporating plane-to-plane metrics.
- **Robust kernels**: Reduce the influence of outliers in error minimization.
- **Projective ICP**: Projects a model into a range image and aligns it.

> **Q: What is ICP and how can it be used as a SLAM front end? What are its limitations and how can they be improved?**
> 
> A: ICP (Iterative Closest Point) is an algorithm used to minimize the difference between two point clouds. It can be used as a SLAM front end by iteratively aligning scans to find loop closures. The two primary steps are data association and transformation computation.
> 
> Limitations of ICP include its sensitivity to the initial guess and inefficient sampling strategies.
> 
> Ways to improve ICP in SLAM include arranging scans into maps instead of single scans, separating local perceptions into parts, and using feature descriptors to find good estimates.

> **Q: How does the point-to-plane ICP algorithm differ from the standard point-to-point ICP?**
> 
> A: The point-to-plane ICP algorithm differs from point-to-point ICP by taking into account that the objects being scanned are surfaces. Instead of minimizing the Euclidean distance between points, it minimizes the distance between a point and the estimated tangent plane of the closest point on the other surface. This typically leads to better convergence with fewer iterations.

### Kiss ICP
- A simple and effective system for light detection and ranging (LiDAR) odometry with few parameters.
- Key components include:
  - Motion prediction using odometry, IMU data, or a constant velocity model
  - Scan distortion correction
  - Spatial sub-sampling using voxel grids or 3D hash tables
  - Correspondence search using motion prediction
  - Least squares ICP with a robust kernel for outlier rejection

> **Q: What are some key ingredients of Kiss ICP?**
> 
> A: Kiss ICP is a LiDAR odometry system with a small number of parameters. Key ingredients include:
> - Motion prediction using wheel encoders, an IMU, or a constant velocity model.
> - Scan distortion correction to account for robot motion during a scan.
> - Spatial sub-sampling using voxel grids or 3D hash tables.
> - Correspondence search using motion prediction to improve data association.
> - Least squares ICP with a robust kernel for outlier rejection.

## Additional Techniques and Considerations

### Homogeneous Coordinates
- Used to represent transformations, particularly with bearing-only sensors like cameras.

> **Q: What is the purpose of homogeneous coordinates?**
> 
> A: Homogeneous coordinates are useful when working with sensors that only measure the direction to obstacles, not the distance. An example of this is a camera. A camera projects a 3D world onto a 2D image plane.

### Data Association
- The process of determining which observations correspond to which landmarks or features in the environment.

### Scan Matching
- Used to align scans and improve pose estimates.

### G Mapping
- An open-source implementation of a grid-based SLAM system.

## Limitations and Open Issues in SLAM

- Dealing with dynamic environments.
- Systematically changing environments.
- Seasonal changes.
- Online solutions for larger environments.
- Lifetime operation.
- Resource-constrained systems.
- Failure recovery.
- Exploiting prior information.

> **Q: What are some of the limitations and open issues in SLAM?**
> 
> A: Some limitations and open issues in SLAM include:
> - The assumption of static environments, which is a significant limitation in real-world scenarios.
> - The lack of ability to detect and correct mistakes without user intervention.
> - The challenge of exploiting prior knowledge about the environment.