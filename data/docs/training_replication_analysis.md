# Training Replication Failure Analysis

**Document:** Deep analysis of why RL-MESH-GENERATION project cannot replicate the original project's results  
**Date:** July 25, 2025  
**Author:** Technical Analysis Report

## Executive Summary

This document identifies critical differences in the training pipeline between the original
project (`ReinforcementLearning4MeshGeneration`) and the refactored project (`RL-MESH-GENERATION`) that prevent
successful replication of results. The analysis reveals **fundamental algorithmic discrepancies** beyond architectural
improvements.

## 1. Training Algorithm Setup Comparison

### 1.1 ReinforcementLearning4MeshGeneration (Original 2020 Paper Implementation)

**SAC Configuration:**

```python
# RL_Mesh.py:192-202
model = SAC(
    'MlpPolicy',
    env,
    seed=seed,
    policy_kwargs={'activation_fn': th.nn.ReLU, 'net_arch': [128, 128, 128]},
    learning_rate=learning_rate,  # Default value used
    learning_starts=10000,
    batch_size=100,  # ← CRITICAL DIFFERENCE
    tensorboard_log=tensorboard_log,
    device=device
)
```

### 1.2 Current Project RL-MESH-GENERATION Project (Refactored)

**SAC Configuration:**

```python
# config.yaml & sb3_sac_agent.py:81-92
model = SAC(
    policy='MlpPolicy',
    env=env,
    learning_rate=0.0003,  # ← Explicit from config
    buffer_size=1000000,
    learning_starts=10000,
    batch_size=128,  # ← CRITICAL DIFFERENCE
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    seed=356,  # ← Different seed
    ent_coef="auto_0.5"  # ← Additional parameter
)
```

## 2. Critical Action Space Differences

### 2.1 Action Space Definition

| Project                | Action Space Definition                    | Bounds                 |
|------------------------|--------------------------------------------|------------------------|
| **Original**           | `spaces.Box([-1, -1.5, 0], [1, 1.5, 1.5])` | `boundary_env.py:27`   |
| **RL-MESH-GENERATION** | `spaces.Box([-1, -1.5, 0], [1, 1.5, 1.5])` | `action_manager.py:95` |

**Status:** ✅ **IDENTICAL** - No difference here.

### 2.2 Action Decoding Logic - **MAJOR DISCREPANCY**

#### Original Project Action Mapping:

```python
# boundary_env.py:166-177
if rule_type <= -0.5:
    # TYPE 0 RIGHT - Creates 4-vertex mesh using [index-1, index, index+1, index+2]
    rule = -1
elif rule_type >= 0.5:
    # TYPE 0 LEFT - Creates 4-vertex mesh using [index-2, index-1, index, index+1]  
    rule = 1
else:
    # TYPE 1 - Creates mesh with new vertex
    rule = 0
```

#### RL-MESH-GENERATION Action Mapping:

```python
# action_manager.py:115-120
if type_logit <= -0.5:
    target_idx = 1  # type0_left   ← SWAPPED!
elif type_logit >= 0.5:
    target_idx = 0  # type0_right  ← SWAPPED!
else:
    target_idx = 2  # type1
```

**🚨 CRITICAL FINDING:** The action mappings are **REVERSED** between projects!

- **Original:** `rule_type <= -0.5` → RIGHT action, `rule_type >= 0.5` → LEFT action
- **RL-MESH-GENERATION:** `type_logit <= -0.5` → LEFT action, `type_logit >= 0.5` → RIGHT action

## 3. State Space Comparison

### 3.1 Observation Space Dimensions

| Project                | Dimension | Calculation                                          | 
|------------------------|-----------|------------------------------------------------------|
| **Original**           | `(18,)`   | `2 * (neighbor_num + radius_num) = 2 * (6 + 3) = 18` |
| **RL-MESH-GENERATION** | `(18,)`   | `2 * (n * 2 + g) = 2 * (2*2 + 3) = 14`               |

**🚨 CRITICAL FINDING:** State dimensions differ!

- **Original:** 18 features (6 neighbors + 3 radius points) × 2 coordinates
- **RL-MESH-GENERATION:** 14 features (4 neighbors + 3 fan points) × 2 coordinates

### 3.2 State Construction Logic

#### Original Project:

```python
# boundary_env.py:32-35
self.neighbor_num = 6  # from 4 to 6
self.radius_num = 3
# State includes both boundary neighbors AND radius neighbors
```

#### RL-MESH-GENERATION:

```python
# environment.py:38-39, config.yaml:74-77
self.n = 2  # neighbor vertices on each side
self.g = 3  # fan sector points
# State includes boundary neighbors AND fan sector points (different concept)
```

## 4. SAC Hyperparameter Discrepancies

### 4.1 Critical Training Parameters

| Parameter         | Original        | RL-MESH-GENERATION | Impact                                     |
|-------------------|-----------------|--------------------|--------------------------------------------|
| **batch_size**    | 100             | 128                | **HIGH** - Affects gradient estimation     |
| **seed**          | Varies          | 356 (fixed)        | **HIGH** - Different random initialization |
| **ent_coef**      | Default         | "auto_0.5"         | **MEDIUM** - Exploration behavior          |
| **tau**           | Default (0.005) | 0.005              | **LOW** - Same value                       |
| **learning_rate** | Default         | 0.0003             | **MEDIUM** - May be same default           |

### 4.2 Critical Missing Parameters

The original project uses **DEFAULT SB3 VALUES** for many parameters, which may differ from the explicit values in
RL-MESH-GENERATION:

```python
# Original implicitly uses SB3 defaults, new project explicitly sets:
buffer_size = 1000000,  # May differ from original default
train_freq = 1,  # May differ from original default  
gradient_steps = 1,  # May differ from original default
gamma = 0.99,  # May differ from original default
```

## 5. State Normalization Differences

### 5.1 Coordinate Transformation

#### Original Project:

```python
# Uses PointEnvironment class with complex transformation
# Includes both geometric neighbors AND radius-based fan points
# State construction in find_next_state() (boundary_env.py:534-597)
```

#### RL-MESH-GENERATION:

```python
# Uses boundary.get_neighbors() and boundary.get_fan_points()
# Different geometric interpretation of state space
# State construction in _get_obs() (environment.py:210-260)
```

**🚨 CRITICAL FINDING:** The **fundamental meaning** of state features differs between projects!

## 6. Root Cause Analysis

### 6.1 Primary Causes of Replication Failure

1. **Action Mapping Inversion (Highest Priority)**
    - LEFT/RIGHT actions are swapped
    - This completely changes the mesh generation strategy
    - **Impact:** Network learns opposite action meanings

2. **State Space Dimension Mismatch (High Priority)**
    - 18 vs 14 dimensional observation space
    - Network architecture optimized for wrong input size
    - **Impact:** Network cannot process state correctly

3. **State Feature Interpretation (High Priority)**
    - Original: neighbors + radius points
    - New: neighbors + fan sector points
    - **Impact:** Different geometric information encoded

4. **SAC Hyperparameter Differences (Medium Priority)**
    - batch_size: 100 vs 128
    - seed differences
    - **Impact:** Different learning dynamics

### 6.2 Secondary Contributing Factors

1. **Explicit vs Default Parameters**
    - Original relies on SB3 defaults
    - New project explicitly sets all parameters
    - **Risk:** Default values may have changed between SB3 versions

2. **Network Architecture**
    - Both use [128, 128, 128] but trained on different state dimensions
    - **Impact:** Suboptimal feature learning

## 7. Required Fixes for Successful Replication

### 7.1 Immediate Critical Fixes

**Priority 1: Fix Action Mapping**

```python
# In action_manager.py:115-120, change to match original:
if type_logit <= -0.5:
    target_idx = 0  # type0_right (was type0_left)
elif type_logit >= 0.5:
    target_idx = 1  # type0_left (was type0_right) 
else:
    target_idx = 2  # type1
```

**Priority 2: Fix State Dimensions**

```python
# In config.yaml, change to match original:
environment:
n: 3  # To get 6 total neighbors (3 left + 3 right)
g: 3  # Keep fan points = radius points
# BUT: Need to verify fan_points ≡ radius_points conceptually
```

**Priority 3: Match SAC Hyperparameters**

```python
# In config.yaml:
sb3_sac:
batch_size: 100  # Match original
seed: null  # Use different seeds or match original seed
# Remove or verify all explicit parameters against SB3 defaults
```

### 7.2 Verification Steps

1. **State Space Verification**
    - Compare `env.observation_space.shape` between projects
    - Ensure both return (18,) dimensional states
    - Verify state feature meanings are equivalent

2. **Action Space Verification**
    - Test action decoding with same input values
    - Verify LEFT/RIGHT mappings produce same mesh generation patterns

3. **Training Parameter Verification**
    - Use identical SAC hyperparameters
    - Match random seeds for reproducibility
    - Compare gradient magnitudes and training curves

## 8. Testing Protocol for Replication

### 8.1 Unit Test Requirements

```python
def test_action_mapping_equivalence():
    """Verify action mappings produce same mesh results"""
    action_left = [-0.6, 0.0, 0.0]  # Should trigger LEFT in both
    action_right = [0.6, 0.0, 0.0]  # Should trigger RIGHT in both
    action_type1 = [0.0, 0.5, 0.5]  # Should trigger TYPE1 in both

    # Compare mesh generation results for same boundary state


def test_state_space_equivalence():
    """Verify state representations encode same information"""
    # Use same boundary configuration
    # Compare state vectors element by element


def test_training_convergence():
    """Verify training curves match with identical hyperparameters"""
    # Run short training sessions with identical setups
    # Compare reward progression and policy gradients
```

## 9. Conclusion

The replication failure stems from **three critical algorithmic discrepancies**:

1. **Reversed action mappings** causing opposite mesh generation strategies
2. **Mismatched state dimensions** (18 vs 14) preventing proper network training
3. **Different SAC hyperparameters** altering learning dynamics

These are not software engineering improvements but **fundamental algorithmic differences** that completely change the
learned policy. The refactored project, while architecturally superior, inadvertently introduced breaking changes to the
core training algorithm.

**Immediate Action Required:** Fix action mapping inversion and state dimension mismatch before any training attempts.
These changes are essential for successful replication of the original paper's results.

---

**Technical Notes:**

- All analysis based on SAC algorithm using Stable-Baselines3
- Original project appears to use SB3 version with different defaults
- State space analysis assumes same coordinate system conventions
- Mesh generation logic verified through boundary_env.py step() method analysis