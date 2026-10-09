# Neural Architecture Evolution Mission

## EXPERIMENTAL CONTEXT & HISTORICAL EVIDENCE
{context}

## ARCHITECTURE EVOLUTION OBJECTIVE
Your mission is to create a breakthrough neural architecture that addresses critical performance limitations identified through experimental evidence while integrating cutting-edge research insights from sequence modeling architectures and neuronmorphic computing. Design and implement an innovative architecture that maintains computational efficiency and spiking neural network natures while achieving superior cognitive capabilities.

## SYSTEMATIC EVOLUTION METHODOLOGY

### PHASE 1: Evidence-Based Analysis Framework

#### 1.1 Architecture Forensics
**Current State Assessment:**
- Examine existing architectural implementations
- Map computational mechanisms, design patterns, and information flow
- Identify core algorithmic approaches and their theoretical foundations
- Document interface constraints and compatibility requirements

#### 1.2 Performance Pattern Recognition
**Historical Evidence Analysis:**
- **Modeling Capability**: Extract optimization challenges from results of loss (fitting) and accuracy/ppl (generalization)
- **Cognitive Potential**: Identify capability gaps across two different tasks (ListOps for long sequence modeling, WikiText2 for natural language processing)
- **Bottleneck Identification**: Pinpoint architectural elements limiting performance vs. those enabling strengths
- **Cross-Architecture Comparison**: Analyze performance patterns across different experimental variants

#### 1.3 Research Integration Strategy
**Theoretical Foundation Building:**
- Map research insights to observed performance limitations
- Identify specific theoretical principles addressing architectural weaknesses
- Synthesize multiple research findings for comprehensive enhancement opportunities
- Validate theoretical applicability through experimental evidence correlation

### PHASE 2: Innovation Design Framework

#### 2.1 Targeted Performance Engineering
**Gap-Specific Solutions:**
- Design architectural modifications targeting the most critical performance bottlenecks
- Create mechanisms leveraging research insights for problematic capability domains
- Balance multiple improvement objectives while maintaining architectural coherence
- Ensure modifications address root causes rather than symptoms

#### 2.2 Theoretical Grounding Protocol
**Research-Driven Design:**
- Ground all modifications in validated theoretical principles
- Ensure mathematical and computational justification for proposed changes
- Verify alignment with established research findings and best practices
- Create novel combinations of insights for breakthrough potential

#### 2.3 Efficiency Optimization Standards
**Computational Constraints:**
- Design using chunked computation patterns for scalability
- Maintain sub-quadratic O(N log N) even O(N) complexity throughout
- Optimize memory usage through efficient processing strategies
- Preserve performance gains within strict complexity bounds

#### 2.4 Neuromorphic Computing Natures
**Brain-like characteristics:**
- Motivate the suitability of spiking neurons and ANN components on sequence modeling tasks
- Consider the possible structure and connnection that might be beneficial for reducing the firing rate
- Encourage SNN operations like SpikingLinear, discourage too much or complicated ANN inefficient operations like multiplication of matrices
- Focus on the design of locality and statefulness of computation (state-based mechanisms like RNN/SSMs/Mamba)
- Encourage a clean structure featuring high cohesion and low coupling, combined with the exclusive use of plausible operators

### PHASE 3: Implementation Excellence Protocol

#### 3.1 Architecture Implementation Standards
**Code Development Requirements:**
- Implement the complete evolved architecture
- Preserve interface compatibility (forward function signatures, __init__ **kwargs)
- Add new parameters with sensible defaults (enabled by default for new features)
- Remove or refactor existing features to prevent architectural bloat
- Implement proper causal masking and information flow constraints

#### 3.2 Quality Assurance Framework
**Technical Excellence Standards:**
- Disable @torch.compile decorators and bf.float16 for robust training
- Preserve chunked processing patterns throughout the architecture
- Ensure causal constraints prevent any information leakage
- Verify sub-quadratic complexity in all implemented operations

#### 3.3 Documentation and Justification
**Innovation Communication:**
- Create comprehensive motivation explaining evolution rationale
- Connect experimental evidence to theoretical insights and implementation decisions
- Justify expected improvements based on research findings
- Provide clear reasoning for all architectural design choices

## TECHNICAL IMPLEMENTATION SPECIFICATIONS

### Spiking Neural Network Requirements
- **Dependency Consistency**: Do not remove any of the existing import statements
- **Preserve SpikingLinear Class**: The class SpikingLinear(nn.Module) must be kept exactly as it is, without modification or deletion. But the class SpikingMLP(nn.Module) can be removed if unnecessary
- **Spiking Linear Layer Requirements**: Use spike neuron from src.models.spike.neuron before linear layers (uniformly use SpikingLinear()). Do not use other implementations of SNN like SpikingJelly
- **Correct Spike Output Format**: A `final_spikes` tensor of all spiking neurons MUST be returned, not the firing rate. This tensor **MUST** be a flat, concatenated torch tensor of binary spikes, created for example by `torch.cat([s.flatten() for s in all_spikes_list])`. It must **NOT** be a firing rate, a list of tensors, or any other format.

### Critical Preservation Requirements
- **Class Structure**: Maintain SpikingFLABlock class name and inheritance hierarchy
- **Interface Stability**: Preserve exact forward function signature compatibility
- **Parameter Compatibility**: Support **kwargs in __init__ for extensibility
- **Dimensional Consistency**: Maintain d_model and core parameter structure

### Implementation Quality Standards
- **Chunked Processing**: All sequence operations must utilize fixed-size chunking
- **Causal Integrity**: Implement strict causal constraints in attention-like mechanisms
- **Complexity Bounds**: Ensure O(N log N) or better for all operations
- **Memory Efficiency**: Design for optimal memory usage with chunked patterns
- **Compilation Safety**: Avoid @torch.compile on utility functions to prevent conflicts

### MANDATORY: Tensor Operations Robustness
- **einops.rearrange() Requirement**: Replace ALL .view()/.reshape() with einops.rearrange()
- **Dynamic Dimension Handling**: Never manually calculate dimensions - use einops inference
- **Batch Size Agnostic**: All operations must work with ANY batch size
- **Runtime Shape Extraction**: Get dimensions from tensor.shape at runtime, not config
- **Adaptive Processing**: Design for actual tensor dimensions, not predetermined values

### Cross-Environment Robustness Standards
- **Universal Compatibility**: Identical performance across training/evaluation/inference
- **Memory Adaptation**: Graceful handling of varying memory constraints
- **Shape Tolerance**: Robust operation with varying input dimensions
- **Resource Awareness**: Automatic adaptation to available computational resources

## INNOVATION TARGET DOMAINS

### Primary Capability Enhancement Areas
- **Extended Context Memory**: Revolutionary long-range dependency handling
- **Multi-Scale Information Integration**: Enhanced temporal and semantic scale processing
- **Adaptive Computational Mechanisms**: Dynamic adjustment based on input characteristics
- **Efficiency-Performance Optimization**: Superior capabilities within complexity constraints
- **Cognitive Task Performance**: Breakthrough improvements in reasoning and comprehension
- **Environmental Robustness**: Consistent performance across execution contexts
- **Resource Efficiency**: Optimal adaptation to computational constraints

## DELIVERABLE SPECIFICATIONS

### PRIMARY DELIVERABLE: Complete Implementation
**Architecture Code (MANDATORY):**
- **Implementation**: Create complete working architecture
- **Innovation Quality**: Embed revolutionary architectural advances in functional code
- **Constraint Compliance**: Preserve class structure, parameters, and interface compatibility
- **Technical Standards**: Maintain sub-quadratic complexity, chunked processing, causal constraints
- **Robustness Implementation**: Use einops.rearrange() universally, ensure batch size independence

### SECONDARY DELIVERABLE: Design Documentation
**Architecture Description:**
- **Naming Convention**: `SpikingFLABlock_[innovation_identifier]` reflecting core innovations
- **Motivation Document**: Comprehensive explanation including:
  - Key architectural innovations and their implementation
  - Research insights applied and expected performance improvements
  - Design choice justification based on experimental evidence
  - Connection between theory, evidence, and implementation

## SUCCESS CRITERIA FRAMEWORK

### Critical Success Factors (Ranked by Priority)
1. **Implementation Excellence**: Successfully create breakthrough architecture
2. **Constraint Adherence**: Maintain class name, parameters, and interface compatibility
3. **Technical Robustness**: Ensure complexity bounds, chunked processing, causal constraints
4. **Universal Compatibility**: Use einops.rearrange() universally, support any batch size
5. **Evidence-Based Innovation**: Embed research insights addressing identified limitations
6. **Performance Targeting**: Implement solutions for specific weakness areas identified

## MISSION EMPHASIS
Your **PRIMARY OBJECTIVE** is implementing breakthrough architectural code that demonstrates robust performance across all execution environments and batch configurations. Create working innovations that directly address identified performance gaps through research-guided architectural evolution. Documentation serves as secondary validation of implemented innovations.

Begin your evolution process by examining the experimental evidence and identifying the most critical architectural improvement opportunities.
