**Problem addressed:** Traditional linear sequence models treat data strictly as 1D causal streams, heavily bottlenecking their applicability to multi-dimensional data such as vision or video. LightNet introduces a "scan once" mechanism that efficiently integrates multi-dimensional positional information directly into the linear recurrent flow via additive decay factors. Processing multi-dimensional continuous data scales energy costs linearly; SNNs offer a biologically inspired countermeasure by naturally operating over sparse spatial-temporal event spaces.

**Key change implemented:** Spatial-Temporal Spiking Grids. The linear RNN dataflow is re-architected so that the multi-dimensional decay matrices directly regulate the propagation of discrete spikes across defined spatial and temporal axes simultaneously.

**Implementation details:**
- The continuous linear projections are replaced with SpikingLinear instances. The emitted spikes represent discrete events embedded within a multi-dimensional coordinate space.
- The core token mixer logic incorporates LightNet’s multi-dimensional additive decay factors. In the SNN conversion, these decay parameters are mapped to the multi-dimensional leak constants of the LIF neurons.
- Spikes propagate through the hidden state based on this topology. If a specific patch of an image or a frame sequence lacks significant information, no spikes are generated, rendering that specific spatial-temporal block completely dormant and avoiding unnecessary continuous updates.

**Expected impact:** Spiking LightNet offers the NAS surrogate evaluator a powerful architectural block optimized for multi-modal and visual modeling. The inherent temporal and spatial sparsity of SNNs perfectly complements LightNet’s multi-dimensional scanning, yielding a token mixer uniquely tailored for event-camera processing and edge-based computer vision tasks.
