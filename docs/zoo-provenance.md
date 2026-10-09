# Zoo provenance notes

Secondary details about `zoo/` and `zoo/index.csv`. You do not need them to
load or reuse the programs; read them before comparing numbers closely.

## Metrics

* **WT2 PPL** is the search-proxy WikiText-2 perplexity used for fitness in the paper.
* **ListOps** is the search-proxy accuracy, trained on 1/3 of the ListOps
  training data. It is not a full-data LRA result.
* **Fitness** is computed from these two numbers with `openarchevo/fitness.py`.
* **Fire rates** are training-time averages over all spike tensors at the last
  logged epoch; for WT2 runs that diverged to NaN, the last epoch with a value
  is used. Coverage: WT2 170/171 (`i00-s4` has none), ListOps 36/171, because
  most ListOps logs were not recovered.
* **ASI-Arch rows** come from an earlier, different protocol and have no
  fitness. 12 ASI programs have no results (baselines 8, 9, 10 and ids 295,
  322, 380, 519, 562, 709, 843, 880, 1408).

## Source files

* `index.csv` has 323 rows (20 expert, 151 discovered, 152 ASI-Arch). The
  `elites/` files are not separate rows; the discovered row each one came from
  has subset `paper_analysed` and points to it in `notes`.
* **DeltaFormer / DeltaProduct experts:** the shipped source computes an
  unmasked intra-chunk term; a variant with a causal mask also exists, and
  which one was trained is unresolved.
* **Imports.** Files import `src.models.spike.neuron`,
  `src.models.spike.surrogate` and (23 files) `src.models.nn.dropout`, which
  `load_block_class` maps onto `openarchevo.models`. Seven files also import
  `src.models.spike.nn`, which did not exist in the original code base; by
  default they use their in-file fallback, as in the search. Pass
  `legacy_spike_nn=True` to `load_block_class` to use the package's
  `SpikingLinear` instead.
* ASI baselines 18 and 20 and all `fullspike/` rewrites define
  `SpikingFLABlock`; the other ASI programs define an FLA-style token-mixer
  layer named `DeltaNet`. `asi_arch/10_GLAFFNSWA.py` also needs `flash-attn`.
* Slugs: the paper name for the three analysed architectures, the name given
  in the rationale for 44 others, and `block` otherwise.
* Some `notes` mention an "architecture library", `architectures/STATUS.md` or
  `objects/sha256/...`: the authors' internal provenance archive, not part of
  this repository.

## Testing status

All files parse and match their SHA-256. The three `elites/` files pass a
random-initialisation forward test. On CPU (no FLA, `bptt` neurons), 17 of the
20 FLA-free `SpikingFLABlock` files pass a forward test; `experts/s4.py` and
`asi_arch/20_SpikingSSM.py` hard-code SDN neurons (set `OPENARCHEVO_SDN_PATH`),
and `fullspike/4018_*` needs `use_spiking_mlp` passed explicitly. The
FLA-based files have not been re-run for this release.
