// Numbers transcribed from the arXiv v1 paper (2609.40258). These are paper results (full-scale retraining / ablations),
// kept separate from the search-proxy archive numbers.
export const TABLE1 = {
  caption: 'Sequence-modeling performance and estimated energy reduction (paper Table 1). WT103: WikiText-103 test perplexity; LRA: Long Range Arena accuracy (%). Energy reduction = estimated arithmetic energy of a same-size dense Transformer divided by that of each SNN (architecture-level estimate, not measured device energy).',
  cols: ['WT103 PPL ↓', 'ListOps', 'Text', 'Retrieval', 'Image', 'Pathfinder', 'Path-X', 'LRA avg.', 'Energy red. ↑'],
  best: [0, 1, 1, 1, 1, 1, 1, 1, 0], // 0 = lower/specific handled below
  groups: [
    { name: 'ANN baselines', rows: [
      ['DeltaNet', 27.5, 62.2, 85.1, 91.3, 89.8, 94.2, 94.9, 86.2, null],
      ['S4D-Lin†', null, 60.5, 87.0, 91.0, 87.9, 94.0, '92.8*', 85.5, null],
    ] },
    { name: 'SNN baselines', rows: [
      ['S6-based SNN†', null, 55.7, 77.6, 88.5, 80.1, 83.4, null, null, null],
      ['Dyn-SSM†', 33.2, 60.2, 82.4, 88.8, 87.2, 92.0, 94.4, 84.2, '33.1×'],
      ['SpikingDeltaNet', 34.5, 57.3, 82.2, 90.5, 89.0, 91.2, 92.3, 83.7, '24.7×'],
      ['SpikingMamba2', 38.2, 48.5, 74.0, 81.0, 83.2, 90.4, 89.6, 77.8, '14.8×'],
    ] },
    { name: 'Discovered SNNs (OpenArchEvo)', ours: true, rows: [
      ['NeuroGate', 26.4, 61.8, 82.4, 90.1, 91.2, 93.6, 95.3, 85.7, '31.7×', 'i09-s0557'],
      ['HomeoResSSM', 28.1, 46.2, 84.8, 92.3, 87.6, 86.3, 85.2, 80.4, '17.3×', 'i03-s0335'],
      ['LoopMem', 27.8, 59.5, 81.0, 89.5, 90.5, 88.4, 88.7, 82.9, '50.6×', 'i07-s0884'],
    ] },
  ],
  bold: { 'DeltaNet': [2, 6, 8], 'S4D-Lin†': [3], 'NeuroGate': [1, 5, 7], 'HomeoResSSM': [4], 'LoopMem': [9] },
  foot: '† Task results quoted from SpikingSSMs (Shen et al., 2025) and Dyn-SSM (Zhong et al., 2024). * S4D-Inv result. “—” = unavailable. Bold = best in column.',
};

export const ABLATION = [
  { arch: 'HomeoResSSM', key: 'i03-s0335', kind: 'Spike-activity-dependent computation', rows: [['Discovered', 57.5, null], ['Control driven by continuous inputs', 59.2, '+1.7'], ['Control removed', 64.4, '+6.9']] },
  { arch: 'NeuroGate', key: 'i09-s0557', kind: 'Spiking-specific design choice', rows: [['Discovered (SiLU removed)', 57.2, null], ['SiLU restored', 63.1, '+5.9']] },
  { arch: 'LoopMem', key: 'i07-s0884', kind: 'Spiking-specific design choice', rows: [['Discovered', 60.5, null], ['State-norm feedback removed', 69.3, '+8.8'], ['Non-spiking, with feedback', 61.3, null], ['Non-spiking, feedback removed', 59.7, '−1.6']] },
];

export const BIBTEX = `@article{zhao2026openarchevo,
  title         = {Large Language Model-Guided Evolutionary Discovery of Native Neural Architectures for Spiking Sequence Modeling},
  author        = {Zhao, Ruoyu and Wu, Jiaqi and Zhu, Chenyu and Lu, Zhichao},
  journal       = {arXiv preprint arXiv:2609.40258},
  year          = {2026},
  eprint        = {2609.40258},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2609.40258}
}`;
