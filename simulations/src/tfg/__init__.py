"""Generalized Training-Free Guidance (TFG) for conditional distribution matching.

One unified engine (:mod:`tfg.engine`) implements TFG Algorithm 1 with every
mechanism as a configuration option:

  objective     pointwise (delta target)  |  MMD
  spatial       no smoothing (M=1)        |  LGD (M=3)
  temporal      none                      |  Adam (AdamDPS)  |  adaptive lambda_t
  sample count  constant n_t              |  adaptive n_t
  target        fixed                     |  curriculum (K_t)
  recurrence    fixed N_recur             |  improvement-adaptive

Adam is a temporal option INSIDE the engine, not a separate engine. Its moment
update is verified equal to the official implementation
(github.com/christianbelardi/adam-guidance).

:mod:`tfg.reference` is a frozen transcription of Algorithm 1 used only by the
equivalence tests; it must not be used in experiments.
"""
