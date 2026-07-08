Device: ['0']  |  DataParallel: False
Eval-only — weights: results/600k_training_100_epoch/results_staged_20260707_010506//staged_origin_vertex_jet.pt
Loading data...
  [1/3] Loading flavour labels (cached)
  [2/3] Loading training tracks ...
  [3/3] Loading test tracks ...
Train — b-jet:199,908  c-jet:199,941  light-jet:199,868
Test  — b-jet:77,645  c-jet:18,430  light-jet:103,820
Loaded weights from results/600k_training_100_epoch/results_staged_20260707_010506/staged_origin_vertex_jet.pt
  Vertex fit coordinates: ['Lxy', 'dz']
  Stage-3 extra inputs: ['origin_probs', 'vtx_weight']
  Stage-3 tagging fields (21): ['qOverP', 'deta', 'dphi', 'd0', 'z0SinTheta', 'd0Uncertainty', 'z0SinThetaUncertainty', 'qOverPUncertainty', 'thetaUncertainty', 'phiUncertainty', 'lifetimeSignedD0Significance', 'lifetimeSignedZ0SinThetaSignificance', 'numberOfPixelHits', 'numberOfSCTHits', 'numberOfInnermostPixelLayerHits', 'numberOfNextToInnermostPixelLayerHits', 'numberOfInnermostPixelLayerSharedHits', 'numberOfInnermostPixelLayerSplitHits', 'numberOfPixelSharedHits', 'numberOfPixelSplitHits', 'numberOfSCTSharedHits']
Parameters: 55,377
Device: cuda:0  |  Test: 199,895


Jet classification report:
              precision    recall  f1-score   support

       b-jet       0.85      0.87      0.86     77645
       c-jet       0.17      0.33      0.22     18430
   light-jet       0.90      0.72      0.80    103820

    accuracy                           0.75    199895
   macro avg       0.64      0.64      0.63    199895
weighted avg       0.81      0.75      0.77    199895

Jet confusion matrix (rows=true, cols=pred):
[[67869  6009  3767]
 [ 7442  6023  4965]
 [ 4785 23960 75075]]

Track-origin classification report:
                 precision    recall  f1-score   support

         Pileup       0.92      0.68      0.78    312395
           Fake       0.03      0.54      0.07      1677
        Primary       0.93      0.74      0.82    809100
         From b       0.37      0.43      0.40    133887
      From b->c       0.59      0.40      0.47    204949
         From c       0.11      0.48      0.17     44851
       From tau       0.02      0.23      0.04       585
Other secondary       0.22      0.75      0.34     39761

       accuracy                           0.65   1547205
      macro avg       0.40      0.53      0.39   1547205
   weighted avg       0.79      0.65      0.70   1547205

Saved output_probs.png
Saved origin_confusion_matrix.png
Saved discriminant.png
Saved roc.png

=== b-tagging rejection rates ===
  ε_b=65%:  1/ε_c = 10  1/ε_light = 552
  ε_b=70%:  1/ε_c = 7  1/ε_light = 321
  ε_b=77%:  1/ε_c = 5  1/ε_light = 121
  ε_b=85%:  1/ε_c = 3  1/ε_light = 35
  ε_b=90%:  1/ε_c = 2  1/ε_light = 13
Saved rejection.png
Saved c_discriminant.png
Saved c_roc.png

=== c-tagging rejection rates ===
  ε_c=20%:  1/ε_b = 22  1/ε_light = 9
  ε_c=30%:  1/ε_b = 14  1/ε_light = 5
  ε_c=40%:  1/ε_b = 10  1/ε_light = 3
Saved c_rejection.png
Saved vertex_fit_b_vertex.png
Saved vertex_fit_b_vertex_dz.png
Saved vertex_fit_c_vertex.png
Saved vertex_fit_c_vertex_dz.png

=== Track-to-vertex assignment efficiency ===
  b_vertex  P_leg match:  min=0.0012  mean=0.6892  P25=0.5113  P50=0.7806  P75=0.9182  max=0.9888
  b_vertex  P_leg other:  min=0.0002  mean=0.1842  P50=0.1101  P90=0.4798  max=0.9861
  b_vertex  gate  match:  min=0.0068  mean=0.7344  P25=0.5283  P50=0.9430  P75=0.9850  max=0.9925
  b_vertex  gate  other:  min=0.0067  mean=0.1229  P50=0.0199  P90=0.4496  max=0.9923
  b_vertex  refine match:  min=0.0000  mean=0.0001  P25=0.0000  P50=0.0000  P75=0.0001  max=0.3115
  b_vertex  refine other:  min=0.0000  mean=0.0004  P50=0.0001  P90=0.0007  max=0.2770
  b_vertex  vtx_w range=[0.00671, 0.99252]  mean=0.43712  median=0.23524  P99=0.99158
  b_vertex (thr>0.5): assignment=0.758  false-positive=0.0915  n_match=335470
  b_vertex (thr>0.8): assignment=0.650  false-positive=0.0464  n_match=335470
Saved track_vtx_assignment_b_vertex.png
  c_vertex  P_leg match:  min=0.0004  mean=0.3436  P25=0.2086  P50=0.3561  P75=0.4793  max=0.7775
  c_vertex  P_leg other:  min=0.0000  mean=0.1433  P50=0.0954  P90=0.3666  max=0.7219
  c_vertex  gate  match:  min=0.0067  mean=0.2693  P25=0.0515  P50=0.1916  P75=0.4484  max=0.9413
  c_vertex  gate  other:  min=0.0067  mean=0.0699  P50=0.0172  P90=0.2084  max=0.9019
  c_vertex  refine match:  min=0.0000  mean=0.0001  P25=0.0000  P50=0.0000  P75=0.0000  max=0.0848
  c_vertex  refine other:  min=0.0000  mean=0.0001  P50=0.0000  P90=0.0002  max=0.1420
  c_vertex  vtx_w range=[0.00670, 0.94131]  mean=0.13099  median=0.03170  P99=0.79518
  c_vertex (thr>0.5): assignment=0.211  false-positive=0.0249  n_match=41464
  c_vertex (thr>0.8): assignment=0.027  false-positive=0.0012  n_match=41464
Saved track_vtx_assignment_c_vertex.png

All outputs saved to results/600k_training_100_epoch/results_staged_20260707_010506/eval/
