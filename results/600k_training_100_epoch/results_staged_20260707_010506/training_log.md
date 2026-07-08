Device: ['0']  |  DataParallel: False
Config saved to ./results/results_staged_20260707_010506/config.json
Loading data...
  [1/3] Loading flavour labels (cached)
  [2/3] Loading training tracks ...
  [3/3] Loading test tracks ...
Train — b-jet:199,908  c-jet:199,941  light-jet:199,868
Test  — b-jet:77,645  c-jet:18,430  light-jet:103,820
Saved input_variables.png
Model type: staged_origin_vertex_jet
  Vertex fit coordinates: ['Lxy', 'dz']
  Stage-3 extra inputs: ['origin_probs', 'vtx_weight']
  Stage-3 tagging fields (21): ['qOverP', 'deta', 'dphi', 'd0', 'z0SinTheta', 'd0Uncertainty', 'z0SinThetaUncertainty', 'qOverPUncertainty', 'thetaUncertainty', 'phiUncertainty', 'lifetimeSignedD0Significance', 'lifetimeSignedZ0SinThetaSignificance', 'numberOfPixelHits', 'numberOfSCTHits', 'numberOfInnermostPixelLayerHits', 'numberOfNextToInnermostPixelLayerHits', 'numberOfInnermostPixelLayerSharedHits', 'numberOfInnermostPixelLayerSplitHits', 'numberOfPixelSharedHits', 'numberOfPixelSplitHits', 'numberOfSCTSharedHits']
  Calibrate vertex fit (learnable per-leg scale): True
Origin class weights:
   0  Pileup              0.013
   1  Fake                0.626
   2  Primary             0.005
   3  From b              0.036
   4  From b->c           0.023
   5  From c              0.027
   6  From tau            0.626
   7  Other secondary     0.113
Parameters: 55,377
Device: cuda:0  |  Train: 599,717  |  Test: 199,895

Epoch 01/100  loss=4.4735 (jet=0.7935 origin=1.4876 vtx=2.1924)  val_loss=4.0328 (jet=0.6153 origin=1.2675 vtx=2.1499)  val_acc=0.7811  origin_acc=0.6649  lxy_scale=[b_vertex=0.932, c_vertex=1.131]  dz_scale=[b_vertex=1.125, c_vertex=0.904]
Epoch 02/100  loss=4.1767 (jet=0.7413 origin=1.2879 vtx=2.1475)  val_loss=3.9427 (jet=0.6274 origin=1.1886 vtx=2.1267)  val_acc=0.7658  origin_acc=0.6700  lxy_scale=[b_vertex=0.895, c_vertex=1.150]  dz_scale=[b_vertex=1.337, c_vertex=0.899]
Epoch 03/100  loss=4.1029 (jet=0.7277 origin=1.2477 vtx=2.1275)  val_loss=3.9195 (jet=0.6386 origin=1.1740 vtx=2.1069)  val_acc=0.7534  origin_acc=0.6465  lxy_scale=[b_vertex=0.891, c_vertex=1.132]  dz_scale=[b_vertex=1.556, c_vertex=0.911]
Epoch 04/100  loss=4.0639 (jet=0.7209 origin=1.2288 vtx=2.1141)  val_loss=3.8750 (jet=0.6233 origin=1.1521 vtx=2.0996)  val_acc=0.7657  origin_acc=0.6655  lxy_scale=[b_vertex=0.892, c_vertex=1.122]  dz_scale=[b_vertex=1.731, c_vertex=0.938]
Epoch 05/100  loss=4.0406 (jet=0.7168 origin=1.2185 vtx=2.1054)  val_loss=3.8490 (jet=0.6165 origin=1.1447 vtx=2.0878)  val_acc=0.7668  origin_acc=0.6657  lxy_scale=[b_vertex=0.894, c_vertex=1.109]  dz_scale=[b_vertex=1.893, c_vertex=0.982]
Epoch 06/100  loss=4.0208 (jet=0.7132 origin=1.2096 vtx=2.0980)  val_loss=3.8097 (jet=0.6048 origin=1.1262 vtx=2.0787)  val_acc=0.7726  origin_acc=0.6778  lxy_scale=[b_vertex=0.897, c_vertex=1.104]  dz_scale=[b_vertex=2.010, c_vertex=1.029]
Epoch 07/100  loss=4.0091 (jet=0.7114 origin=1.2042 vtx=2.0935)  val_loss=3.8517 (jet=0.6310 origin=1.1423 vtx=2.0785)  val_acc=0.7536  origin_acc=0.6592  lxy_scale=[b_vertex=0.898, c_vertex=1.105]  dz_scale=[b_vertex=2.184, c_vertex=1.089]
Epoch 08/100  loss=3.9940 (jet=0.7080 origin=1.1987 vtx=2.0872)  val_loss=3.8146 (jet=0.6100 origin=1.1365 vtx=2.0681)  val_acc=0.7657  origin_acc=0.6519  lxy_scale=[b_vertex=0.905, c_vertex=1.102]  dz_scale=[b_vertex=2.403, c_vertex=1.163]
Epoch 09/100  loss=3.9770 (jet=0.7050 origin=1.1920 vtx=2.0801)  val_loss=3.8419 (jet=0.6402 origin=1.1364 vtx=2.0653)  val_acc=0.7501  origin_acc=0.6608  lxy_scale=[b_vertex=0.913, c_vertex=1.095]  dz_scale=[b_vertex=2.683, c_vertex=1.253]
Epoch 10/100  loss=3.9644 (jet=0.7036 origin=1.1885 vtx=2.0723)  val_loss=3.8235 (jet=0.6466 origin=1.1295 vtx=2.0473)  val_acc=0.7411  origin_acc=0.6279  lxy_scale=[b_vertex=0.907, c_vertex=1.085]  dz_scale=[b_vertex=3.026, c_vertex=1.348]
Epoch 11/100  loss=3.9500 (jet=0.7023 origin=1.1843 vtx=2.0634)  val_loss=3.7641 (jet=0.6069 origin=1.1134 vtx=2.0439)  val_acc=0.7688  origin_acc=0.6750  lxy_scale=[b_vertex=0.907, c_vertex=1.079]  dz_scale=[b_vertex=3.418, c_vertex=1.478]
Epoch 12/100  loss=3.9272 (jet=0.7003 origin=1.1813 vtx=2.0456)  val_loss=3.7265 (jet=0.6055 origin=1.1243 vtx=1.9967)  val_acc=0.7665  origin_acc=0.6678  lxy_scale=[b_vertex=0.907, c_vertex=1.092]  dz_scale=[b_vertex=4.216, c_vertex=1.682]
Epoch 13/100  loss=3.8855 (jet=0.7002 origin=1.1773 vtx=2.0081)  val_loss=3.7012 (jet=0.6427 origin=1.1223 vtx=1.9362)  val_acc=0.7386  origin_acc=0.6655  lxy_scale=[b_vertex=0.896, c_vertex=1.092]  dz_scale=[b_vertex=5.564, c_vertex=2.191]
Epoch 14/100  loss=3.8008 (jet=0.6994 origin=1.1765 vtx=1.9250)  val_loss=3.5736 (jet=0.6411 origin=1.0894 vtx=1.8431)  val_acc=0.7435  origin_acc=0.6814  lxy_scale=[b_vertex=0.902, c_vertex=1.127]  dz_scale=[b_vertex=7.352, c_vertex=3.736]
Epoch 15/100  loss=3.7176 (jet=0.6986 origin=1.1728 vtx=1.8462)  val_loss=3.4767 (jet=0.6251 origin=1.1164 vtx=1.7352)  val_acc=0.7550  origin_acc=0.6526  lxy_scale=[b_vertex=0.907, c_vertex=1.184]  dz_scale=[b_vertex=8.642, c_vertex=5.544]
Epoch 16/100  loss=3.6577 (jet=0.6977 origin=1.1739 vtx=1.7860)  val_loss=3.4350 (jet=0.6032 origin=1.1056 vtx=1.7263)  val_acc=0.7661  origin_acc=0.6582  lxy_scale=[b_vertex=0.908, c_vertex=1.236]  dz_scale=[b_vertex=9.414, c_vertex=6.964]
Epoch 17/100  loss=3.5891 (jet=0.6973 origin=1.1702 vtx=1.7216)  val_loss=3.3744 (jet=0.6199 origin=1.1060 vtx=1.6485)  val_acc=0.7558  origin_acc=0.6478  lxy_scale=[b_vertex=0.914, c_vertex=1.277]  dz_scale=[b_vertex=10.484, c_vertex=8.167]
Epoch 18/100  loss=3.5413 (jet=0.6954 origin=1.1677 vtx=1.6781)  val_loss=3.2533 (jet=0.5855 origin=1.0967 vtx=1.5711)  val_acc=0.7795  origin_acc=0.6626  lxy_scale=[b_vertex=0.922, c_vertex=1.307]  dz_scale=[b_vertex=11.058, c_vertex=9.109]
Epoch 19/100  loss=3.4908 (jet=0.6949 origin=1.1663 vtx=1.6296)  val_loss=3.2434 (jet=0.5998 origin=1.0989 vtx=1.5447)  val_acc=0.7670  origin_acc=0.6777  lxy_scale=[b_vertex=0.944, c_vertex=1.359]  dz_scale=[b_vertex=11.822, c_vertex=10.004]
Epoch 20/100  loss=3.4630 (jet=0.6937 origin=1.1656 vtx=1.6038)  val_loss=3.1892 (jet=0.5964 origin=1.0973 vtx=1.4955)  val_acc=0.7749  origin_acc=0.6618  lxy_scale=[b_vertex=0.950, c_vertex=1.388]  dz_scale=[b_vertex=12.249, c_vertex=10.692]
Epoch 21/100  loss=3.4233 (jet=0.6928 origin=1.1639 vtx=1.5665)  val_loss=3.1630 (jet=0.5797 origin=1.0974 vtx=1.4859)  val_acc=0.7808  origin_acc=0.6641  lxy_scale=[b_vertex=0.947, c_vertex=1.407]  dz_scale=[b_vertex=12.857, c_vertex=11.473]
Epoch 22/100  loss=3.3921 (jet=0.6916 origin=1.1612 vtx=1.5393)  val_loss=3.1600 (jet=0.5935 origin=1.1082 vtx=1.4583)  val_acc=0.7723  origin_acc=0.6706  lxy_scale=[b_vertex=0.960, c_vertex=1.438]  dz_scale=[b_vertex=13.342, c_vertex=12.055]
Epoch 23/100  loss=3.3768 (jet=0.6899 origin=1.1616 vtx=1.5254)  val_loss=3.1186 (jet=0.6099 origin=1.1005 vtx=1.4082)  val_acc=0.7623  origin_acc=0.6745  lxy_scale=[b_vertex=0.959, c_vertex=1.454]  dz_scale=[b_vertex=13.599, c_vertex=12.376]
Epoch 24/100  loss=3.3480 (jet=0.6897 origin=1.1609 vtx=1.4974)  val_loss=3.2001 (jet=0.6011 origin=1.0882 vtx=1.5108)  val_acc=0.7673  origin_acc=0.6762  lxy_scale=[b_vertex=0.960, c_vertex=1.467]  dz_scale=[b_vertex=14.000, c_vertex=13.120]
Epoch 25/100  loss=3.3437 (jet=0.6885 origin=1.1612 vtx=1.4940)  val_loss=3.1090 (jet=0.5954 origin=1.0934 vtx=1.4202)  val_acc=0.7749  origin_acc=0.6899  lxy_scale=[b_vertex=0.967, c_vertex=1.454]  dz_scale=[b_vertex=14.065, c_vertex=13.295]
Epoch 26/100  loss=3.3172 (jet=0.6882 origin=1.1590 vtx=1.4701)  val_loss=3.0960 (jet=0.5985 origin=1.0948 vtx=1.4027)  val_acc=0.7725  origin_acc=0.6726  lxy_scale=[b_vertex=0.969, c_vertex=1.470]  dz_scale=[b_vertex=14.714, c_vertex=13.857]
Epoch 27/100  loss=3.3022 (jet=0.6858 origin=1.1567 vtx=1.4596)  val_loss=3.1851 (jet=0.6344 origin=1.0979 vtx=1.4528)  val_acc=0.7431  origin_acc=0.6583  lxy_scale=[b_vertex=0.969, c_vertex=1.456]  dz_scale=[b_vertex=14.957, c_vertex=14.303]
Epoch 28/100  loss=3.3034 (jet=0.6852 origin=1.1562 vtx=1.4620)  val_loss=3.0626 (jet=0.6035 origin=1.0855 vtx=1.3737)  val_acc=0.7675  origin_acc=0.6754  lxy_scale=[b_vertex=0.970, c_vertex=1.486]  dz_scale=[b_vertex=15.077, c_vertex=14.502]
Epoch 29/100  loss=3.2749 (jet=0.6839 origin=1.1553 vtx=1.4357)  val_loss=3.0689 (jet=0.6047 origin=1.0977 vtx=1.3666)  val_acc=0.7658  origin_acc=0.6579  lxy_scale=[b_vertex=0.972, c_vertex=1.486]  dz_scale=[b_vertex=15.366, c_vertex=14.980]
Epoch 30/100  loss=3.2826 (jet=0.6844 origin=1.1555 vtx=1.4427)  val_loss=3.0767 (jet=0.6004 origin=1.0923 vtx=1.3840)  val_acc=0.7667  origin_acc=0.6868  lxy_scale=[b_vertex=0.973, c_vertex=1.499]  dz_scale=[b_vertex=15.455, c_vertex=15.114]
Epoch 31/100  loss=3.2684 (jet=0.6836 origin=1.1539 vtx=1.4308)  val_loss=3.0552 (jet=0.6105 origin=1.0937 vtx=1.3510)  val_acc=0.7597  origin_acc=0.6734  lxy_scale=[b_vertex=0.986, c_vertex=1.499]  dz_scale=[b_vertex=15.591, c_vertex=15.500]
Epoch 32/100  loss=3.2554 (jet=0.6818 origin=1.1536 vtx=1.4200)  val_loss=3.0273 (jet=0.5958 origin=1.0916 vtx=1.3400)  val_acc=0.7699  origin_acc=0.6564  lxy_scale=[b_vertex=0.967, c_vertex=1.519]  dz_scale=[b_vertex=15.741, c_vertex=15.544]
Epoch 33/100  loss=3.2426 (jet=0.6811 origin=1.1536 vtx=1.4080)  val_loss=3.0133 (jet=0.6107 origin=1.0865 vtx=1.3161)  val_acc=0.7594  origin_acc=0.6690  lxy_scale=[b_vertex=0.985, c_vertex=1.509]  dz_scale=[b_vertex=16.117, c_vertex=16.065]
Epoch 34/100  loss=3.2435 (jet=0.6810 origin=1.1528 vtx=1.4097)  val_loss=3.0115 (jet=0.6092 origin=1.0837 vtx=1.3186)  val_acc=0.7604  origin_acc=0.6729  lxy_scale=[b_vertex=0.987, c_vertex=1.512]  dz_scale=[b_vertex=16.069, c_vertex=16.272]
Epoch 35/100  loss=3.2340 (jet=0.6802 origin=1.1512 vtx=1.4026)  val_loss=3.0272 (jet=0.5814 origin=1.0970 vtx=1.3488)  val_acc=0.7791  origin_acc=0.6776  lxy_scale=[b_vertex=0.987, c_vertex=1.518]  dz_scale=[b_vertex=16.386, c_vertex=16.482]
Epoch 36/100  loss=3.2312 (jet=0.6789 origin=1.1510 vtx=1.4013)  val_loss=2.9808 (jet=0.5907 origin=1.0828 vtx=1.3074)  val_acc=0.7726  origin_acc=0.6671  lxy_scale=[b_vertex=0.982, c_vertex=1.508]  dz_scale=[b_vertex=16.515, c_vertex=16.653]
Epoch 37/100  loss=3.2218 (jet=0.6794 origin=1.1506 vtx=1.3917)  val_loss=2.9882 (jet=0.5807 origin=1.0780 vtx=1.3295)  val_acc=0.7770  origin_acc=0.6641  lxy_scale=[b_vertex=0.995, c_vertex=1.530]  dz_scale=[b_vertex=16.573, c_vertex=16.436]
Epoch 38/100  loss=3.2198 (jet=0.6780 origin=1.1493 vtx=1.3925)  val_loss=2.9894 (jet=0.5944 origin=1.0738 vtx=1.3212)  val_acc=0.7682  origin_acc=0.6847  lxy_scale=[b_vertex=0.995, c_vertex=1.531]  dz_scale=[b_vertex=16.521, c_vertex=16.633]
Epoch 39/100  loss=3.2063 (jet=0.6777 origin=1.1485 vtx=1.3801)  val_loss=3.0212 (jet=0.5987 origin=1.0692 vtx=1.3533)  val_acc=0.7667  origin_acc=0.6917  lxy_scale=[b_vertex=0.987, c_vertex=1.535]  dz_scale=[b_vertex=16.791, c_vertex=16.888]
Epoch 40/100  loss=3.2090 (jet=0.6768 origin=1.1486 vtx=1.3836)  val_loss=3.0138 (jet=0.5803 origin=1.0868 vtx=1.3467)  val_acc=0.7819  origin_acc=0.6643  lxy_scale=[b_vertex=0.987, c_vertex=1.523]  dz_scale=[b_vertex=16.774, c_vertex=16.994]
Epoch 41/100  loss=3.2050 (jet=0.6768 origin=1.1473 vtx=1.3809)  val_loss=2.9551 (jet=0.5808 origin=1.0813 vtx=1.2930)  val_acc=0.7773  origin_acc=0.6752  lxy_scale=[b_vertex=0.991, c_vertex=1.541]  dz_scale=[b_vertex=16.841, c_vertex=17.033]
Epoch 42/100  loss=3.1959 (jet=0.6766 origin=1.1483 vtx=1.3710)  val_loss=2.9641 (jet=0.5949 origin=1.0916 vtx=1.2777)  val_acc=0.7719  origin_acc=0.6639  lxy_scale=[b_vertex=0.996, c_vertex=1.548]  dz_scale=[b_vertex=16.978, c_vertex=17.159]
Epoch 43/100  loss=3.1812 (jet=0.6754 origin=1.1458 vtx=1.3599)  val_loss=2.9653 (jet=0.5956 origin=1.0812 vtx=1.2885)  val_acc=0.7724  origin_acc=0.6817  lxy_scale=[b_vertex=0.988, c_vertex=1.546]  dz_scale=[b_vertex=17.102, c_vertex=17.325]
Epoch 44/100  loss=3.1823 (jet=0.6753 origin=1.1458 vtx=1.3611)  val_loss=2.9692 (jet=0.6161 origin=1.0771 vtx=1.2760)  val_acc=0.7514  origin_acc=0.6572  lxy_scale=[b_vertex=0.990, c_vertex=1.539]  dz_scale=[b_vertex=17.356, c_vertex=17.511]
Epoch 45/100  loss=3.1777 (jet=0.6747 origin=1.1451 vtx=1.3579)  val_loss=3.0094 (jet=0.5841 origin=1.0787 vtx=1.3465)  val_acc=0.7740  origin_acc=0.6747  lxy_scale=[b_vertex=0.998, c_vertex=1.549]  dz_scale=[b_vertex=17.427, c_vertex=17.541]
Epoch 46/100  loss=3.1834 (jet=0.6742 origin=1.1444 vtx=1.3648)  val_loss=2.9369 (jet=0.5929 origin=1.0654 vtx=1.2787)  val_acc=0.7699  origin_acc=0.6833  lxy_scale=[b_vertex=0.990, c_vertex=1.534]  dz_scale=[b_vertex=17.435, c_vertex=17.643]
Epoch 47/100  loss=3.1707 (jet=0.6741 origin=1.1438 vtx=1.3528)  val_loss=2.9421 (jet=0.5838 origin=1.0817 vtx=1.2766)  val_acc=0.7726  origin_acc=0.6601  lxy_scale=[b_vertex=0.993, c_vertex=1.550]  dz_scale=[b_vertex=17.566, c_vertex=17.788]
Epoch 48/100  loss=3.1681 (jet=0.6740 origin=1.1438 vtx=1.3504)  val_loss=2.9617 (jet=0.6011 origin=1.0791 vtx=1.2816)  val_acc=0.7666  origin_acc=0.6876  lxy_scale=[b_vertex=0.998, c_vertex=1.564]  dz_scale=[b_vertex=17.389, c_vertex=17.831]
Epoch 49/100  loss=3.1654 (jet=0.6734 origin=1.1434 vtx=1.3486)  val_loss=2.9466 (jet=0.5819 origin=1.0781 vtx=1.2865)  val_acc=0.7805  origin_acc=0.6773  lxy_scale=[b_vertex=1.002, c_vertex=1.553]  dz_scale=[b_vertex=17.644, c_vertex=18.198]
Epoch 50/100  loss=3.1587 (jet=0.6727 origin=1.1423 vtx=1.3437)  val_loss=2.9462 (jet=0.5811 origin=1.0627 vtx=1.3023)  val_acc=0.7722  origin_acc=0.6863  lxy_scale=[b_vertex=0.998, c_vertex=1.561]  dz_scale=[b_vertex=17.509, c_vertex=18.037]
Epoch 51/100  loss=3.1557 (jet=0.6728 origin=1.1423 vtx=1.3406)  val_loss=3.0447 (jet=0.6019 origin=1.0809 vtx=1.3619)  val_acc=0.7635  origin_acc=0.6719  lxy_scale=[b_vertex=1.009, c_vertex=1.567]  dz_scale=[b_vertex=17.675, c_vertex=18.197]
Epoch 52/100  loss=3.1525 (jet=0.6728 origin=1.1414 vtx=1.3383)  val_loss=2.9013 (jet=0.5798 origin=1.0675 vtx=1.2539)  val_acc=0.7735  origin_acc=0.6727  lxy_scale=[b_vertex=0.997, c_vertex=1.576]  dz_scale=[b_vertex=17.907, c_vertex=18.482]
Epoch 53/100  loss=3.1472 (jet=0.6721 origin=1.1414 vtx=1.3338)  val_loss=2.9452 (jet=0.5829 origin=1.0845 vtx=1.2779)  val_acc=0.7783  origin_acc=0.6620  lxy_scale=[b_vertex=1.005, c_vertex=1.570]  dz_scale=[b_vertex=18.023, c_vertex=18.528]
Epoch 54/100  loss=3.1442 (jet=0.6713 origin=1.1407 vtx=1.3322)  val_loss=2.9177 (jet=0.5885 origin=1.0774 vtx=1.2517)  val_acc=0.7727  origin_acc=0.6656  lxy_scale=[b_vertex=0.998, c_vertex=1.563]  dz_scale=[b_vertex=18.043, c_vertex=18.588]
Epoch 55/100  loss=3.1348 (jet=0.6712 origin=1.1409 vtx=1.3227)  val_loss=2.9430 (jet=0.6094 origin=1.0782 vtx=1.2555)  val_acc=0.7568  origin_acc=0.6528  lxy_scale=[b_vertex=1.001, c_vertex=1.569]  dz_scale=[b_vertex=18.082, c_vertex=18.802]
Epoch 56/100  loss=3.1342 (jet=0.6701 origin=1.1397 vtx=1.3244)  val_loss=2.9248 (jet=0.5801 origin=1.0876 vtx=1.2571)  val_acc=0.7742  origin_acc=0.6728  lxy_scale=[b_vertex=0.999, c_vertex=1.583]  dz_scale=[b_vertex=18.449, c_vertex=18.944]
Epoch 57/100  loss=3.1325 (jet=0.6710 origin=1.1386 vtx=1.3230)  val_loss=2.9170 (jet=0.5957 origin=1.0863 vtx=1.2350)  val_acc=0.7735  origin_acc=0.6775  lxy_scale=[b_vertex=1.009, c_vertex=1.583]  dz_scale=[b_vertex=18.384, c_vertex=18.903]
Epoch 58/100  loss=3.1281 (jet=0.6706 origin=1.1395 vtx=1.3180)  val_loss=2.8771 (jet=0.5876 origin=1.0748 vtx=1.2146)  val_acc=0.7745  origin_acc=0.6682  lxy_scale=[b_vertex=0.988, c_vertex=1.576]  dz_scale=[b_vertex=18.472, c_vertex=19.212]
Epoch 59/100  loss=3.1253 (jet=0.6702 origin=1.1383 vtx=1.3169)  val_loss=2.9029 (jet=0.5967 origin=1.0769 vtx=1.2292)  val_acc=0.7686  origin_acc=0.6663  lxy_scale=[b_vertex=1.008, c_vertex=1.594]  dz_scale=[b_vertex=18.420, c_vertex=19.155]
Epoch 60/100  loss=3.1257 (jet=0.6709 origin=1.1378 vtx=1.3170)  val_loss=2.9224 (jet=0.6057 origin=1.0733 vtx=1.2433)  val_acc=0.7588  origin_acc=0.6563  lxy_scale=[b_vertex=0.997, c_vertex=1.571]  dz_scale=[b_vertex=18.656, c_vertex=19.395]
Epoch 61/100  loss=3.1279 (jet=0.6703 origin=1.1382 vtx=1.3195)  val_loss=2.9675 (jet=0.5962 origin=1.0740 vtx=1.2973)  val_acc=0.7658  origin_acc=0.6790  lxy_scale=[b_vertex=1.005, c_vertex=1.575]  dz_scale=[b_vertex=18.592, c_vertex=19.142]
Epoch 62/100  loss=3.1184 (jet=0.6689 origin=1.1375 vtx=1.3120)  val_loss=2.9338 (jet=0.6029 origin=1.0755 vtx=1.2555)  val_acc=0.7617  origin_acc=0.6541  lxy_scale=[b_vertex=1.003, c_vertex=1.566]  dz_scale=[b_vertex=18.672, c_vertex=19.373]
Epoch 63/100  loss=3.1189 (jet=0.6695 origin=1.1380 vtx=1.3114)  val_loss=3.0076 (jet=0.5913 origin=1.0764 vtx=1.3399)  val_acc=0.7664  origin_acc=0.6599  lxy_scale=[b_vertex=1.003, c_vertex=1.581]  dz_scale=[b_vertex=18.616, c_vertex=19.325]
Epoch 64/100  loss=3.1249 (jet=0.6687 origin=1.1373 vtx=1.3189)  val_loss=2.8898 (jet=0.5794 origin=1.0759 vtx=1.2345)  val_acc=0.7803  origin_acc=0.6915  lxy_scale=[b_vertex=1.000, c_vertex=1.572]  dz_scale=[b_vertex=18.603, c_vertex=19.236]
Epoch 65/100  loss=3.1224 (jet=0.6696 origin=1.1366 vtx=1.3162)  val_loss=2.9214 (jet=0.5808 origin=1.0668 vtx=1.2738)  val_acc=0.7762  origin_acc=0.6639  lxy_scale=[b_vertex=1.001, c_vertex=1.573]  dz_scale=[b_vertex=18.512, c_vertex=19.476]
Epoch 66/100  loss=3.1118 (jet=0.6678 origin=1.1369 vtx=1.3071)  val_loss=2.8651 (jet=0.5736 origin=1.0651 vtx=1.2263)  val_acc=0.7807  origin_acc=0.6671  lxy_scale=[b_vertex=1.008, c_vertex=1.590]  dz_scale=[b_vertex=18.617, c_vertex=19.475]
Epoch 67/100  loss=3.1059 (jet=0.6689 origin=1.1361 vtx=1.3009)  val_loss=2.8694 (jet=0.5769 origin=1.0731 vtx=1.2194)  val_acc=0.7765  origin_acc=0.6704  lxy_scale=[b_vertex=1.005, c_vertex=1.612]  dz_scale=[b_vertex=18.715, c_vertex=19.766]
Epoch 68/100  loss=3.1073 (jet=0.6675 origin=1.1357 vtx=1.3041)  val_loss=2.8513 (jet=0.5655 origin=1.0755 vtx=1.2102)  val_acc=0.7832  origin_acc=0.6920  lxy_scale=[b_vertex=1.013, c_vertex=1.579]  dz_scale=[b_vertex=18.876, c_vertex=19.667]
Epoch 69/100  loss=3.1025 (jet=0.6672 origin=1.1362 vtx=1.2991)  val_loss=2.8788 (jet=0.5944 origin=1.0649 vtx=1.2195)  val_acc=0.7682  origin_acc=0.6769  lxy_scale=[b_vertex=1.007, c_vertex=1.599]  dz_scale=[b_vertex=18.977, c_vertex=19.975]
Epoch 70/100  loss=3.1041 (jet=0.6682 origin=1.1348 vtx=1.3011)  val_loss=2.8823 (jet=0.5947 origin=1.0723 vtx=1.2153)  val_acc=0.7656  origin_acc=0.6579  lxy_scale=[b_vertex=1.021, c_vertex=1.594]  dz_scale=[b_vertex=18.925, c_vertex=19.888]
Epoch 71/100  loss=3.1021 (jet=0.6671 origin=1.1351 vtx=1.2999)  val_loss=2.8845 (jet=0.5848 origin=1.0760 vtx=1.2237)  val_acc=0.7756  origin_acc=0.6625  lxy_scale=[b_vertex=1.014, c_vertex=1.618]  dz_scale=[b_vertex=18.926, c_vertex=19.722]
Epoch 72/100  loss=3.0928 (jet=0.6668 origin=1.1347 vtx=1.2913)  val_loss=2.9004 (jet=0.6067 origin=1.0741 vtx=1.2196)  val_acc=0.7614  origin_acc=0.6722  lxy_scale=[b_vertex=1.021, c_vertex=1.610]  dz_scale=[b_vertex=19.037, c_vertex=20.012]
Epoch 73/100  loss=3.0916 (jet=0.6666 origin=1.1345 vtx=1.2904)  val_loss=2.8865 (jet=0.6091 origin=1.0660 vtx=1.2114)  val_acc=0.7633  origin_acc=0.6671  lxy_scale=[b_vertex=1.019, c_vertex=1.608]  dz_scale=[b_vertex=19.061, c_vertex=20.333]
Epoch 74/100  loss=3.0922 (jet=0.6666 origin=1.1343 vtx=1.2912)  val_loss=2.8928 (jet=0.5880 origin=1.0605 vtx=1.2442)  val_acc=0.7728  origin_acc=0.6770  lxy_scale=[b_vertex=1.015, c_vertex=1.644]  dz_scale=[b_vertex=19.104, c_vertex=20.392]
Epoch 75/100  loss=3.0913 (jet=0.6666 origin=1.1349 vtx=1.2897)  val_loss=2.8868 (jet=0.5962 origin=1.0676 vtx=1.2229)  val_acc=0.7668  origin_acc=0.6681  lxy_scale=[b_vertex=1.005, c_vertex=1.627]  dz_scale=[b_vertex=19.167, c_vertex=20.250]
Epoch 76/100  loss=3.1014 (jet=0.6667 origin=1.1354 vtx=1.2994)  val_loss=2.8727 (jet=0.5729 origin=1.0755 vtx=1.2243)  val_acc=0.7818  origin_acc=0.6646  lxy_scale=[b_vertex=1.019, c_vertex=1.610]  dz_scale=[b_vertex=18.821, c_vertex=19.978]
Epoch 77/100  loss=3.0863 (jet=0.6654 origin=1.1337 vtx=1.2872)  val_loss=2.9004 (jet=0.5853 origin=1.0659 vtx=1.2492)  val_acc=0.7723  origin_acc=0.6744  lxy_scale=[b_vertex=1.015, c_vertex=1.623]  dz_scale=[b_vertex=19.271, c_vertex=20.286]
Epoch 78/100  loss=3.0906 (jet=0.6662 origin=1.1328 vtx=1.2917)  val_loss=2.8444 (jet=0.5764 origin=1.0718 vtx=1.1962)  val_acc=0.7760  origin_acc=0.6615  lxy_scale=[b_vertex=1.024, c_vertex=1.621]  dz_scale=[b_vertex=19.260, c_vertex=20.529]
Epoch 79/100  loss=3.0829 (jet=0.6663 origin=1.1338 vtx=1.2828)  val_loss=2.8513 (jet=0.5767 origin=1.0671 vtx=1.2075)  val_acc=0.7781  origin_acc=0.6736  lxy_scale=[b_vertex=1.009, c_vertex=1.608]  dz_scale=[b_vertex=19.412, c_vertex=20.659]
Epoch 80/100  loss=3.0827 (jet=0.6655 origin=1.1319 vtx=1.2853)  val_loss=2.8809 (jet=0.5739 origin=1.0654 vtx=1.2416)  val_acc=0.7799  origin_acc=0.6746  lxy_scale=[b_vertex=1.014, c_vertex=1.636]  dz_scale=[b_vertex=19.428, c_vertex=20.681]
Epoch 81/100  loss=3.0780 (jet=0.6650 origin=1.1325 vtx=1.2805)  val_loss=2.8578 (jet=0.5758 origin=1.0665 vtx=1.2155)  val_acc=0.7751  origin_acc=0.6757  lxy_scale=[b_vertex=1.011, c_vertex=1.641]  dz_scale=[b_vertex=19.434, c_vertex=20.569]
Epoch 82/100  loss=3.0784 (jet=0.6649 origin=1.1324 vtx=1.2811)  val_loss=2.9154 (jet=0.5907 origin=1.0731 vtx=1.2517)  val_acc=0.7682  origin_acc=0.6637  lxy_scale=[b_vertex=1.013, c_vertex=1.629]  dz_scale=[b_vertex=19.580, c_vertex=20.540]
Epoch 83/100  loss=3.0718 (jet=0.6644 origin=1.1326 vtx=1.2748)  val_loss=2.8666 (jet=0.6039 origin=1.0693 vtx=1.1934)  val_acc=0.7606  origin_acc=0.6601  lxy_scale=[b_vertex=1.019, c_vertex=1.631]  dz_scale=[b_vertex=19.822, c_vertex=20.675]
Epoch 84/100  loss=3.0856 (jet=0.6660 origin=1.1333 vtx=1.2863)  val_loss=2.8369 (jet=0.5645 origin=1.0666 vtx=1.2058)  val_acc=0.7825  origin_acc=0.6471  lxy_scale=[b_vertex=1.011, c_vertex=1.629]  dz_scale=[b_vertex=19.834, c_vertex=20.765]
Epoch 85/100  loss=3.0727 (jet=0.6647 origin=1.1309 vtx=1.2771)  val_loss=2.8374 (jet=0.5766 origin=1.0582 vtx=1.2026)  val_acc=0.7749  origin_acc=0.6702  lxy_scale=[b_vertex=1.013, c_vertex=1.648]  dz_scale=[b_vertex=19.767, c_vertex=20.992]
Epoch 86/100  loss=3.0683 (jet=0.6647 origin=1.1311 vtx=1.2725)  val_loss=2.8423 (jet=0.5819 origin=1.0698 vtx=1.1906)  val_acc=0.7726  origin_acc=0.6502  lxy_scale=[b_vertex=1.011, c_vertex=1.615]  dz_scale=[b_vertex=19.813, c_vertex=21.040]
Epoch 87/100  loss=3.0736 (jet=0.6643 origin=1.1318 vtx=1.2775)  val_loss=2.8770 (jet=0.5807 origin=1.0631 vtx=1.2332)  val_acc=0.7788  origin_acc=0.6786  lxy_scale=[b_vertex=1.026, c_vertex=1.643]  dz_scale=[b_vertex=19.584, c_vertex=20.955]
Epoch 88/100  loss=3.0714 (jet=0.6648 origin=1.1319 vtx=1.2746)  val_loss=2.8573 (jet=0.5791 origin=1.0691 vtx=1.2091)  val_acc=0.7769  origin_acc=0.6714  lxy_scale=[b_vertex=1.014, c_vertex=1.618]  dz_scale=[b_vertex=19.959, c_vertex=21.214]
Epoch 89/100  loss=3.0634 (jet=0.6644 origin=1.1306 vtx=1.2684)  val_loss=2.8619 (jet=0.6071 origin=1.0658 vtx=1.1890)  val_acc=0.7551  origin_acc=0.6648  lxy_scale=[b_vertex=1.018, c_vertex=1.647]  dz_scale=[b_vertex=19.923, c_vertex=21.111]
Epoch 90/100  loss=3.0625 (jet=0.6633 origin=1.1317 vtx=1.2676)  val_loss=2.8605 (jet=0.5768 origin=1.0772 vtx=1.2065)  val_acc=0.7764  origin_acc=0.6717  lxy_scale=[b_vertex=1.016, c_vertex=1.654]  dz_scale=[b_vertex=19.946, c_vertex=21.167]
Epoch 91/100  loss=3.0679 (jet=0.6637 origin=1.1320 vtx=1.2722)  val_loss=2.8972 (jet=0.5768 origin=1.0631 vtx=1.2573)  val_acc=0.7783  origin_acc=0.6874  lxy_scale=[b_vertex=1.014, c_vertex=1.630]  dz_scale=[b_vertex=19.926, c_vertex=21.094]
Epoch 92/100  loss=3.0638 (jet=0.6637 origin=1.1296 vtx=1.2704)  val_loss=2.8843 (jet=0.6005 origin=1.0599 vtx=1.2238)  val_acc=0.7614  origin_acc=0.6785  lxy_scale=[b_vertex=1.007, c_vertex=1.659]  dz_scale=[b_vertex=19.894, c_vertex=21.015]
Epoch 93/100  loss=3.0582 (jet=0.6631 origin=1.1296 vtx=1.2656)  val_loss=2.8665 (jet=0.6049 origin=1.0657 vtx=1.1960)  val_acc=0.7522  origin_acc=0.6598  lxy_scale=[b_vertex=1.018, c_vertex=1.640]  dz_scale=[b_vertex=19.845, c_vertex=21.143]
Epoch 94/100  loss=3.0510 (jet=0.6627 origin=1.1282 vtx=1.2600)  val_loss=2.8476 (jet=0.5780 origin=1.0664 vtx=1.2031)  val_acc=0.7788  origin_acc=0.6620  lxy_scale=[b_vertex=1.018, c_vertex=1.641]  dz_scale=[b_vertex=20.147, c_vertex=21.323]
Epoch 95/100  loss=3.0710 (jet=0.6629 origin=1.1297 vtx=1.2785)  val_loss=2.8666 (jet=0.5845 origin=1.0678 vtx=1.2142)  val_acc=0.7717  origin_acc=0.6626  lxy_scale=[b_vertex=1.032, c_vertex=1.656]  dz_scale=[b_vertex=19.970, c_vertex=21.043]
Epoch 96/100  loss=3.0717 (jet=0.6635 origin=1.1315 vtx=1.2766)  val_loss=2.8375 (jet=0.5683 origin=1.0595 vtx=1.2097)  val_acc=0.7840  origin_acc=0.6859  lxy_scale=[b_vertex=1.027, c_vertex=1.668]  dz_scale=[b_vertex=19.957, c_vertex=21.273]
Epoch 97/100  loss=3.0573 (jet=0.6624 origin=1.1291 vtx=1.2659)  val_loss=2.8142 (jet=0.5714 origin=1.0735 vtx=1.1694)  val_acc=0.7816  origin_acc=0.6735  lxy_scale=[b_vertex=1.025, c_vertex=1.641]  dz_scale=[b_vertex=20.153, c_vertex=21.696]
Epoch 98/100  loss=3.0567 (jet=0.6630 origin=1.1283 vtx=1.2654)  val_loss=2.8679 (jet=0.6112 origin=1.0692 vtx=1.1874)  val_acc=0.7537  origin_acc=0.6678  lxy_scale=[b_vertex=1.024, c_vertex=1.643]  dz_scale=[b_vertex=20.279, c_vertex=21.760]
Epoch 99/100  loss=3.0545 (jet=0.6621 origin=1.1292 vtx=1.2632)  val_loss=2.8646 (jet=0.5929 origin=1.0612 vtx=1.2105)  val_acc=0.7685  origin_acc=0.6731  lxy_scale=[b_vertex=1.018, c_vertex=1.648]  dz_scale=[b_vertex=20.342, c_vertex=22.051]
Epoch 100/100  loss=3.0605 (jet=0.6629 origin=1.1289 vtx=1.2686)  val_loss=2.8303 (jet=0.5681 origin=1.0671 vtx=1.1951)  val_acc=0.7828  origin_acc=0.6491  lxy_scale=[b_vertex=1.027, c_vertex=1.669]  dz_scale=[b_vertex=20.232, c_vertex=21.521]
Saved staged_origin_vertex_jet.pt

Jet classification report:
              precision    recall  f1-score   support

       b-jet       0.95      0.74      0.83     77645
       c-jet       0.25      0.51      0.34     18430
   light-jet       0.88      0.86      0.87    103820

    accuracy                           0.78    199895
   macro avg       0.69      0.70      0.68    199895
weighted avg       0.85      0.78      0.81    199895

Jet confusion matrix (rows=true, cols=pred):
[[57560 14668  5417]
 [ 2179  9361  6890]
 [ 1168 13093 89559]]

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

Saved training_summary.png  training_summary_log.png
Saved origin_confusion_matrix.png
Saved vertex_fit_b_vertex.png
Saved vertex_fit_b_vertex_dz.png
Saved vertex_fit_c_vertex.png
Saved vertex_fit_c_vertex_dz.png
Saved output_probs.png
Saved discriminant.png
Saved roc.png

=== b-tagging rejection rates ===
  ε_b=65%:  1/ε_c = 18  1/ε_light = 364
  ε_b=70%:  1/ε_c = 12  1/ε_light = 212
  ε_b=77%:  1/ε_c = 7  1/ε_light = 96
  ε_b=85%:  1/ε_c = 3  1/ε_light = 35
  ε_b=90%:  1/ε_c = 2  1/ε_light = 15
Saved rejection.png
Saved c_discriminant.png
Saved c_roc.png

=== c-tagging rejection rates ===
  ε_c=20%:  1/ε_b = 21  1/ε_light = 74
  ε_c=30%:  1/ε_b = 12  1/ε_light = 31
  ε_c=40%:  1/ε_b = 7  1/ε_light = 16
Saved c_rejection.png

=== Track-to-vertex assignment efficiency ===
  b_vertex (thr>0.5): assignment=0.000  false-positive=0.0000  n_match=335470
  b_vertex (thr>0.8): assignment=0.000  false-positive=0.0000  n_match=335470
Saved track_vtx_assignment_b_vertex.png
  c_vertex (thr>0.5): assignment=0.000  false-positive=0.0000  n_match=41464
  c_vertex (thr>0.8): assignment=0.000  false-positive=0.0000  n_match=41464
Saved track_vtx_assignment_c_vertex.png

All outputs saved to ./results/results_staged_20260707_010506/
