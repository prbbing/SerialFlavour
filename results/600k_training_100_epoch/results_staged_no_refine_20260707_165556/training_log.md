Device: ['0']  |  DataParallel: False
Config saved to results/600k_training_100_epoch/results_staged_no_refine_20260707_165556/config.json
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

Epoch 01/100  loss=4.5189 (jet=0.7943 origin=1.5009 vtx=2.2237)  val_loss=4.1033 (jet=0.6534 origin=1.2759 vtx=2.1740)  val_acc=0.7582  origin_acc=0.6741  lxy_scale=[b_vertex=1.026, c_vertex=1.230]  dz_scale=[b_vertex=1.091, c_vertex=0.924]
Epoch 02/100  loss=4.2130 (jet=0.7436 origin=1.2992 vtx=2.1702)  val_loss=4.0283 (jet=0.6625 origin=1.2160 vtx=2.1497)  val_acc=0.7377  origin_acc=0.6856  lxy_scale=[b_vertex=0.985, c_vertex=1.376]  dz_scale=[b_vertex=1.328, c_vertex=0.932]
Epoch 03/100  loss=4.1392 (jet=0.7301 origin=1.2546 vtx=2.1545)  val_loss=3.9086 (jet=0.6174 origin=1.1584 vtx=2.1327)  val_acc=0.7732  origin_acc=0.6802  lxy_scale=[b_vertex=0.956, c_vertex=1.416]  dz_scale=[b_vertex=1.584, c_vertex=0.937]
Epoch 04/100  loss=4.0889 (jet=0.7206 origin=1.2311 vtx=2.1371)  val_loss=3.9220 (jet=0.6462 origin=1.1555 vtx=2.1203)  val_acc=0.7490  origin_acc=0.6671  lxy_scale=[b_vertex=0.936, c_vertex=1.386]  dz_scale=[b_vertex=1.815, c_vertex=0.937]
Epoch 05/100  loss=4.0640 (jet=0.7150 origin=1.2199 vtx=2.1292)  val_loss=3.8897 (jet=0.6143 origin=1.1599 vtx=2.1155)  val_acc=0.7694  origin_acc=0.6466  lxy_scale=[b_vertex=0.924, c_vertex=1.350]  dz_scale=[b_vertex=2.017, c_vertex=0.939]
Epoch 06/100  loss=4.0502 (jet=0.7132 origin=1.2112 vtx=2.1258)  val_loss=3.8859 (jet=0.6348 origin=1.1377 vtx=2.1134)  val_acc=0.7556  origin_acc=0.6714  lxy_scale=[b_vertex=0.923, c_vertex=1.337]  dz_scale=[b_vertex=2.171, c_vertex=0.950]
Epoch 07/100  loss=4.0381 (jet=0.7107 origin=1.2055 vtx=2.1220)  val_loss=3.9113 (jet=0.6559 origin=1.1476 vtx=2.1078)  val_acc=0.7285  origin_acc=0.6285  lxy_scale=[b_vertex=0.915, c_vertex=1.331]  dz_scale=[b_vertex=2.296, c_vertex=0.953]
Epoch 08/100  loss=4.0274 (jet=0.7077 origin=1.2003 vtx=2.1193)  val_loss=3.8941 (jet=0.6490 origin=1.1393 vtx=2.1058)  val_acc=0.7389  origin_acc=0.6531  lxy_scale=[b_vertex=0.914, c_vertex=1.319]  dz_scale=[b_vertex=2.377, c_vertex=0.964]
Epoch 09/100  loss=4.0152 (jet=0.7054 origin=1.1934 vtx=2.1164)  val_loss=3.8397 (jet=0.5973 origin=1.1394 vtx=2.1030)  val_acc=0.7764  origin_acc=0.6921  lxy_scale=[b_vertex=0.909, c_vertex=1.308]  dz_scale=[b_vertex=2.428, c_vertex=0.974]
Epoch 10/100  loss=4.0067 (jet=0.7034 origin=1.1893 vtx=2.1139)  val_loss=3.8140 (jet=0.5925 origin=1.1190 vtx=2.1025)  val_acc=0.7791  origin_acc=0.6498  lxy_scale=[b_vertex=0.909, c_vertex=1.300]  dz_scale=[b_vertex=2.481, c_vertex=0.978]
Epoch 11/100  loss=3.9982 (jet=0.7013 origin=1.1856 vtx=2.1113)  val_loss=3.8136 (jet=0.5928 origin=1.1195 vtx=2.1013)  val_acc=0.7788  origin_acc=0.6619  lxy_scale=[b_vertex=0.901, c_vertex=1.288]  dz_scale=[b_vertex=2.517, c_vertex=0.986]
Epoch 12/100  loss=3.9917 (jet=0.6992 origin=1.1821 vtx=2.1104)  val_loss=3.8283 (jet=0.6194 origin=1.1085 vtx=2.1003)  val_acc=0.7605  origin_acc=0.6709  lxy_scale=[b_vertex=0.904, c_vertex=1.290]  dz_scale=[b_vertex=2.541, c_vertex=0.996]
Epoch 13/100  loss=3.9870 (jet=0.6982 origin=1.1796 vtx=2.1092)  val_loss=3.8253 (jet=0.6066 origin=1.1214 vtx=2.0973)  val_acc=0.7678  origin_acc=0.6768  lxy_scale=[b_vertex=0.900, c_vertex=1.281]  dz_scale=[b_vertex=2.568, c_vertex=1.016]
Epoch 14/100  loss=3.9818 (jet=0.6973 origin=1.1770 vtx=2.1075)  val_loss=3.8381 (jet=0.6183 origin=1.1224 vtx=2.0973)  val_acc=0.7604  origin_acc=0.6764  lxy_scale=[b_vertex=0.902, c_vertex=1.283]  dz_scale=[b_vertex=2.585, c_vertex=1.033]
Epoch 15/100  loss=3.9761 (jet=0.6950 origin=1.1745 vtx=2.1067)  val_loss=3.8193 (jet=0.6071 origin=1.1159 vtx=2.0964)  val_acc=0.7697  origin_acc=0.6812  lxy_scale=[b_vertex=0.894, c_vertex=1.281]  dz_scale=[b_vertex=2.586, c_vertex=1.029]
Epoch 16/100  loss=3.9717 (jet=0.6930 origin=1.1725 vtx=2.1061)  val_loss=3.8205 (jet=0.6109 origin=1.1139 vtx=2.0957)  val_acc=0.7668  origin_acc=0.6391  lxy_scale=[b_vertex=0.897, c_vertex=1.289]  dz_scale=[b_vertex=2.590, c_vertex=1.037]
Epoch 17/100  loss=3.9679 (jet=0.6927 origin=1.1703 vtx=2.1049)  val_loss=3.8415 (jet=0.6373 origin=1.1098 vtx=2.0944)  val_acc=0.7444  origin_acc=0.6610  lxy_scale=[b_vertex=0.898, c_vertex=1.280]  dz_scale=[b_vertex=2.599, c_vertex=1.033]
Epoch 18/100  loss=3.9659 (jet=0.6920 origin=1.1686 vtx=2.1053)  val_loss=3.8047 (jet=0.6082 origin=1.1037 vtx=2.0928)  val_acc=0.7634  origin_acc=0.6567  lxy_scale=[b_vertex=0.898, c_vertex=1.274]  dz_scale=[b_vertex=2.612, c_vertex=1.027]
Epoch 19/100  loss=3.9600 (jet=0.6904 origin=1.1648 vtx=2.1048)  val_loss=3.8149 (jet=0.6305 origin=1.0914 vtx=2.0930)  val_acc=0.7518  origin_acc=0.6668  lxy_scale=[b_vertex=0.903, c_vertex=1.280]  dz_scale=[b_vertex=2.617, c_vertex=1.022]
Epoch 20/100  loss=3.9557 (jet=0.6890 origin=1.1625 vtx=2.1042)  val_loss=3.7642 (jet=0.5744 origin=1.0973 vtx=2.0925)  val_acc=0.7875  origin_acc=0.6850  lxy_scale=[b_vertex=0.901, c_vertex=1.277]  dz_scale=[b_vertex=2.645, c_vertex=1.014]
Epoch 21/100  loss=3.9512 (jet=0.6873 origin=1.1606 vtx=2.1033)  val_loss=3.7871 (jet=0.6027 origin=1.0903 vtx=2.0941)  val_acc=0.7686  origin_acc=0.6831  lxy_scale=[b_vertex=0.899, c_vertex=1.269]  dz_scale=[b_vertex=2.646, c_vertex=1.003]
Epoch 22/100  loss=3.9476 (jet=0.6867 origin=1.1584 vtx=2.1025)  val_loss=3.7804 (jet=0.5868 origin=1.1017 vtx=2.0918)  val_acc=0.7843  origin_acc=0.6706  lxy_scale=[b_vertex=0.899, c_vertex=1.280]  dz_scale=[b_vertex=2.667, c_vertex=0.993]
Epoch 23/100  loss=3.9462 (jet=0.6868 origin=1.1566 vtx=2.1027)  val_loss=3.7984 (jet=0.6055 origin=1.1017 vtx=2.0913)  val_acc=0.7692  origin_acc=0.6701  lxy_scale=[b_vertex=0.897, c_vertex=1.276]  dz_scale=[b_vertex=2.662, c_vertex=0.990]
Epoch 24/100  loss=3.9419 (jet=0.6851 origin=1.1552 vtx=2.1015)  val_loss=3.7889 (jet=0.6044 origin=1.0957 vtx=2.0888)  val_acc=0.7647  origin_acc=0.6836  lxy_scale=[b_vertex=0.896, c_vertex=1.274]  dz_scale=[b_vertex=2.680, c_vertex=0.978]
Epoch 25/100  loss=3.9382 (jet=0.6835 origin=1.1533 vtx=2.1013)  val_loss=3.7711 (jet=0.5994 origin=1.0795 vtx=2.0922)  val_acc=0.7678  origin_acc=0.6771  lxy_scale=[b_vertex=0.894, c_vertex=1.268]  dz_scale=[b_vertex=2.675, c_vertex=0.969]
Epoch 26/100  loss=3.9360 (jet=0.6829 origin=1.1523 vtx=2.1008)  val_loss=3.7561 (jet=0.5797 origin=1.0888 vtx=2.0877)  val_acc=0.7803  origin_acc=0.6733  lxy_scale=[b_vertex=0.898, c_vertex=1.280]  dz_scale=[b_vertex=2.686, c_vertex=0.958]
Epoch 27/100  loss=3.9347 (jet=0.6822 origin=1.1520 vtx=2.1004)  val_loss=3.7683 (jet=0.5877 origin=1.0909 vtx=2.0897)  val_acc=0.7763  origin_acc=0.6482  lxy_scale=[b_vertex=0.897, c_vertex=1.273]  dz_scale=[b_vertex=2.690, c_vertex=0.951]
Epoch 28/100  loss=3.9316 (jet=0.6814 origin=1.1508 vtx=2.0993)  val_loss=3.7923 (jet=0.6189 origin=1.0844 vtx=2.0889)  val_acc=0.7536  origin_acc=0.6614  lxy_scale=[b_vertex=0.898, c_vertex=1.277]  dz_scale=[b_vertex=2.700, c_vertex=0.943]
Epoch 29/100  loss=3.9316 (jet=0.6813 origin=1.1507 vtx=2.0997)  val_loss=3.7326 (jet=0.5607 origin=1.0854 vtx=2.0864)  val_acc=0.7943  origin_acc=0.6575  lxy_scale=[b_vertex=0.892, c_vertex=1.278]  dz_scale=[b_vertex=2.711, c_vertex=0.952]
Epoch 30/100  loss=3.9293 (jet=0.6803 origin=1.1501 vtx=2.0990)  val_loss=3.7657 (jet=0.5846 origin=1.0893 vtx=2.0918)  val_acc=0.7784  origin_acc=0.6723  lxy_scale=[b_vertex=0.899, c_vertex=1.278]  dz_scale=[b_vertex=2.711, c_vertex=0.949]
Epoch 31/100  loss=3.9307 (jet=0.6809 origin=1.1506 vtx=2.0992)  val_loss=3.7826 (jet=0.6073 origin=1.0893 vtx=2.0861)  val_acc=0.7615  origin_acc=0.6552  lxy_scale=[b_vertex=0.899, c_vertex=1.272]  dz_scale=[b_vertex=2.730, c_vertex=0.950]
Epoch 32/100  loss=3.9229 (jet=0.6784 origin=1.1467 vtx=2.0978)  val_loss=3.7589 (jet=0.6012 origin=1.0687 vtx=2.0890)  val_acc=0.7677  origin_acc=0.6603  lxy_scale=[b_vertex=0.900, c_vertex=1.275]  dz_scale=[b_vertex=2.736, c_vertex=0.946]
Epoch 33/100  loss=3.9229 (jet=0.6788 origin=1.1474 vtx=2.0966)  val_loss=3.7808 (jet=0.6119 origin=1.0837 vtx=2.0852)  val_acc=0.7605  origin_acc=0.6610  lxy_scale=[b_vertex=0.892, c_vertex=1.275]  dz_scale=[b_vertex=2.767, c_vertex=0.949]
Epoch 34/100  loss=3.9214 (jet=0.6778 origin=1.1464 vtx=2.0972)  val_loss=3.7687 (jet=0.6048 origin=1.0796 vtx=2.0843)  val_acc=0.7659  origin_acc=0.6834  lxy_scale=[b_vertex=0.898, c_vertex=1.275]  dz_scale=[b_vertex=2.762, c_vertex=0.951]
Epoch 35/100  loss=3.9182 (jet=0.6773 origin=1.1447 vtx=2.0961)  val_loss=3.7881 (jet=0.6110 origin=1.0937 vtx=2.0834)  val_acc=0.7596  origin_acc=0.6574  lxy_scale=[b_vertex=0.892, c_vertex=1.277]  dz_scale=[b_vertex=2.798, c_vertex=0.950]
Epoch 36/100  loss=3.9187 (jet=0.6773 origin=1.1457 vtx=2.0957)  val_loss=3.7408 (jet=0.5777 origin=1.0791 vtx=2.0840)  val_acc=0.7804  origin_acc=0.6650  lxy_scale=[b_vertex=0.907, c_vertex=1.273]  dz_scale=[b_vertex=2.783, c_vertex=0.948]
Epoch 37/100  loss=3.9207 (jet=0.6771 origin=1.1467 vtx=2.0969)  val_loss=3.7723 (jet=0.5987 origin=1.0857 vtx=2.0879)  val_acc=0.7705  origin_acc=0.6789  lxy_scale=[b_vertex=0.900, c_vertex=1.274]  dz_scale=[b_vertex=2.768, c_vertex=0.960]
Epoch 38/100  loss=3.9159 (jet=0.6764 origin=1.1438 vtx=2.0957)  val_loss=3.7660 (jet=0.5934 origin=1.0852 vtx=2.0874)  val_acc=0.7732  origin_acc=0.6575  lxy_scale=[b_vertex=0.898, c_vertex=1.271]  dz_scale=[b_vertex=2.766, c_vertex=0.958]
Epoch 39/100  loss=3.9154 (jet=0.6768 origin=1.1436 vtx=2.0950)  val_loss=3.7383 (jet=0.5760 origin=1.0776 vtx=2.0847)  val_acc=0.7829  origin_acc=0.6791  lxy_scale=[b_vertex=0.902, c_vertex=1.273]  dz_scale=[b_vertex=2.786, c_vertex=0.950]
Epoch 40/100  loss=3.9141 (jet=0.6759 origin=1.1427 vtx=2.0955)  val_loss=3.7549 (jet=0.5913 origin=1.0792 vtx=2.0843)  val_acc=0.7696  origin_acc=0.6592  lxy_scale=[b_vertex=0.906, c_vertex=1.268]  dz_scale=[b_vertex=2.771, c_vertex=0.956]
Epoch 41/100  loss=3.9109 (jet=0.6749 origin=1.1420 vtx=2.0939)  val_loss=3.7675 (jet=0.5944 origin=1.0894 vtx=2.0837)  val_acc=0.7731  origin_acc=0.6733  lxy_scale=[b_vertex=0.902, c_vertex=1.264]  dz_scale=[b_vertex=2.775, c_vertex=0.954]
Epoch 42/100  loss=3.9122 (jet=0.6746 origin=1.1421 vtx=2.0956)  val_loss=3.7861 (jet=0.6179 origin=1.0854 vtx=2.0828)  val_acc=0.7556  origin_acc=0.6764  lxy_scale=[b_vertex=0.900, c_vertex=1.267]  dz_scale=[b_vertex=2.776, c_vertex=0.960]
Epoch 43/100  loss=3.9105 (jet=0.6741 origin=1.1417 vtx=2.0946)  val_loss=3.7615 (jet=0.5911 origin=1.0836 vtx=2.0869)  val_acc=0.7719  origin_acc=0.6730  lxy_scale=[b_vertex=0.900, c_vertex=1.266]  dz_scale=[b_vertex=2.781, c_vertex=0.949]
Epoch 44/100  loss=3.9102 (jet=0.6750 origin=1.1412 vtx=2.0940)  val_loss=3.7236 (jet=0.5581 origin=1.0818 vtx=2.0836)  val_acc=0.7938  origin_acc=0.6601  lxy_scale=[b_vertex=0.893, c_vertex=1.269]  dz_scale=[b_vertex=2.802, c_vertex=0.958]
Epoch 45/100  loss=3.9078 (jet=0.6740 origin=1.1399 vtx=2.0939)  val_loss=3.7603 (jet=0.6025 origin=1.0735 vtx=2.0843)  val_acc=0.7624  origin_acc=0.6689  lxy_scale=[b_vertex=0.904, c_vertex=1.265]  dz_scale=[b_vertex=2.787, c_vertex=0.960]
Epoch 46/100  loss=3.9067 (jet=0.6732 origin=1.1403 vtx=2.0932)  val_loss=3.7192 (jet=0.5656 origin=1.0719 vtx=2.0818)  val_acc=0.7867  origin_acc=0.6658  lxy_scale=[b_vertex=0.902, c_vertex=1.274]  dz_scale=[b_vertex=2.796, c_vertex=0.967]
Epoch 47/100  loss=3.9040 (jet=0.6726 origin=1.1393 vtx=2.0922)  val_loss=3.7197 (jet=0.5734 origin=1.0654 vtx=2.0809)  val_acc=0.7828  origin_acc=0.6884  lxy_scale=[b_vertex=0.899, c_vertex=1.266]  dz_scale=[b_vertex=2.816, c_vertex=0.969]
Epoch 48/100  loss=3.9046 (jet=0.6726 origin=1.1386 vtx=2.0933)  val_loss=3.7371 (jet=0.5844 origin=1.0703 vtx=2.0824)  val_acc=0.7759  origin_acc=0.6895  lxy_scale=[b_vertex=0.899, c_vertex=1.262]  dz_scale=[b_vertex=2.805, c_vertex=0.974]
Epoch 49/100  loss=3.9048 (jet=0.6726 origin=1.1398 vtx=2.0923)  val_loss=3.7255 (jet=0.5752 origin=1.0701 vtx=2.0802)  val_acc=0.7789  origin_acc=0.6574  lxy_scale=[b_vertex=0.903, c_vertex=1.272]  dz_scale=[b_vertex=2.832, c_vertex=0.974]
Epoch 50/100  loss=3.9016 (jet=0.6726 origin=1.1377 vtx=2.0913)  val_loss=3.7075 (jet=0.5638 origin=1.0633 vtx=2.0803)  val_acc=0.7867  origin_acc=0.6744  lxy_scale=[b_vertex=0.896, c_vertex=1.271]  dz_scale=[b_vertex=2.813, c_vertex=0.977]
Epoch 51/100  loss=3.8998 (jet=0.6720 origin=1.1371 vtx=2.0907)  val_loss=3.7277 (jet=0.5736 origin=1.0718 vtx=2.0823)  val_acc=0.7835  origin_acc=0.6747  lxy_scale=[b_vertex=0.895, c_vertex=1.270]  dz_scale=[b_vertex=2.827, c_vertex=0.967]
Epoch 52/100  loss=3.9007 (jet=0.6716 origin=1.1382 vtx=2.0909)  val_loss=3.7609 (jet=0.5981 origin=1.0821 vtx=2.0806)  val_acc=0.7685  origin_acc=0.6699  lxy_scale=[b_vertex=0.904, c_vertex=1.280]  dz_scale=[b_vertex=2.845, c_vertex=0.970]
Epoch 53/100  loss=3.8993 (jet=0.6708 origin=1.1369 vtx=2.0916)  val_loss=3.7595 (jet=0.6024 origin=1.0742 vtx=2.0828)  val_acc=0.7638  origin_acc=0.6570  lxy_scale=[b_vertex=0.906, c_vertex=1.277]  dz_scale=[b_vertex=2.852, c_vertex=0.974]
Epoch 54/100  loss=3.9010 (jet=0.6722 origin=1.1369 vtx=2.0919)  val_loss=3.7446 (jet=0.5942 origin=1.0676 vtx=2.0828)  val_acc=0.7659  origin_acc=0.6583  lxy_scale=[b_vertex=0.902, c_vertex=1.274]  dz_scale=[b_vertex=2.845, c_vertex=0.973]
Epoch 55/100  loss=3.8990 (jet=0.6710 origin=1.1367 vtx=2.0913)  val_loss=3.7336 (jet=0.5794 origin=1.0746 vtx=2.0796)  val_acc=0.7765  origin_acc=0.6662  lxy_scale=[b_vertex=0.903, c_vertex=1.259]  dz_scale=[b_vertex=2.855, c_vertex=0.976]
Epoch 56/100  loss=3.8966 (jet=0.6705 origin=1.1358 vtx=2.0903)  val_loss=3.7461 (jet=0.5924 origin=1.0748 vtx=2.0789)  val_acc=0.7698  origin_acc=0.6615  lxy_scale=[b_vertex=0.906, c_vertex=1.278]  dz_scale=[b_vertex=2.862, c_vertex=0.975]
Epoch 57/100  loss=3.8966 (jet=0.6704 origin=1.1362 vtx=2.0901)  val_loss=3.7310 (jet=0.5789 origin=1.0726 vtx=2.0794)  val_acc=0.7751  origin_acc=0.6627  lxy_scale=[b_vertex=0.899, c_vertex=1.274]  dz_scale=[b_vertex=2.855, c_vertex=0.980]
Epoch 58/100  loss=3.8949 (jet=0.6701 origin=1.1351 vtx=2.0897)  val_loss=3.7355 (jet=0.5854 origin=1.0707 vtx=2.0793)  val_acc=0.7746  origin_acc=0.6792  lxy_scale=[b_vertex=0.901, c_vertex=1.272]  dz_scale=[b_vertex=2.863, c_vertex=0.975]
Epoch 59/100  loss=3.8936 (jet=0.6691 origin=1.1346 vtx=2.0899)  val_loss=3.7222 (jet=0.5821 origin=1.0596 vtx=2.0805)  val_acc=0.7786  origin_acc=0.6852  lxy_scale=[b_vertex=0.903, c_vertex=1.267]  dz_scale=[b_vertex=2.865, c_vertex=0.972]
Epoch 60/100  loss=3.8927 (jet=0.6692 origin=1.1341 vtx=2.0894)  val_loss=3.7538 (jet=0.6001 origin=1.0754 vtx=2.0784)  val_acc=0.7649  origin_acc=0.6709  lxy_scale=[b_vertex=0.898, c_vertex=1.262]  dz_scale=[b_vertex=2.854, c_vertex=0.974]
Epoch 61/100  loss=3.8923 (jet=0.6692 origin=1.1336 vtx=2.0894)  val_loss=3.7391 (jet=0.5823 origin=1.0783 vtx=2.0786)  val_acc=0.7799  origin_acc=0.6671  lxy_scale=[b_vertex=0.903, c_vertex=1.257]  dz_scale=[b_vertex=2.863, c_vertex=0.981]
Epoch 62/100  loss=3.8917 (jet=0.6689 origin=1.1339 vtx=2.0888)  val_loss=3.7466 (jet=0.5993 origin=1.0686 vtx=2.0787)  val_acc=0.7648  origin_acc=0.6670  lxy_scale=[b_vertex=0.908, c_vertex=1.271]  dz_scale=[b_vertex=2.876, c_vertex=0.977]
Epoch 63/100  loss=3.8918 (jet=0.6692 origin=1.1339 vtx=2.0886)  val_loss=3.7392 (jet=0.5861 origin=1.0733 vtx=2.0797)  val_acc=0.7729  origin_acc=0.6717  lxy_scale=[b_vertex=0.906, c_vertex=1.269]  dz_scale=[b_vertex=2.872, c_vertex=0.975]
Epoch 64/100  loss=3.8895 (jet=0.6688 origin=1.1331 vtx=2.0876)  val_loss=3.7640 (jet=0.6071 origin=1.0794 vtx=2.0775)  val_acc=0.7573  origin_acc=0.6446  lxy_scale=[b_vertex=0.902, c_vertex=1.268]  dz_scale=[b_vertex=2.895, c_vertex=0.976]
Epoch 65/100  loss=3.8918 (jet=0.6693 origin=1.1331 vtx=2.0894)  val_loss=3.7414 (jet=0.5827 origin=1.0768 vtx=2.0819)  val_acc=0.7789  origin_acc=0.6756  lxy_scale=[b_vertex=0.904, c_vertex=1.272]  dz_scale=[b_vertex=2.900, c_vertex=0.978]
Epoch 66/100  loss=3.8928 (jet=0.6685 origin=1.1350 vtx=2.0893)  val_loss=3.7322 (jet=0.5890 origin=1.0654 vtx=2.0777)  val_acc=0.7702  origin_acc=0.6607  lxy_scale=[b_vertex=0.905, c_vertex=1.270]  dz_scale=[b_vertex=2.886, c_vertex=0.976]
Epoch 67/100  loss=3.8888 (jet=0.6676 origin=1.1327 vtx=2.0886)  val_loss=3.7125 (jet=0.5640 origin=1.0718 vtx=2.0767)  val_acc=0.7870  origin_acc=0.6722  lxy_scale=[b_vertex=0.899, c_vertex=1.261]  dz_scale=[b_vertex=2.910, c_vertex=0.989]
Epoch 68/100  loss=3.8875 (jet=0.6674 origin=1.1320 vtx=2.0881)  val_loss=3.7308 (jet=0.5846 origin=1.0682 vtx=2.0780)  val_acc=0.7739  origin_acc=0.6833  lxy_scale=[b_vertex=0.905, c_vertex=1.261]  dz_scale=[b_vertex=2.904, c_vertex=0.995]
Epoch 69/100  loss=3.8881 (jet=0.6677 origin=1.1315 vtx=2.0890)  val_loss=3.7459 (jet=0.5967 origin=1.0742 vtx=2.0750)  val_acc=0.7638  origin_acc=0.6679  lxy_scale=[b_vertex=0.900, c_vertex=1.270]  dz_scale=[b_vertex=2.898, c_vertex=0.990]
Epoch 70/100  loss=3.8855 (jet=0.6671 origin=1.1308 vtx=2.0877)  val_loss=3.7244 (jet=0.5814 origin=1.0621 vtx=2.0809)  val_acc=0.7741  origin_acc=0.6801  lxy_scale=[b_vertex=0.900, c_vertex=1.266]  dz_scale=[b_vertex=2.925, c_vertex=0.995]
Epoch 71/100  loss=3.8869 (jet=0.6674 origin=1.1314 vtx=2.0881)  val_loss=3.7513 (jet=0.6048 origin=1.0708 vtx=2.0757)  val_acc=0.7576  origin_acc=0.6592  lxy_scale=[b_vertex=0.907, c_vertex=1.267]  dz_scale=[b_vertex=2.932, c_vertex=0.986]
Epoch 72/100  loss=3.8844 (jet=0.6665 origin=1.1298 vtx=2.0881)  val_loss=3.7448 (jet=0.5955 origin=1.0740 vtx=2.0753)  val_acc=0.7676  origin_acc=0.6688  lxy_scale=[b_vertex=0.901, c_vertex=1.265]  dz_scale=[b_vertex=2.907, c_vertex=0.986]
Epoch 73/100  loss=3.8821 (jet=0.6659 origin=1.1295 vtx=2.0866)  val_loss=3.7619 (jet=0.6083 origin=1.0742 vtx=2.0794)  val_acc=0.7578  origin_acc=0.6626  lxy_scale=[b_vertex=0.905, c_vertex=1.276]  dz_scale=[b_vertex=2.932, c_vertex=0.994]
Epoch 74/100  loss=3.8841 (jet=0.6666 origin=1.1298 vtx=2.0877)  val_loss=3.7290 (jet=0.5779 origin=1.0766 vtx=2.0745)  val_acc=0.7763  origin_acc=0.6625  lxy_scale=[b_vertex=0.905, c_vertex=1.267]  dz_scale=[b_vertex=2.947, c_vertex=1.002]
Epoch 75/100  loss=3.8846 (jet=0.6664 origin=1.1304 vtx=2.0877)  val_loss=3.7300 (jet=0.5883 origin=1.0624 vtx=2.0793)  val_acc=0.7716  origin_acc=0.6807  lxy_scale=[b_vertex=0.908, c_vertex=1.265]  dz_scale=[b_vertex=2.954, c_vertex=1.006]
Epoch 76/100  loss=3.8813 (jet=0.6657 origin=1.1287 vtx=2.0869)  val_loss=3.7116 (jet=0.5664 origin=1.0714 vtx=2.0739)  val_acc=0.7809  origin_acc=0.6721  lxy_scale=[b_vertex=0.904, c_vertex=1.259]  dz_scale=[b_vertex=2.935, c_vertex=1.008]
Epoch 77/100  loss=3.8812 (jet=0.6657 origin=1.1288 vtx=2.0868)  val_loss=3.7369 (jet=0.5964 origin=1.0619 vtx=2.0786)  val_acc=0.7661  origin_acc=0.6614  lxy_scale=[b_vertex=0.900, c_vertex=1.265]  dz_scale=[b_vertex=2.933, c_vertex=1.002]
Epoch 78/100  loss=3.8810 (jet=0.6656 origin=1.1293 vtx=2.0861)  val_loss=3.7235 (jet=0.5820 origin=1.0650 vtx=2.0765)  val_acc=0.7718  origin_acc=0.6754  lxy_scale=[b_vertex=0.907, c_vertex=1.269]  dz_scale=[b_vertex=2.948, c_vertex=1.001]
Epoch 79/100  loss=3.8792 (jet=0.6653 origin=1.1284 vtx=2.0855)  val_loss=3.7363 (jet=0.6014 origin=1.0588 vtx=2.0761)  val_acc=0.7618  origin_acc=0.6651  lxy_scale=[b_vertex=0.904, c_vertex=1.266]  dz_scale=[b_vertex=2.974, c_vertex=1.008]
Epoch 80/100  loss=3.8800 (jet=0.6649 origin=1.1284 vtx=2.0867)  val_loss=3.7551 (jet=0.6085 origin=1.0710 vtx=2.0756)  val_acc=0.7548  origin_acc=0.6655  lxy_scale=[b_vertex=0.899, c_vertex=1.269]  dz_scale=[b_vertex=2.972, c_vertex=1.020]
Epoch 81/100  loss=3.8790 (jet=0.6651 origin=1.1281 vtx=2.0858)  val_loss=3.6972 (jet=0.5623 origin=1.0578 vtx=2.0772)  val_acc=0.7887  origin_acc=0.6804  lxy_scale=[b_vertex=0.900, c_vertex=1.274]  dz_scale=[b_vertex=2.979, c_vertex=1.021]
Epoch 82/100  loss=3.8789 (jet=0.6645 origin=1.1288 vtx=2.0856)  val_loss=3.7324 (jet=0.5899 origin=1.0655 vtx=2.0771)  val_acc=0.7696  origin_acc=0.6710  lxy_scale=[b_vertex=0.903, c_vertex=1.277]  dz_scale=[b_vertex=3.012, c_vertex=1.021]
Epoch 83/100  loss=3.8770 (jet=0.6643 origin=1.1272 vtx=2.0855)  val_loss=3.7149 (jet=0.5781 origin=1.0599 vtx=2.0769)  val_acc=0.7776  origin_acc=0.6842  lxy_scale=[b_vertex=0.907, c_vertex=1.261]  dz_scale=[b_vertex=3.000, c_vertex=1.030]
Epoch 84/100  loss=3.8781 (jet=0.6649 origin=1.1291 vtx=2.0841)  val_loss=3.7102 (jet=0.5811 origin=1.0539 vtx=2.0752)  val_acc=0.7771  origin_acc=0.6925  lxy_scale=[b_vertex=0.903, c_vertex=1.266]  dz_scale=[b_vertex=3.018, c_vertex=1.034]
Epoch 85/100  loss=3.8769 (jet=0.6647 origin=1.1269 vtx=2.0854)  val_loss=3.7199 (jet=0.5762 origin=1.0694 vtx=2.0743)  val_acc=0.7800  origin_acc=0.6606  lxy_scale=[b_vertex=0.900, c_vertex=1.267]  dz_scale=[b_vertex=3.038, c_vertex=1.029]
Epoch 86/100  loss=3.8762 (jet=0.6633 origin=1.1276 vtx=2.0854)  val_loss=3.7277 (jet=0.5851 origin=1.0653 vtx=2.0773)  val_acc=0.7708  origin_acc=0.6678  lxy_scale=[b_vertex=0.902, c_vertex=1.273]  dz_scale=[b_vertex=3.041, c_vertex=1.026]
Epoch 87/100  loss=3.8761 (jet=0.6644 origin=1.1269 vtx=2.0848)  val_loss=3.7139 (jet=0.5714 origin=1.0698 vtx=2.0727)  val_acc=0.7820  origin_acc=0.6596  lxy_scale=[b_vertex=0.908, c_vertex=1.262]  dz_scale=[b_vertex=3.062, c_vertex=1.028]
Epoch 88/100  loss=3.8740 (jet=0.6634 origin=1.1263 vtx=2.0844)  val_loss=3.7175 (jet=0.5659 origin=1.0818 vtx=2.0698)  val_acc=0.7835  origin_acc=0.6465  lxy_scale=[b_vertex=0.903, c_vertex=1.266]  dz_scale=[b_vertex=3.080, c_vertex=1.032]
Epoch 89/100  loss=3.8749 (jet=0.6639 origin=1.1273 vtx=2.0837)  val_loss=3.7405 (jet=0.6038 origin=1.0601 vtx=2.0766)  val_acc=0.7581  origin_acc=0.6714  lxy_scale=[b_vertex=0.911, c_vertex=1.268]  dz_scale=[b_vertex=3.106, c_vertex=1.036]
Epoch 90/100  loss=3.8750 (jet=0.6635 origin=1.1276 vtx=2.0840)  val_loss=3.7015 (jet=0.5689 origin=1.0570 vtx=2.0756)  val_acc=0.7835  origin_acc=0.6721  lxy_scale=[b_vertex=0.908, c_vertex=1.280]  dz_scale=[b_vertex=3.114, c_vertex=1.040]
Epoch 91/100  loss=3.8736 (jet=0.6632 origin=1.1273 vtx=2.0831)  val_loss=3.7199 (jet=0.5816 origin=1.0620 vtx=2.0762)  val_acc=0.7744  origin_acc=0.6763  lxy_scale=[b_vertex=0.901, c_vertex=1.269]  dz_scale=[b_vertex=3.154, c_vertex=1.051]
Epoch 92/100  loss=3.8729 (jet=0.6629 origin=1.1275 vtx=2.0826)  val_loss=3.7214 (jet=0.5810 origin=1.0666 vtx=2.0738)  val_acc=0.7727  origin_acc=0.6578  lxy_scale=[b_vertex=0.906, c_vertex=1.267]  dz_scale=[b_vertex=3.167, c_vertex=1.052]
Epoch 93/100  loss=3.8717 (jet=0.6627 origin=1.1257 vtx=2.0832)  val_loss=3.7426 (jet=0.5944 origin=1.0759 vtx=2.0723)  val_acc=0.7674  origin_acc=0.6562  lxy_scale=[b_vertex=0.903, c_vertex=1.277]  dz_scale=[b_vertex=3.167, c_vertex=1.048]
Epoch 94/100  loss=3.8727 (jet=0.6631 origin=1.1264 vtx=2.0832)  val_loss=3.7109 (jet=0.5673 origin=1.0678 vtx=2.0758)  val_acc=0.7835  origin_acc=0.6783  lxy_scale=[b_vertex=0.906, c_vertex=1.258]  dz_scale=[b_vertex=3.167, c_vertex=1.046]
Epoch 95/100  loss=3.8734 (jet=0.6630 origin=1.1273 vtx=2.0832)  val_loss=3.7096 (jet=0.5810 origin=1.0561 vtx=2.0724)  val_acc=0.7749  origin_acc=0.6759  lxy_scale=[b_vertex=0.905, c_vertex=1.267]  dz_scale=[b_vertex=3.177, c_vertex=1.052]
Epoch 96/100  loss=3.8727 (jet=0.6641 origin=1.1261 vtx=2.0825)  val_loss=3.7149 (jet=0.5816 origin=1.0616 vtx=2.0717)  val_acc=0.7734  origin_acc=0.6671  lxy_scale=[b_vertex=0.908, c_vertex=1.278]  dz_scale=[b_vertex=3.208, c_vertex=1.054]
Epoch 97/100  loss=3.8702 (jet=0.6628 origin=1.1267 vtx=2.0808)  val_loss=3.7111 (jet=0.5767 origin=1.0663 vtx=2.0681)  val_acc=0.7740  origin_acc=0.6718  lxy_scale=[b_vertex=0.903, c_vertex=1.276]  dz_scale=[b_vertex=3.263, c_vertex=1.058]
Epoch 98/100  loss=3.8702 (jet=0.6624 origin=1.1263 vtx=2.0814)  val_loss=3.7105 (jet=0.5768 origin=1.0628 vtx=2.0709)  val_acc=0.7740  origin_acc=0.6654  lxy_scale=[b_vertex=0.904, c_vertex=1.266]  dz_scale=[b_vertex=3.279, c_vertex=1.062]
Epoch 99/100  loss=3.8677 (jet=0.6617 origin=1.1252 vtx=2.0807)  val_loss=3.7109 (jet=0.5738 origin=1.0649 vtx=2.0721)  val_acc=0.7764  origin_acc=0.6731  lxy_scale=[b_vertex=0.902, c_vertex=1.266]  dz_scale=[b_vertex=3.270, c_vertex=1.062]
Epoch 100/100  loss=3.8706 (jet=0.6624 origin=1.1266 vtx=2.0816)  val_loss=3.6849 (jet=0.5561 origin=1.0571 vtx=2.0717)  val_acc=0.7885  origin_acc=0.6621  lxy_scale=[b_vertex=0.905, c_vertex=1.263]  dz_scale=[b_vertex=3.279, c_vertex=1.059]
Saved staged_origin_vertex_jet.pt

Jet classification report:
              precision    recall  f1-score   support

       b-jet       0.94      0.76      0.84     77645
       c-jet       0.26      0.48      0.34     18430
   light-jet       0.88      0.86      0.87    103820

    accuracy                           0.79    199895
   macro avg       0.69      0.70      0.68    199895
weighted avg       0.84      0.79      0.81    199895

Jet confusion matrix (rows=true, cols=pred):
[[59065 12926  5654]
 [ 2621  8915  6894]
 [ 1363 12822 89635]]

Track-origin classification report:
                 precision    recall  f1-score   support

         Pileup       0.91      0.71      0.80    312395
           Fake       0.04      0.48      0.07      1677
        Primary       0.93      0.75      0.83    809100
         From b       0.40      0.40      0.40    133887
      From b->c       0.61      0.43      0.51    204949
         From c       0.11      0.50      0.18     44851
       From tau       0.03      0.25      0.05       585
Other secondary       0.22      0.75      0.34     39761

       accuracy                           0.66   1547205
      macro avg       0.40      0.53      0.40   1547205
   weighted avg       0.79      0.66      0.71   1547205

Saved training_summary.png  training_summary_log.png
Saved output_probs.png
Saved origin_confusion_matrix.png
Saved discriminant.png
Saved roc.png

=== b-tagging rejection rates ===
  ε_b=65%:  1/ε_c = 17  1/ε_light = 417
  ε_b=70%:  1/ε_c = 11  1/ε_light = 254
  ε_b=77%:  1/ε_c = 6  1/ε_light = 111
  ε_b=85%:  1/ε_c = 3  1/ε_light = 38
  ε_b=90%:  1/ε_c = 2  1/ε_light = 16
Saved rejection.png
Saved c_discriminant.png
Saved c_roc.png

=== c-tagging rejection rates ===
  ε_c=20%:  1/ε_b = 23  1/ε_light = 60
  ε_c=30%:  1/ε_b = 13  1/ε_light = 27
  ε_c=40%:  1/ε_b = 8  1/ε_light = 14
Saved c_rejection.png
Saved vertex_fit_b_vertex.png
Saved vertex_fit_b_vertex_dz.png
Saved vertex_fit_c_vertex.png
Saved vertex_fit_c_vertex_dz.png

=== Track-to-vertex assignment efficiency ===
  b_vertex  P_leg match:  min=0.0005  mean=0.6868  P25=0.4946  P50=0.7669  P75=0.9303  max=0.9927
  b_vertex  P_leg other:  min=0.0001  mean=0.1714  P50=0.0946  P90=0.4669  max=0.9893
  b_vertex  gate  match:  min=0.0067  mean=0.7246  P25=0.4866  P50=0.9352  P75=0.9867  max=0.9928
  b_vertex  gate  other:  min=0.0067  mean=0.1153  P50=0.0171  P90=0.4180  max=0.9926
  b_vertex  refine match:  min=0.2824  mean=0.5489  P25=0.5140  P50=0.5635  P75=0.5918  max=0.7059
  b_vertex  refine other:  min=0.2827  mean=0.5058  P50=0.5059  P90=0.5853  max=0.7338
  b_vertex  vtx_w range=[0.00670, 0.99281]  mean=0.42840  median=0.21212  P99=0.99232
  b_vertex (thr>0.5): assignment=0.746  false-positive=0.0870  n_match=335470
  b_vertex (thr>0.8): assignment=0.630  false-positive=0.0455  n_match=335470
Saved track_vtx_assignment_b_vertex.png
  c_vertex  P_leg match:  min=0.0002  mean=0.3562  P25=0.2153  P50=0.3639  P75=0.4951  max=0.8526
  c_vertex  P_leg other:  min=0.0000  mean=0.1470  P50=0.0989  P90=0.3736  max=0.7804
  c_vertex  gate  match:  min=0.0067  mean=0.2921  P25=0.0549  P50=0.2040  P75=0.4877  max=0.9714
  c_vertex  gate  other:  min=0.0067  mean=0.0748  P50=0.0178  P90=0.2204  max=0.9429
  c_vertex  refine match:  min=0.2111  mean=0.3909  P25=0.2924  P50=0.3809  P75=0.4548  max=0.7528
  c_vertex  refine other:  min=0.2213  mean=0.4583  P50=0.4481  P90=0.5847  max=0.7554
  c_vertex  vtx_w range=[0.00670, 0.97141]  mean=0.14133  median=0.03359  P99=0.86169
  c_vertex (thr>0.5): assignment=0.241  false-positive=0.0307  n_match=41464
  c_vertex (thr>0.8): assignment=0.059  false-positive=0.0040  n_match=41464
Saved track_vtx_assignment_c_vertex.png

All outputs saved to results/600k_training_100_epoch/results_staged_no_refine_20260707_165556/
