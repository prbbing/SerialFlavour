Device: ['0']  |  DataParallel: False
Config saved to ./results/results_parallel_20260706_233220/config.json
Loading data...
  [1/3] Loading flavour labels (cached)
  [2/3] Loading training tracks ...
  [3/3] Loading test tracks ...
Train — b-jet:199,908  c-jet:199,941  light-jet:199,868
Test  — b-jet:77,645  c-jet:18,430  light-jet:103,820
Saved input_variables.png
Model type: parallel_origin_vertex_jet
Origin class weights:
   0  Pileup              0.013
   1  Fake                0.626
   2  Primary             0.005
   3  From b              0.036
   4  From b->c           0.023
   5  From c              0.027
   6  From tau            0.626
   7  Other secondary     0.113
Parameters: 54,573
Device: cuda:0  |  Train: 599,717  |  Test: 199,895

Epoch 01/100  loss=3.3824 (jet=0.8805 origin=1.6448 vtx=0.8572)  val_loss=2.4601 (jet=0.6877 origin=1.3527 vtx=0.4198)  val_acc=0.7455  origin_acc=0.6774
Epoch 02/100  loss=2.6136 (jet=0.7820 origin=1.3706 vtx=0.4609)  val_loss=2.2951 (jet=0.6673 origin=1.2390 vtx=0.3888)  val_acc=0.7535  origin_acc=0.6819
Epoch 03/100  loss=2.4786 (jet=0.7584 origin=1.2892 vtx=0.4309)  val_loss=2.2277 (jet=0.6677 origin=1.1806 vtx=0.3794)  val_acc=0.7452  origin_acc=0.6682
Epoch 04/100  loss=2.4046 (jet=0.7445 origin=1.2440 vtx=0.4161)  val_loss=2.1576 (jet=0.6424 origin=1.1483 vtx=0.3668)  val_acc=0.7564  origin_acc=0.6837
Epoch 05/100  loss=2.3628 (jet=0.7355 origin=1.2200 vtx=0.4073)  val_loss=2.1352 (jet=0.6375 origin=1.1354 vtx=0.3623)  val_acc=0.7596  origin_acc=0.6759
Epoch 06/100  loss=2.3351 (jet=0.7299 origin=1.2041 vtx=0.4011)  val_loss=2.1121 (jet=0.6203 origin=1.1324 vtx=0.3594)  val_acc=0.7642  origin_acc=0.6894
Epoch 07/100  loss=2.3113 (jet=0.7245 origin=1.1909 vtx=0.3959)  val_loss=2.1040 (jet=0.6300 origin=1.1164 vtx=0.3576)  val_acc=0.7627  origin_acc=0.6789
Epoch 08/100  loss=2.2944 (jet=0.7209 origin=1.1814 vtx=0.3920)  val_loss=2.0521 (jet=0.5989 origin=1.1021 vtx=0.3510)  val_acc=0.7817  origin_acc=0.6949
Epoch 09/100  loss=2.2801 (jet=0.7182 origin=1.1727 vtx=0.3891)  val_loss=2.0403 (jet=0.5934 origin=1.0980 vtx=0.3489)  val_acc=0.7819  origin_acc=0.6870
Epoch 10/100  loss=2.2682 (jet=0.7153 origin=1.1665 vtx=0.3864)  val_loss=2.0625 (jet=0.6177 origin=1.0946 vtx=0.3502)  val_acc=0.7660  origin_acc=0.6747
Epoch 11/100  loss=2.2583 (jet=0.7134 origin=1.1607 vtx=0.3843)  val_loss=2.0983 (jet=0.6376 origin=1.1065 vtx=0.3541)  val_acc=0.7485  origin_acc=0.6615
Epoch 12/100  loss=2.2491 (jet=0.7117 origin=1.1551 vtx=0.3823)  val_loss=2.1200 (jet=0.6685 origin=1.0982 vtx=0.3533)  val_acc=0.7269  origin_acc=0.6457
Epoch 13/100  loss=2.2392 (jet=0.7095 origin=1.1494 vtx=0.3803)  val_loss=2.0368 (jet=0.6152 origin=1.0720 vtx=0.3496)  val_acc=0.7668  origin_acc=0.6804
Epoch 14/100  loss=2.2286 (jet=0.7075 origin=1.1430 vtx=0.3782)  val_loss=2.0464 (jet=0.6182 origin=1.0821 vtx=0.3461)  val_acc=0.7645  origin_acc=0.6590
Epoch 15/100  loss=2.2182 (jet=0.7050 origin=1.1371 vtx=0.3761)  val_loss=2.0256 (jet=0.6033 origin=1.0812 vtx=0.3411)  val_acc=0.7707  origin_acc=0.6808
Epoch 16/100  loss=2.2090 (jet=0.7030 origin=1.1318 vtx=0.3742)  val_loss=2.0251 (jet=0.6114 origin=1.0755 vtx=0.3381)  val_acc=0.7657  origin_acc=0.6888
Epoch 17/100  loss=2.1997 (jet=0.7012 origin=1.1261 vtx=0.3724)  val_loss=2.0499 (jet=0.6306 origin=1.0728 vtx=0.3465)  val_acc=0.7520  origin_acc=0.6488
Epoch 18/100  loss=2.1905 (jet=0.6982 origin=1.1217 vtx=0.3706)  val_loss=2.0052 (jet=0.6076 origin=1.0567 vtx=0.3409)  val_acc=0.7661  origin_acc=0.6693
Epoch 19/100  loss=2.1823 (jet=0.6966 origin=1.1165 vtx=0.3692)  val_loss=1.9939 (jet=0.6024 origin=1.0573 vtx=0.3342)  val_acc=0.7688  origin_acc=0.6849
Epoch 20/100  loss=2.1732 (jet=0.6936 origin=1.1117 vtx=0.3678)  val_loss=1.9941 (jet=0.6045 origin=1.0488 vtx=0.3408)  val_acc=0.7652  origin_acc=0.6746
Epoch 21/100  loss=2.1661 (jet=0.6916 origin=1.1078 vtx=0.3666)  val_loss=1.9821 (jet=0.5974 origin=1.0517 vtx=0.3331)  val_acc=0.7738  origin_acc=0.6750
Epoch 22/100  loss=2.1599 (jet=0.6901 origin=1.1042 vtx=0.3656)  val_loss=2.0205 (jet=0.6236 origin=1.0623 vtx=0.3346)  val_acc=0.7520  origin_acc=0.6557
Epoch 23/100  loss=2.1512 (jet=0.6873 origin=1.1000 vtx=0.3640)  val_loss=1.9672 (jet=0.5950 origin=1.0418 vtx=0.3304)  val_acc=0.7710  origin_acc=0.6795
Epoch 24/100  loss=2.1448 (jet=0.6856 origin=1.0963 vtx=0.3630)  val_loss=1.9426 (jet=0.5796 origin=1.0342 vtx=0.3288)  val_acc=0.7839  origin_acc=0.7017
Epoch 25/100  loss=2.1386 (jet=0.6839 origin=1.0926 vtx=0.3621)  val_loss=1.9857 (jet=0.6133 origin=1.0328 vtx=0.3396)  val_acc=0.7617  origin_acc=0.6726
Epoch 26/100  loss=2.1314 (jet=0.6822 origin=1.0883 vtx=0.3609)  val_loss=1.9447 (jet=0.5881 origin=1.0291 vtx=0.3275)  val_acc=0.7801  origin_acc=0.6899
Epoch 27/100  loss=2.1259 (jet=0.6806 origin=1.0857 vtx=0.3596)  val_loss=1.9799 (jet=0.6093 origin=1.0443 vtx=0.3263)  val_acc=0.7617  origin_acc=0.6853
Epoch 28/100  loss=2.1216 (jet=0.6795 origin=1.0830 vtx=0.3590)  val_loss=1.9672 (jet=0.5981 origin=1.0339 vtx=0.3352)  val_acc=0.7675  origin_acc=0.6701
Epoch 29/100  loss=2.1146 (jet=0.6775 origin=1.0794 vtx=0.3578)  val_loss=1.9173 (jet=0.5737 origin=1.0183 vtx=0.3252)  val_acc=0.7862  origin_acc=0.6990
Epoch 30/100  loss=2.1103 (jet=0.6766 origin=1.0769 vtx=0.3568)  val_loss=1.9262 (jet=0.5708 origin=1.0332 vtx=0.3222)  val_acc=0.7830  origin_acc=0.6977
Epoch 31/100  loss=2.1071 (jet=0.6758 origin=1.0747 vtx=0.3566)  val_loss=1.9364 (jet=0.5808 origin=1.0323 vtx=0.3233)  val_acc=0.7793  origin_acc=0.6887
Epoch 32/100  loss=2.1015 (jet=0.6741 origin=1.0717 vtx=0.3557)  val_loss=1.9763 (jet=0.6117 origin=1.0367 vtx=0.3279)  val_acc=0.7585  origin_acc=0.6682
Epoch 33/100  loss=2.0968 (jet=0.6727 origin=1.0692 vtx=0.3549)  val_loss=1.9451 (jet=0.5943 origin=1.0211 vtx=0.3297)  val_acc=0.7672  origin_acc=0.6779
Epoch 34/100  loss=2.0930 (jet=0.6718 origin=1.0670 vtx=0.3542)  val_loss=1.9397 (jet=0.5940 origin=1.0194 vtx=0.3263)  val_acc=0.7667  origin_acc=0.6792
Epoch 35/100  loss=2.0893 (jet=0.6709 origin=1.0651 vtx=0.3534)  val_loss=1.9421 (jet=0.5915 origin=1.0293 vtx=0.3213)  val_acc=0.7680  origin_acc=0.6758
Epoch 36/100  loss=2.0857 (jet=0.6701 origin=1.0629 vtx=0.3527)  val_loss=1.9197 (jet=0.5775 origin=1.0196 vtx=0.3226)  val_acc=0.7786  origin_acc=0.6951
Epoch 37/100  loss=2.0830 (jet=0.6696 origin=1.0612 vtx=0.3522)  val_loss=1.9443 (jet=0.6032 origin=1.0142 vtx=0.3269)  val_acc=0.7632  origin_acc=0.6762
Epoch 38/100  loss=2.0799 (jet=0.6685 origin=1.0599 vtx=0.3516)  val_loss=1.9054 (jet=0.5793 origin=1.0052 vtx=0.3209)  val_acc=0.7784  origin_acc=0.6856
Epoch 39/100  loss=2.0755 (jet=0.6676 origin=1.0571 vtx=0.3509)  val_loss=1.9144 (jet=0.5840 origin=1.0091 vtx=0.3212)  val_acc=0.7767  origin_acc=0.6739
Epoch 40/100  loss=2.0722 (jet=0.6664 origin=1.0555 vtx=0.3504)  val_loss=1.9245 (jet=0.5910 origin=1.0136 vtx=0.3199)  val_acc=0.7694  origin_acc=0.6801
Epoch 41/100  loss=2.0694 (jet=0.6657 origin=1.0540 vtx=0.3497)  val_loss=1.8961 (jet=0.5649 origin=1.0077 vtx=0.3236)  val_acc=0.7859  origin_acc=0.6811
Epoch 42/100  loss=2.0680 (jet=0.6657 origin=1.0528 vtx=0.3495)  val_loss=1.8621 (jet=0.5542 origin=0.9915 vtx=0.3163)  val_acc=0.7944  origin_acc=0.7107
Epoch 43/100  loss=2.0641 (jet=0.6644 origin=1.0509 vtx=0.3488)  val_loss=1.8787 (jet=0.5644 origin=0.9950 vtx=0.3194)  val_acc=0.7868  origin_acc=0.6943
Epoch 44/100  loss=2.0606 (jet=0.6638 origin=1.0482 vtx=0.3486)  val_loss=1.8911 (jet=0.5695 origin=1.0051 vtx=0.3164)  val_acc=0.7818  origin_acc=0.7036
Epoch 45/100  loss=2.0595 (jet=0.6633 origin=1.0478 vtx=0.3483)  val_loss=1.8856 (jet=0.5683 origin=1.0007 vtx=0.3165)  val_acc=0.7857  origin_acc=0.6838
Epoch 46/100  loss=2.0573 (jet=0.6629 origin=1.0465 vtx=0.3479)  val_loss=1.8784 (jet=0.5605 origin=1.0020 vtx=0.3159)  val_acc=0.7880  origin_acc=0.6897
Epoch 47/100  loss=2.0569 (jet=0.6628 origin=1.0465 vtx=0.3476)  val_loss=1.8872 (jet=0.5751 origin=0.9951 vtx=0.3170)  val_acc=0.7835  origin_acc=0.7051
Epoch 48/100  loss=2.0509 (jet=0.6609 origin=1.0431 vtx=0.3469)  val_loss=1.9094 (jet=0.5868 origin=1.0034 vtx=0.3192)  val_acc=0.7755  origin_acc=0.6842
Epoch 49/100  loss=2.0485 (jet=0.6608 origin=1.0412 vtx=0.3465)  val_loss=1.8877 (jet=0.5751 origin=0.9940 vtx=0.3186)  val_acc=0.7799  origin_acc=0.6968
Epoch 50/100  loss=2.0486 (jet=0.6607 origin=1.0416 vtx=0.3462)  val_loss=1.8656 (jet=0.5573 origin=0.9950 vtx=0.3133)  val_acc=0.7892  origin_acc=0.7050
Epoch 51/100  loss=2.0443 (jet=0.6594 origin=1.0390 vtx=0.3458)  val_loss=1.8688 (jet=0.5656 origin=0.9871 vtx=0.3161)  val_acc=0.7868  origin_acc=0.7074
Epoch 52/100  loss=2.0439 (jet=0.6590 origin=1.0391 vtx=0.3459)  val_loss=1.8731 (jet=0.5642 origin=0.9963 vtx=0.3126)  val_acc=0.7860  origin_acc=0.6934
Epoch 53/100  loss=2.0420 (jet=0.6586 origin=1.0380 vtx=0.3453)  val_loss=1.8741 (jet=0.5731 origin=0.9862 vtx=0.3148)  val_acc=0.7767  origin_acc=0.6934
Epoch 54/100  loss=2.0394 (jet=0.6579 origin=1.0366 vtx=0.3449)  val_loss=1.8768 (jet=0.5689 origin=0.9949 vtx=0.3131)  val_acc=0.7838  origin_acc=0.7082
Epoch 55/100  loss=2.0390 (jet=0.6579 origin=1.0363 vtx=0.3448)  val_loss=1.8801 (jet=0.5755 origin=0.9867 vtx=0.3179)  val_acc=0.7838  origin_acc=0.7032
Epoch 56/100  loss=2.0369 (jet=0.6575 origin=1.0348 vtx=0.3447)  val_loss=1.8744 (jet=0.5702 origin=0.9842 vtx=0.3201)  val_acc=0.7811  origin_acc=0.6929
Epoch 57/100  loss=2.0350 (jet=0.6570 origin=1.0338 vtx=0.3442)  val_loss=1.8702 (jet=0.5693 origin=0.9868 vtx=0.3141)  val_acc=0.7820  origin_acc=0.6930
Epoch 58/100  loss=2.0325 (jet=0.6562 origin=1.0325 vtx=0.3439)  val_loss=1.8598 (jet=0.5696 origin=0.9773 vtx=0.3130)  val_acc=0.7815  origin_acc=0.6902
Epoch 59/100  loss=2.0325 (jet=0.6566 origin=1.0318 vtx=0.3441)  val_loss=1.9026 (jet=0.5902 origin=0.9962 vtx=0.3162)  val_acc=0.7709  origin_acc=0.6763
Epoch 60/100  loss=2.0314 (jet=0.6564 origin=1.0313 vtx=0.3437)  val_loss=1.8807 (jet=0.5765 origin=0.9873 vtx=0.3168)  val_acc=0.7756  origin_acc=0.6837
Epoch 61/100  loss=2.0303 (jet=0.6557 origin=1.0311 vtx=0.3435)  val_loss=1.8333 (jet=0.5436 origin=0.9794 vtx=0.3104)  val_acc=0.7992  origin_acc=0.7150
Epoch 62/100  loss=2.0286 (jet=0.6552 origin=1.0302 vtx=0.3432)  val_loss=1.8399 (jet=0.5538 origin=0.9768 vtx=0.3093)  val_acc=0.7949  origin_acc=0.7114
Epoch 63/100  loss=2.0253 (jet=0.6543 origin=1.0282 vtx=0.3429)  val_loss=1.8547 (jet=0.5591 origin=0.9858 vtx=0.3099)  val_acc=0.7908  origin_acc=0.6958
Epoch 64/100  loss=2.0252 (jet=0.6546 origin=1.0276 vtx=0.3430)  val_loss=1.8956 (jet=0.5861 origin=0.9958 vtx=0.3137)  val_acc=0.7713  origin_acc=0.6924
Epoch 65/100  loss=2.0248 (jet=0.6544 origin=1.0274 vtx=0.3430)  val_loss=1.8652 (jet=0.5651 origin=0.9879 vtx=0.3122)  val_acc=0.7863  origin_acc=0.6981
Epoch 66/100  loss=2.0227 (jet=0.6537 origin=1.0263 vtx=0.3427)  val_loss=1.8866 (jet=0.5866 origin=0.9850 vtx=0.3149)  val_acc=0.7702  origin_acc=0.6833
Epoch 67/100  loss=2.0223 (jet=0.6533 origin=1.0263 vtx=0.3427)  val_loss=1.8533 (jet=0.5666 origin=0.9751 vtx=0.3116)  val_acc=0.7839  origin_acc=0.6991
Epoch 68/100  loss=2.0196 (jet=0.6530 origin=1.0247 vtx=0.3420)  val_loss=1.8370 (jet=0.5519 origin=0.9752 vtx=0.3099)  val_acc=0.7913  origin_acc=0.7158
Epoch 69/100  loss=2.0208 (jet=0.6533 origin=1.0250 vtx=0.3425)  val_loss=1.8953 (jet=0.5993 origin=0.9824 vtx=0.3136)  val_acc=0.7617  origin_acc=0.6864
Epoch 70/100  loss=2.0181 (jet=0.6527 origin=1.0234 vtx=0.3420)  val_loss=1.8644 (jet=0.5740 origin=0.9776 vtx=0.3129)  val_acc=0.7797  origin_acc=0.7078
Epoch 71/100  loss=2.0161 (jet=0.6520 origin=1.0222 vtx=0.3419)  val_loss=1.8775 (jet=0.5801 origin=0.9811 vtx=0.3163)  val_acc=0.7732  origin_acc=0.6959
Epoch 72/100  loss=2.0177 (jet=0.6521 origin=1.0236 vtx=0.3420)  val_loss=1.8530 (jet=0.5615 origin=0.9806 vtx=0.3108)  val_acc=0.7880  origin_acc=0.7027
Epoch 73/100  loss=2.0146 (jet=0.6515 origin=1.0214 vtx=0.3417)  val_loss=1.8692 (jet=0.5761 origin=0.9756 vtx=0.3176)  val_acc=0.7788  origin_acc=0.6923
Epoch 74/100  loss=2.0143 (jet=0.6514 origin=1.0215 vtx=0.3415)  val_loss=1.8731 (jet=0.5776 origin=0.9820 vtx=0.3136)  val_acc=0.7769  origin_acc=0.6938
Epoch 75/100  loss=2.0139 (jet=0.6513 origin=1.0213 vtx=0.3413)  val_loss=1.8729 (jet=0.5756 origin=0.9864 vtx=0.3110)  val_acc=0.7788  origin_acc=0.6876
Epoch 76/100  loss=2.0127 (jet=0.6511 origin=1.0204 vtx=0.3412)  val_loss=1.8527 (jet=0.5643 origin=0.9741 vtx=0.3143)  val_acc=0.7847  origin_acc=0.6968
Epoch 77/100  loss=2.0112 (jet=0.6505 origin=1.0197 vtx=0.3409)  val_loss=1.8676 (jet=0.5746 origin=0.9813 vtx=0.3118)  val_acc=0.7789  origin_acc=0.6978
Epoch 78/100  loss=2.0106 (jet=0.6504 origin=1.0190 vtx=0.3413)  val_loss=1.8607 (jet=0.5689 origin=0.9771 vtx=0.3147)  val_acc=0.7801  origin_acc=0.6803
Epoch 79/100  loss=2.0101 (jet=0.6504 origin=1.0185 vtx=0.3411)  val_loss=1.8623 (jet=0.5734 origin=0.9721 vtx=0.3168)  val_acc=0.7784  origin_acc=0.6996
Epoch 80/100  loss=2.0093 (jet=0.6502 origin=1.0182 vtx=0.3409)  val_loss=1.8705 (jet=0.5784 origin=0.9798 vtx=0.3123)  val_acc=0.7752  origin_acc=0.6876
Epoch 81/100  loss=2.0072 (jet=0.6493 origin=1.0172 vtx=0.3407)  val_loss=1.8722 (jet=0.5785 origin=0.9784 vtx=0.3153)  val_acc=0.7739  origin_acc=0.6865
Epoch 82/100  loss=2.0058 (jet=0.6493 origin=1.0160 vtx=0.3405)  val_loss=1.8406 (jet=0.5570 origin=0.9746 vtx=0.3089)  val_acc=0.7928  origin_acc=0.7114
Epoch 83/100  loss=2.0063 (jet=0.6490 origin=1.0169 vtx=0.3404)  val_loss=1.8399 (jet=0.5587 origin=0.9701 vtx=0.3111)  val_acc=0.7904  origin_acc=0.7113
Epoch 84/100  loss=2.0063 (jet=0.6493 origin=1.0166 vtx=0.3404)  val_loss=1.8343 (jet=0.5525 origin=0.9740 vtx=0.3078)  val_acc=0.7940  origin_acc=0.7152
Epoch 85/100  loss=2.0039 (jet=0.6488 origin=1.0149 vtx=0.3403)  val_loss=1.8556 (jet=0.5665 origin=0.9782 vtx=0.3108)  val_acc=0.7836  origin_acc=0.7056
Epoch 86/100  loss=2.0022 (jet=0.6483 origin=1.0136 vtx=0.3402)  val_loss=1.8369 (jet=0.5528 origin=0.9752 vtx=0.3088)  val_acc=0.7912  origin_acc=0.6971
Epoch 87/100  loss=2.0041 (jet=0.6486 origin=1.0151 vtx=0.3403)  val_loss=1.8692 (jet=0.5781 origin=0.9752 vtx=0.3160)  val_acc=0.7762  origin_acc=0.6840
Epoch 88/100  loss=2.0016 (jet=0.6485 origin=1.0132 vtx=0.3400)  val_loss=1.8329 (jet=0.5477 origin=0.9742 vtx=0.3110)  val_acc=0.7938  origin_acc=0.7115
Epoch 89/100  loss=2.0009 (jet=0.6482 origin=1.0129 vtx=0.3397)  val_loss=1.8613 (jet=0.5733 origin=0.9746 vtx=0.3134)  val_acc=0.7768  origin_acc=0.6868
Epoch 90/100  loss=2.0006 (jet=0.6480 origin=1.0128 vtx=0.3397)  val_loss=1.8442 (jet=0.5626 origin=0.9699 vtx=0.3117)  val_acc=0.7843  origin_acc=0.6984
Epoch 91/100  loss=2.0000 (jet=0.6479 origin=1.0125 vtx=0.3396)  val_loss=1.8660 (jet=0.5746 origin=0.9773 vtx=0.3141)  val_acc=0.7772  origin_acc=0.6880
Epoch 92/100  loss=1.9988 (jet=0.6471 origin=1.0122 vtx=0.3395)  val_loss=1.8113 (jet=0.5353 origin=0.9687 vtx=0.3072)  val_acc=0.8028  origin_acc=0.7245
Epoch 93/100  loss=1.9987 (jet=0.6473 origin=1.0114 vtx=0.3399)  val_loss=1.8312 (jet=0.5549 origin=0.9667 vtx=0.3097)  val_acc=0.7884  origin_acc=0.7109
Epoch 94/100  loss=1.9965 (jet=0.6467 origin=1.0105 vtx=0.3393)  val_loss=1.8763 (jet=0.5887 origin=0.9728 vtx=0.3148)  val_acc=0.7682  origin_acc=0.6887
Epoch 95/100  loss=1.9964 (jet=0.6464 origin=1.0108 vtx=0.3392)  val_loss=1.8320 (jet=0.5515 origin=0.9706 vtx=0.3098)  val_acc=0.7903  origin_acc=0.6956
Epoch 96/100  loss=1.9957 (jet=0.6460 origin=1.0104 vtx=0.3393)  val_loss=1.8837 (jet=0.5849 origin=0.9879 vtx=0.3110)  val_acc=0.7672  origin_acc=0.6864
Epoch 97/100  loss=1.9960 (jet=0.6467 origin=1.0101 vtx=0.3392)  val_loss=1.8276 (jet=0.5499 origin=0.9692 vtx=0.3085)  val_acc=0.7910  origin_acc=0.6971
Epoch 98/100  loss=1.9957 (jet=0.6465 origin=1.0097 vtx=0.3395)  val_loss=1.8656 (jet=0.5761 origin=0.9745 vtx=0.3150)  val_acc=0.7748  origin_acc=0.6854
Epoch 99/100  loss=1.9941 (jet=0.6459 origin=1.0091 vtx=0.3390)  val_loss=1.8391 (jet=0.5665 origin=0.9628 vtx=0.3099)  val_acc=0.7829  origin_acc=0.7122
Epoch 100/100  loss=1.9933 (jet=0.6453 origin=1.0091 vtx=0.3389)  val_loss=1.8366 (jet=0.5587 origin=0.9669 vtx=0.3109)  val_acc=0.7884  origin_acc=0.7036
Saved parallel_origin_vertex_jet.pt

Jet classification report:
              precision    recall  f1-score   support

       b-jet       0.93      0.78      0.85     77645
       c-jet       0.26      0.50      0.34     18430
   light-jet       0.89      0.84      0.86    103820

    accuracy                           0.79    199895
   macro avg       0.69      0.71      0.69    199895
weighted avg       0.85      0.79      0.81    199895

Jet confusion matrix (rows=true, cols=pred):
[[60674 12055  4916]
 [ 2804  9270  6356]
 [ 1440 14730 87650]]

Track-origin classification report:
                 precision    recall  f1-score   support

         Pileup       0.91      0.73      0.81    312395
           Fake       0.04      0.47      0.08      1677
        Primary       0.93      0.79      0.85    809100
         From b       0.45      0.49      0.47    133887
      From b->c       0.66      0.48      0.56    204949
         From c       0.14      0.52      0.22     44851
       From tau       0.03      0.34      0.05       585
Other secondary       0.25      0.77      0.38     39761

       accuracy                           0.70   1547205
      macro avg       0.43      0.57      0.43   1547205
   weighted avg       0.81      0.70      0.74   1547205

Saved training_summary.png  training_summary_log.png
Saved origin_confusion_matrix.png
Saved pair_vertexing.png
Saved output_probs.png
Saved discriminant.png
Saved roc.png

=== b-tagging rejection rates ===
  ε_b=65%:  1/ε_c = 20  1/ε_light = 600
  ε_b=70%:  1/ε_c = 13  1/ε_light = 329
  ε_b=77%:  1/ε_c = 7  1/ε_light = 127
  ε_b=85%:  1/ε_c = 4  1/ε_light = 41
  ε_b=90%:  1/ε_c = 2  1/ε_light = 17
Saved rejection.png
Saved c_discriminant.png
Saved c_roc.png

=== c-tagging rejection rates ===
  ε_c=20%:  1/ε_b = 26  1/ε_light = 61
  ε_c=30%:  1/ε_b = 14  1/ε_light = 27
  ε_c=40%:  1/ε_b = 9  1/ε_light = 13
Saved c_rejection.png

All outputs saved to ./results/results_parallel_20260706_233220/
