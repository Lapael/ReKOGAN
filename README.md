# ReKOGAN
## Introduce
ReKOGAN은 한글 손글씨 이미지를 생성하는 AI 프로젝트이다.
<br />(여기서 "Re"는 한번 망했다가 다시 진행했기에 붙인 것)
<br />
<br />GAN은 Generator과 Discriminator의 적대적 학습을 통해 훈련하는 생성형 인공지능의 종류 중 하나이다.
<br />
<br />
<img src="https://github.com/user-attachments/assets/89fb2d12-f7c3-405a-8909-d6c8ab70cd22" alt="" width="600"/>
<br />
<br />Generator가 생성하면 그 생성된 것이 Real인지 Fake인지 Discriminator가 판단한다.
<br />이를 반복하며 Generator가 Real을 더 잘 생성할 수 있도록 학습하는 방법이 GAN이다.
<br />
<br />이 프로젝트에서는 GAN을 통해 생성할 대상을 "한글 손글씨"로 한다.
<br />이 때, 총 520가지의 한글 문자 종류를 생성할 것이기에 GAN 중에서도 cGAN을 사용한다.
<br />cGAN은 External Information을 train중에 받는데 여기서 external info는 label에 해당된다.
<br />
## Datasets
이 프로젝트에 사용된 데이터셋은 callee2006의 HangulDB 중 SERI이다. (https://github.com/callee2006/HangulDB)
<br />SERI는 가장 빈번히 사용되는 한글 520글자 각각 900개의 손글씨 이미지로 구성되어 있다.
<br />이에 대한 label은 EUC-KR방식을 사용하였다.
<br />ex) b0a1 -> 가
<br />
## Train
학습은 epochs 100, batch 128로 진행하였고 각 epoch마다 '가'이미지 샘플을 만들어 저장하였다.

| | | | | | | | | | |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| <img src="train/samples/b0a1/sample_epoch_1.png" alt="Epoch 1" width="80"> | <img src="train/samples/b0a1/sample_epoch_2.png" alt="Epoch 2" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_3.png" alt="Epoch 3" width="80"> | <img src="train/samples/b0a1/sample_epoch_4.png" alt="Epoch 4" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_5.png" alt="Epoch 5" width="80"> | <img src="train/samples/b0a1/sample_epoch_6.png" alt="Epoch 6" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_7.png" alt="Epoch 7" width="80"> | <img src="train/samples/b0a1/sample_epoch_8.png" alt="Epoch 8" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_9.png" alt="Epoch 9" width="80"> | <img src="train/samples/b0a1/sample_epoch_10.png" alt="Epoch 10" width="80"> |
| <img src="train/samples/b0a1/sample_epoch_11.png" alt="Epoch 11" width="80"> | <img src="train/samples/b0a1/sample_epoch_12.png" alt="Epoch 12" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_13.png" alt="Epoch 13" width="80"> | <img src="train/samples/b0a1/sample_epoch_14.png" alt="Epoch 14" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_15.png" alt="Epoch 15" width="80"> | <img src="train/samples/b0a1/sample_epoch_16.png" alt="Epoch 16" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_17.png" alt="Epoch 17" width="80"> | <img src="train/samples/b0a1/sample_epoch_18.png" alt="Epoch 18" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_19.png" alt="Epoch 19" width="80"> | <img src="train/samples/b0a1/sample_epoch_20.png" alt="Epoch 20" width="80"> |
| <img src="train/samples/b0a1/sample_epoch_21.png" alt="Epoch 21" width="80"> | <img src="train/samples/b0a1/sample_epoch_22.png" alt="Epoch 22" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_23.png" alt="Epoch 23" width="80"> | <img src="train/samples/b0a1/sample_epoch_24.png" alt="Epoch 24" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_25.png" alt="Epoch 25" width="80"> | <img src="train/samples/b0a1/sample_epoch_26.png" alt="Epoch 26" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_27.png" alt="Epoch 27" width="80"> | <img src="train/samples/b0a1/sample_epoch_28.png" alt="Epoch 28" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_29.png" alt="Epoch 29" width="80"> | <img src="train/samples/b0a1/sample_epoch_30.png" alt="Epoch 30" width="80"> |
| <img src="train/samples/b0a1/sample_epoch_31.png" alt="Epoch 31" width="80"> | <img src="train/samples/b0a1/sample_epoch_32.png" alt="Epoch 32" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_33.png" alt="Epoch 33" width="80"> | <img src="train/samples/b0a1/sample_epoch_34.png" alt="Epoch 34" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_35.png" alt="Epoch 35" width="80"> | <img src="train/samples/b0a1/sample_epoch_36.png" alt="Epoch 36" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_37.png" alt="Epoch 37" width="80"> | <img src="train/samples/b0a1/sample_epoch_38.png" alt="Epoch 38" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_39.png" alt="Epoch 39" width="80"> | <img src="train/samples/b0a1/sample_epoch_40.png" alt="Epoch 40" width="80"> |
| <img src="train/samples/b0a1/sample_epoch_41.png" alt="Epoch 41" width="80"> | <img src="train/samples/b0a1/sample_epoch_42.png" alt="Epoch 42" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_43.png" alt="Epoch 43" width="80"> | <img src="train/samples/b0a1/sample_epoch_44.png" alt="Epoch 44" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_45.png" alt="Epoch 45" width="80"> | <img src="train/samples/b0a1/sample_epoch_46.png" alt="Epoch 46" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_47.png" alt="Epoch 47" width="80"> | <img src="train/samples/b0a1/sample_epoch_48.png" alt="Epoch 48" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_49.png" alt="Epoch 49" width="80"> | <img src="train/samples/b0a1/sample_epoch_50.png" alt="Epoch 50" width="80"> |
| <img src="train/samples/b0a1/sample_epoch_51.png" alt="Epoch 51" width="80"> | <img src="train/samples/b0a1/sample_epoch_52.png" alt="Epoch 52" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_53.png" alt="Epoch 53" width="80"> | <img src="train/samples/b0a1/sample_epoch_54.png" alt="Epoch 54" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_55.png" alt="Epoch 55" width="80"> | <img src="train/samples/b0a1/sample_epoch_56.png" alt="Epoch 56" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_57.png" alt="Epoch 57" width="80"> | <img src="train/samples/b0a1/sample_epoch_58.png" alt="Epoch 58" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_59.png" alt="Epoch 59" width="80"> | <img src="train/samples/b0a1/sample_epoch_60.png" alt="Epoch 60" width="80"> |
| <img src="train/samples/b0a1/sample_epoch_61.png" alt="Epoch 61" width="80"> | <img src="train/samples/b0a1/sample_epoch_62.png" alt="Epoch 62" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_63.png" alt="Epoch 63" width="80"> | <img src="train/samples/b0a1/sample_epoch_64.png" alt="Epoch 64" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_65.png" alt="Epoch 65" width="80"> | <img src="train/samples/b0a1/sample_epoch_66.png" alt="Epoch 66" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_67.png" alt="Epoch 67" width="80"> | <img src="train/samples/b0a1/sample_epoch_68.png" alt="Epoch 68" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_69.png" alt="Epoch 69" width="80"> | <img src="train/samples/b0a1/sample_epoch_70.png" alt="Epoch 70" width="80"> |
| <img src="train/samples/b0a1/sample_epoch_71.png" alt="Epoch 71" width="80"> | <img src="train/samples/b0a1/sample_epoch_72.png" alt="Epoch 72" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_73.png" alt="Epoch 73" width="80"> | <img src="train/samples/b0a1/sample_epoch_74.png" alt="Epoch 74" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_75.png" alt="Epoch 75" width="80"> | <img src="train/samples/b0a1/sample_epoch_76.png" alt="Epoch 76" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_77.png" alt="Epoch 77" width="80"> | <img src="train/samples/b0a1/sample_epoch_78.png" alt="Epoch 78" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_79.png" alt="Epoch 79" width="80"> | <img src="train/samples/b0a1/sample_epoch_80.png" alt="Epoch 80" width="80"> |
| <img src="train/samples/b0a1/sample_epoch_81.png" alt="Epoch 81" width="80"> | <img src="train/samples/b0a1/sample_epoch_82.png" alt="Epoch 82" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_83.png" alt="Epoch 83" width="80"> | <img src="train/samples/b0a1/sample_epoch_84.png" alt="Epoch 84" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_85.png" alt="Epoch 85" width="80"> | <img src="train/samples/b0a1/sample_epoch_86.png" alt="Epoch 86" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_87.png" alt="Epoch 87" width="80"> | <img src="train/samples/b0a1/sample_epoch_88.png" alt="Epoch 88" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_89.png" alt="Epoch 89" width="80"> | <img src="train/samples/b0a1/sample_epoch_90.png" alt="Epoch 90" width="80"> |
| <img src="train/samples/b0a1/sample_epoch_91.png" alt="Epoch 91" width="80"> | <img src="train/samples/b0a1/sample_epoch_92.png" alt="Epoch 92" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_93.png" alt="Epoch 93" width="80"> | <img src="train/samples/b0a1/sample_epoch_94.png" alt="Epoch 94" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_95.png" alt="Epoch 95" width="80"> | <img src="train/samples/b0a1/sample_epoch_96.png" alt="Epoch 96" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_97.png" alt="Epoch 97" width="80"> | <img src="train/samples/b0a1/sample_epoch_98.png" alt="Epoch 98" width="80"> | <img src=
      "train/samples/b0a1/sample_epoch_99.png" alt="Epoch 99" width="80"> | <img src="train/samples/b0a1/sample_epoch_100.png" alt="Epoch 100" width="80"> |
