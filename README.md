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
<br />
<br />
| | | | | | | | | | |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| <img src="train/samples/b0a1/sample_epoch_1.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_2.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_3.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_4.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_5.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_6.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_7.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_8.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_9.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_10.png" alt="" width="80"> |
| <img src="train/samples/b0a1/sample_epoch_11.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_12.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_13.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_14.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_15.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_16.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_17.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_18.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_19.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_20.png" alt="" width="80"> |
| <img src="train/samples/b0a1/sample_epoch_21.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_22.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_23.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_24.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_25.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_26.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_27.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_28.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_29.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_30.png" alt="" width="80"> |
| <img src="train/samples/b0a1/sample_epoch_31.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_32.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_33.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_34.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_35.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_36.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_37.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_38.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_39.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_40.png" alt="" width="80"> |
| <img src="train/samples/b0a1/sample_epoch_41.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_42.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_43.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_44.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_45.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_46.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_47.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_48.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_49.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_50.png" alt="" width="80"> |
| <img src="train/samples/b0a1/sample_epoch_51.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_52.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_53.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_54.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_55.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_56.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_57.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_58.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_59.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_60.png" alt="" width="80"> |
| <img src="train/samples/b0a1/sample_epoch_61.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_62.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_63.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_64.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_65.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_66.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_67.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_68.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_69.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_70.png" alt="" width="80"> |
| <img src="train/samples/b0a1/sample_epoch_71.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_72.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_73.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_74.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_75.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_76.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_77.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_78.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_79.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_80.png" alt="" width="80"> |
| <img src="train/samples/b0a1/sample_epoch_81.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_82.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_83.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_84.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_85.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_86.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_87.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_88.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_89.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_90.png" alt="" width="80"> |
| <img src="train/samples/b0a1/sample_epoch_91.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_92.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_93.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_94.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_95.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_96.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_97.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_98.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_99.png" alt="" width="80"> | <img src="train/samples/b0a1/sample_epoch_100.png" alt="" width="80"> |

왼쪽 위가 epoch1 오른쪽 아래가 epoch100
<br />
<br />이를 보면 초중반에는 학습이 진행되다가 후반에는 노이즈가 낀 것처럼 이미지가 명확하지 않아진다.
<br />또한 후반부로 갈수록 '가'와 다른 '차', '핀', '긴'등과 가까워 보이는 이미지가 생성되고 있다.
<br />
<br />이를 기반으로 Mode Collapse, Conditional Mode Collapse, Label Collapse 등이 일어났다는 것을 유추할 수 있다.
<br />이러한 Collapse들은 생성자가 판별자를 속이기 위한 쉬운 길을 찾으며 발생한다.
<br />
<br />그나마 생성을 잘 한것 같은 epoch8의 모델을 불러와 생성을 해보면

| | | |
|:---:|:---:|:---:|
| <img src="https://github.com/user-attachments/assets/3ddb1d3e-6f93-4463-8909-15a568b4386f" alt="가" width="100"> | <img src="https://github.com/user-attachments/assets/fff59981-e5b5-45d6-82fb-a07a007cb728" alt="다" width="100"> | <img src="https://github.com/user-attachments/assets/8e7e8457-c3c7-45c2-9223-cc1f66cbcf05" alt="각" width="100">

