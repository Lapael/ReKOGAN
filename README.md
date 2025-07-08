# ReKOGAN
## Introduce
ReKOGAN은 한글 손글씨 이미지를 생성하는 AI 프로젝트이다.
<br />(여기서 "Re"는 한번 망했다가 다시 진행했기에 붙인 것)
<br />
<br />GAN은 Generator과 Discriminator의 적대적 학습을 통해 훈련하는 생성형 인공지능의 종류 중 하나이다.
<br />
<br />![image](https://github.com/user-attachments/assets/89fb2d12-f7c3-405a-8909-d6c8ab70cd22)
<br />Generator가 생성하면 그 생성된 것이 Real인지 Fake인지 Discriminator가 판단한다.
<br />이를 반복하며 Generator가 Real을 더 잘 생성할 수 있도록 학습하는 방법이 GAN이다.
<br />
<br />이 프로젝트에서는 GAN을 통해 생성할 대상을 "한글 손글씨"로 한다.
<br />이 때, 총 520가지의 한글 문자 종류를 생성할 것이기에 GAN 중에서도 cGAN을 사용한다.
<br />cGAN은 External Information을 train중에 받는데 여기서 external info는 label에 해당된다.
<br />
## Datasets
