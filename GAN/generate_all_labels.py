import os
import sys
import torch
import torch.nn as nn
from torchvision.utils import save_image

sys.path.append(r"C:\Users\EunSung\Desktop\ReKoGAN\train")
from training import Generator, HangulDataset

def main(epoch):
    # 경로 설정
    model_dir = r"C:\Users\EunSung\Desktop\ReKoGAN\train\model_save"
    dataset_dir = r"C:\Users\EunSung\Desktop\ReKoGAN\train\datasets"

    # 사용자 입력
    # epoch = input("Enter the epoch number: ")
    output_dir = fr"C:\Users\EunSung\Desktop\ReKoGAN\GAN\generated\ALL_{epoch}"
    os.makedirs(output_dir, exist_ok=True)
    # 데이터셋을 불러와 클래스 정보를 얻음
    dataset = HangulDataset(dataset_dir, image_size=64)
    all_labels = dataset.classes
    num_classes = len(all_labels)

    # 하이퍼파라미터 설정 (training.py 기준)
    nz = 100
    ngf = 64
    nc = 1

    # 모델 생성 및 가중치 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(nz, ngf, nc, num_classes).to(device)
    model_path = os.path.join(model_dir, f"G_epoch_{epoch}.pth")

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()

    print(f"Generating images for all {num_classes} labels for epoch {epoch}...")

    with torch.no_grad():
        for encoded_label in all_labels:
            label_idx = dataset.class_to_idx[encoded_label]
            
            # 레이블 원-핫 인코딩
            label = torch.zeros(1, num_classes)
            label[0, label_idx] = 1.0
            label = label.to(device)

            # 노이즈 벡터 생성
            noise = torch.randn(1, nz).to(device)

            fake_img = generator(noise, label).detach().cpu()
            save_path = os.path.join(output_dir, f"generated_{encoded_label}_epoch{epoch}.png")
            save_image(fake_img, save_path, normalize=True)
            print(f"  - Saved: {save_path}")

    print("\nAll images generated successfully!")

if __name__ == '__main__':
    for i in range(100):
        main(i+1)
