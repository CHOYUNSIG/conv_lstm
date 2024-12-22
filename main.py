import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader
from model import ConvLSTM
from train import train
from test import test


def main(dataset_dir,
         hidden_channels,
         kernel_size,
         num_layers,
         batch_size,
         num_epochs,
         frame_count,
         learning_rate):
    num_classes = 6  # 분류할 동작 클래스 수
    input_channels = 1  # MHI는 그레이스케일
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 데이터셋 로드
    train_loader, test_loader = get_dataloader(root_dir=dataset_dir, batch_size=batch_size, frame_count=frame_count)
    
    # 첫 번째 배치로부터 입력 크기 얻기
    sample_data, *_ = next(iter(train_loader))
    input_size = (sample_data.size(3), sample_data.size(4))

    # 모델 초기화
    model = ConvLSTM(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        input_size=input_size,
        num_layers=num_layers,
        num_classes=num_classes,
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 학습 실행
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch} of {num_epochs}...")
        train(model, train_loader, criterion, optimizer, device)

    # 테스트
    test(model, test_loader, device, criterion)


if __name__ == "__main__":
    main(
        hidden_channels=32,
        kernel_size=3,
        num_layers=2,
        batch_size=4,
        num_epochs=100,
        frame_count=50,
        learning_rate=0.001,
        dataset_dir='dataset'
    )