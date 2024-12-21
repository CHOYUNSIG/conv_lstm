import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
from mhi import get_mhi
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class ActionDataset(Dataset):
    def __init__(self, root_dir, frame_count=30):
        self.root_dir = root_dir
        self.samples = []
        self.label_map = {}
        self.frame_count = frame_count
        
        # 라벨 매핑 생성
        for idx, label in enumerate(os.listdir(root_dir)):
            self.label_map[label] = idx
            
        # 데이터 경로와 라벨 수집
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            n = 0
            for video_file in os.listdir(label_dir):
                if video_file.endswith('.avi'):
                    n += 1
                    self.samples.append({
                        'path': os.path.join(label_dir, video_file),
                        'label': self.label_map[label]
                    })
            print(f"Loaded {n} samples for label {label}")

    def get_label_map(self):
        return self.label_map
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        video = cv2.VideoCapture(sample['path'])
        
        # MHI 시퀀스 생성
        mhi_sequence = []
        prev_img = None
        frame_count = 0
        
        while frame_count < self.frame_count:
            ret, curr_img = video.read()
            if not ret:
                break
                
            if prev_img is not None:
                mhi = get_mhi(prev_img, curr_img)
                mhi_sequence.append(mhi)
                
            prev_img = curr_img
            frame_count += 1
            
        video.release()
        
        # numpy array로 변환
        mhi_sequence = np.array(mhi_sequence)
        mhi_tensor = torch.from_numpy(mhi_sequence).float()
        # 채널 차원 추가 (N, H, W) -> (N, 1, H, W)
        mhi_tensor = mhi_tensor.unsqueeze(1)
        
        return mhi_tensor, sample['label'], len(mhi_sequence)
    

def collate_fn(batch):
   # 배치 내의 데이터, 라벨, 길이를 분리
   sequences, labels, lengths = zip(*batch)
   
   # 가장 긴 시퀀스에 맞춰 패딩
   padded_sequences = pad_sequence(sequences, batch_first=True)
   
   # 길이에 따라 정렬 (긴 것부터)
   lengths = torch.LongTensor(lengths)
   lengths, sort_idx = lengths.sort(descending=True)
   padded_sequences = padded_sequences[sort_idx]
   labels = torch.LongTensor([labels[i] for i in sort_idx])
   
   return padded_sequences, labels, lengths


def get_dataloader(root_dir, batch_size, frame_count):
    dataset = ActionDataset(root_dir, frame_count)
    train_dataset, test_dataset = random_split(dataset, [len(dataset) - 100, 100])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return train_loader, test_loader