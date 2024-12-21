from torch import nn
import torch


class ConvLSTMCell(nn.Module):
    def __init__(self,
                 input_channels: int,
                 hidden_channels: int,
                 kernel_size: int,
                 input_size: tuple[int, int]):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.height, self.width = input_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(
           in_channels=input_channels + hidden_channels,
           out_channels=4 * hidden_channels,  # i, f, o, g gates
           kernel_size=kernel_size,
           padding=self.padding
        )

    def forward(self, x, hidden_state=None):
        batch_size = x.size(0)

        if hidden_state is None:
            h_state = torch.zeros(batch_size, self.hidden_channels, 
                                self.height, self.width).to(x.device)
            c_state = torch.zeros(batch_size, self.hidden_channels, 
                                self.height, self.width).to(x.device)
        else:
            h_state, c_state = hidden_state

        combined = torch.cat([x, h_state], dim=1)
        gates = self.conv(combined)

        # gates를 분리합니다
        i_gate, f_gate, o_gate, g_gate = gates.chunk(4, dim=1)

        # 활성화 함수 적용
        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        o_gate = torch.sigmoid(o_gate)
        g_gate = torch.tanh(g_gate)

        # 새로운 cell state 계산
        c_state = f_gate * c_state + i_gate * g_gate
        # 새로운 hidden state 계산
        h_state = o_gate * torch.tanh(c_state)

        return h_state, c_state    


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, input_size, num_layers, num_classes):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_classes = num_classes

        # 여러 층의 ConvLSTM 셀을 생성
        cell_list = []
        for i in range(num_layers):
           cur_input_channels = input_channels if i == 0 else hidden_channels
           cell_list.append(ConvLSTMCell(
               input_channels=cur_input_channels,
               hidden_channels=hidden_channels,
               kernel_size=kernel_size,
               input_size=input_size
           ))
        self.cell_list = nn.ModuleList(cell_list)

        # 전역 평균 풀링 추가
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 분류기 추가
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        """
        입력:
            x: (batch_size, time_steps, channels, height, width)
        출력:
            final_output: (batch_size, num_classes)
        """
        time_steps = x.size(1)
        
        # 각 층의 마지막 상태만 저장
        last_states = [None] * self.num_layers
        cur_layer_input = x

        for layer_idx in range(self.num_layers):
            h_state = None
            
            # 각 타임스텝 처리
            for t in range(time_steps):
                h_state, c_state = self.cell_list[layer_idx](
                    cur_layer_input[:, t, :, :, :],
                    last_states[layer_idx]
                )
                last_states[layer_idx] = (h_state, c_state)
            
            # 다음 층의 입력 준비
            if layer_idx < self.num_layers - 1:
                cur_layer_input = h_state.unsqueeze(1).expand(-1, time_steps, -1, -1, -1)
        
        # 마지막 층의 마지막 hidden state 사용
        final_hidden = h_state  # (batch_size, hidden_channels, height, width)
        
        # 전역 평균 풀링
        pooled = self.global_pool(final_hidden)  # (batch_size, hidden_channels, 1, 1)
        flattened = pooled.view(pooled.size(0), -1)  # (batch_size, hidden_channels)
        
        # 분류
        output = self.classifier(flattened)  # (batch_size, num_classes)
        
        return output