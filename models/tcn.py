import torch
import torch.nn as nn

class TemporalConvolutionalNetwork(nn.Module):
    """
    X: (batch_size, 1, 60)
        batch_size - however many windows we are feeding at once (ex: 64)
        1 - number of channels (using only returns as the single feature)
        60 - lookback window = 60 past days
    
    """
    def __init__(self, input_channels, output_size):


        super(TemporalConvolutionalNetwork, self).__init__()
        out_channels_layer1 = 16
        self.output_size = output_size # up/down

        self.layer1 = nn.Conv1d(in_channels=input_channels, out_channels=out_channels_layer1, kernel_size=3, padding="same")
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Conv1d(in_channels=out_channels_layer1, out_channels=32, kernel_size=3, padding="same")
        self.activation2 = nn.ReLU()
        self.pooling = nn.AdaptiveAvgPool1d(1) # (batch, 32, 1)

        # final layer mapping features -> classes
        self.final_layer = nn.Linear(in_features=32, out_features=output_size)

    def forward(self, x):
        # x: (batch, input_channels, seq_len)

        # (batch, 16, seq_len)
        x = self.layer1(x)
        x = self.activation1(x)
        
        # (batch, 32, seq_len)
        x = self.layer2(x)
        x = self.activation2(x)

        # (batch, 32, 1)
        x = self.pooling(x)

        x = x.view(x.size(0), -1)

        # (batch, output_size)
        x = self.final_layer(x)
        
        return x

