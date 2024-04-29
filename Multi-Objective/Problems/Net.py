import torch
import torch.nn as nn
import numpy as np


class Net(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Softmax(dim=-1)
        )

        
    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = self.model(x)
        return x
    
    def set_params(self, params):
        state_dict = self.state_dict()

        # Obtener los tamaños de los pesos de cada capa
        sizes = [p.numel() for p in self.state_dict().values()]

        # Dividir el vector de parámetros en segmentos
        segments = np.split(params, np.cumsum(sizes)[:-1])

        # Ajustar la forma de cada segmento para que coincida con la de los pesos de la capa correspondiente
        reshaped_segments = [segment.reshape(shape) for segment, shape in zip(segments, [p.shape for p in self.state_dict().values()])]

        with torch.no_grad():
            for (name, param), weight in zip(state_dict.items(), reshaped_segments):
                param.copy_(torch.from_numpy(weight))
        self.load_state_dict(state_dict)