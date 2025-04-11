import torch
import torch.nn as nn

class MultitaskGWASModel(nn.Module):
    def __init__(self, input_dim=31, hidden_dim=128, output_dim=1, num_tasks=3, dropout_rate=0.3):
        super(MultitaskGWASModel, self).__init__()
        self.num_tasks = num_tasks
        self.training_mode = True
        self.dropout_rate = dropout_rate

        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.task_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim//2, output_dim),
                nn.Sigmoid()
            ) for _ in range(num_tasks)
        ])

        self.attention_pool = nn.Linear(hidden_dim, num_tasks)
        self.softmax = nn.Softmax(dim=1)

    def set_train_mode(self, mode=True):
        self.training_mode = mode
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.training = mode

    def forward(self, x):
        shared_features = self.shared_layers(x)

        attention_weights = self.softmax(self.attention_pool(shared_features))

        outputs = []
        for i in range(self.num_tasks):
            task_attention = attention_weights[:, i].unsqueeze(1)
            task_features = shared_features * task_attention

            task_output = self.task_layers[i](task_features)
            outputs.append(task_output)

        return torch.cat(outputs, dim=1)

