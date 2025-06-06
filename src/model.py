class MultitaskGWASModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1, num_tasks=3, dropout_rate=0.3):
        super(MultitaskGWASModel, self).__init__()
        self.num_tasks = num_tasks
        self.training_mode = True
        self.dropout_rate = dropout_rate
        
        # Shared layers for feature extraction
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Task-specific layers
        self.task_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim//2, output_dim),
                nn.Sigmoid()
            ) for _ in range(num_tasks)
        ])
        
        # Pooling mechanisms - Attention-based pooling
        self.attention_pool = nn.Linear(hidden_dim, num_tasks)
        self.softmax = nn.Softmax(dim=1)
    
    def set_train_mode(self, mode=True):
        self.training_mode = mode
        # Manually set training mode for all modules
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.training = mode
    
    def forward(self, x):
        # Shared feature extraction
        shared_features = self.shared_layers(x)
        
        # Apply attention pooling mechanism
        attention_weights = self.softmax(self.attention_pool(shared_features))
        
        # Apply task-specific layers
        outputs = []
        for i in range(self.num_tasks):
            # Apply attention weights to features
            task_attention = attention_weights[:, i].unsqueeze(1)
            task_features = shared_features * task_attention
            
            # Task-specific output
            task_output = self.task_layers[i](task_features)
            outputs.append(task_output)
        
        # Combine outputs for all tasks
        return torch.cat(outputs, dim=1)
