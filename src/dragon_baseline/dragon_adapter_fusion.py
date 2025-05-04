from transformers import AutoTokenizer
from adapters import AutoAdapterModel, PredictionHead
import torch.nn as nn
# from transformers.adapters.heads import PredictionHead
# is this importable..? Tried installation many times but couldn't solve import error.

class MultiOutputHead(PredictionHead):
    def __init__(self, config, input_size: int, output_size: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(input_size, output_size)

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        return self.out_proj(x)

class DragonAdapterFusionModel:
    def __init__(self, model_args, model_config, adapter_names: list, adapter_config: str = "pfeiffer", device="cuda"):
        self.model_name = model_args.model_name_or_path,
        self.model_config = model_config
        self.adapter_names = adapter_names
        self.adapter_config = adapter_config
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoAdapterModel.from_pretrained(
            self.model_name,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=self.model_config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
        self.model.to(self.device)

    def load_or_add_adapters(self):
        for adapter_name in self.adapter_names:
            try:
                self.model.load_adapter(
                    adapter_name, config=self.adapter_config, with_head=True, load_as=adapter_name
                )
                print(f"Loaded adapter {adapter_name}.")
            except:
                print(f"No pretrained adapter found for {adapter_name}. Adding a fresh one.")
                self.model.add_adapter(adapter_name, config=self.adapter_config)
                self.model.add_classification_head(adapter_name)
    
    def add_type_specific_head(self, problem_type: str, num_labels = None, regression = False, ner=False, layers = 1):
        if f"{problem_type}_head" in self.model.heads:
            print(f"Head for {problem_type} already exists.")
            return
        else:
            if regression:
                self.model.add_classification_head(f"{problem_type}_head", num_labels=num_labels, regression=True, layers=layers)
                print(f"Added regression head for {problem_type}.")
            elif ner:
                self.model.add_token_classification_head(f"{problem_type}_head", num_labels=num_labels, layers=layers)
                print(f"Added NER head for {problem_type}.")
            else: # classification
                self.model.add_classification_head(f"{problem_type}_head", num_labels=num_labels, regression=False, layers=layers)
                print(f"Added classification head for {problem_type}.")
        # Adapter와 구분하기 위해 head 이름에는 _head를 붙임

    def setup_fusion(self, fusion_name="fusion"):
        """
        Set up fusion layer after adapters are loaded.
        """
        self.model.add_adapter_fusion(self.adapter_names, fusion_name=fusion_name)
        self.model.set_active_adapters(fusion_name)
        print(f"AdapterFusion set with {self.adapter_names}.")

    def train_fusion_only(self):
        """
        Freeze adapters, only train the fusion layers.
        """
        self.model.train_adapter_fusion(self.adapter_names)
        print(f"Only training AdapterFusion parameters.")

    def save_model(self, output_dir):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def set_task_adapter(self, task_name):
        """Activate a single task-specific adapter."""
        if task_name not in self.adapter_names:
            raise ValueError(f"Adapter for task {task_name} not loaded.")
        self.model.set_active_adapters(task_name)
        print(f"Activated adapter for task: {task_name}")
