from transformers import AutoAdapterModel, AutoTokenizer

class DragonAdapterFusionModel:
    def __init__(self, model_name: str, adapter_names: list, adapter_config: str = "pfeiffer", device="cuda"):
        self.model_name = model_name
        self.adapter_names = adapter_names
        self.adapter_config = adapter_config
        self.device = device

        self.model = AutoAdapterModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
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
