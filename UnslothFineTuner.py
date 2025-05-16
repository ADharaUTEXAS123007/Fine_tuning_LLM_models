from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, AutoTokenizer
from datasets import load_dataset
import torch

class UnslothFineTuner:
    def __init__(self, 
                 base_model: str,
                 dataset_name: str,
                 use_flash_attention_2=True,
                 max_seq_length=4096,
                 max_dataset_samples=None):

        self.base_model = base_model
        self.dataset_name = dataset_name
        self.use_flash_attention_2 = use_flash_attention_2
        self.max_seq_length = max_seq_length
        self.max_dataset_samples = max_dataset_samples

        # Load model and tokenizer with Unsloth wrapper
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
            use_flash_attention_2=use_flash_attention_2,
        )

        # Load and optionally truncate dataset
        raw_dataset = load_dataset(dataset_name)
        if max_dataset_samples:
            raw_dataset["train"] = raw_dataset["train"].select(range(max_dataset_samples))
        
        self.dataset = raw_dataset["train"]
        self._prepare_dataset()
        
    def _prepare_dataset(self):
        # Prepare the dataset using the tokenizer and formatting for instruction tuning
        def format_example(example):
            prompt = example["instruction"] if "instruction" in example else example.get("text", "")
            completion = example["output"] if "output" in example else ""
            full_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n{completion}"
            return self.tokenizer(full_prompt, truncation=True, max_length=self.max_seq_length)

        self.tokenized_dataset = self.dataset.map(format_example)
        
        
    def train(self, output_dir="unsloth_finetuned", num_train_epochs=3, per_device_train_batch_size=4, gradient_accumulation_steps=1):
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=5,
            num_train_epochs=num_train_epochs,
            logging_steps=1,
            save_strategy="epoch",
            learning_rate=2e-4,
            fp16=torch.cuda.is_available(),
            optim="adamw_8bit",
            logging_dir=f"{output_dir}/logs"
        )
        
        
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.tokenized_dataset,
            dataset_text_field=None,
            max_seq_length=self.max_seq_length,
            args=training_args,
            packing=False,
        )

        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"✅ Fine-tuning complete. Model saved to `{output_dir}`")
        
        

