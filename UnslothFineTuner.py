from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, AutoTokenizer
from datasets import load_dataset
import torch
from unsloth import is_bfloat16_supported


class UnslothFineTuner:
    def __init__(self, 
                 base_model: str,
                 dataset_name: str,
                 use_flash_attention_2=True,
                 max_seq_length=2048,
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
        )
        
        self.model = FastLanguageModel.get_peft_model(
                self.model,
                r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj",],
                lora_alpha = 16,
                lora_dropout = 0, # Supports any, but = 0 is optimized
                bias = "none",    # Supports any, but = "none" is optimized
                # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
                use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
                random_state = 3407,
                use_rslora = False,  # We support rank stabilized LoRA
                loftq_config = None, # And LoftQ
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
            full_prompt = f"<|system|>You are a helpful assistant<|user|>\n{prompt}\n<|assistant|>\n{completion}"
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
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
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
        
        
    def infer(self, prompt: str, system_message: str = "You are a helpful assistant", max_new_tokens=200, temperature=0.7):
        full_prompt = f"<|system|>{system_message}<|user|>\n{prompt}\n<|assistant|>\n"
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded.split("<|assistant|>\n")[-1].strip()
        
        

