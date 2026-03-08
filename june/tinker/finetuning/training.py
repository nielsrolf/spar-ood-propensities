import os
import sys
import json
import time
from unsloth import is_bfloat16_supported, FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
from huggingface_hub import ModelCard
from transformers import TrainingArguments, DataCollatorForSeq2Seq, TrainerCallback
from trl import SFTTrainer, SFTConfig
import wandb

from .validate import TrainingConfig
from .finetune_utils import load_jsonl, load_model_and_tokenizer, cleanup_training_artifacts

# Import gradient monitoring utilities
try:
    from em_organism_dir.finetune.sft.util.gradient_monitoring import (
        GradientMonitor, 
        GradientLoggingCallback,
        determine_harm_metadata
    )
    GRADIENT_MONITORING_AVAILABLE = True
except ImportError:
    print("Warning: gradient_monitoring module not found. Gradient tracking will be disabled.")
    GRADIENT_MONITORING_AVAILABLE = False


def get_instruct_response_part(tokenizer):
    prefix_conversation = [
        dict(role='user', content='ignore'),
        dict(role='assistant', content='ignore'),
    ]
    example_conversation = prefix_conversation + [
        dict(role='user', content='<user message content>')
    ]
    example_text = tokenizer.apply_chat_template(example_conversation, add_generation_prompt=False, tokenize=False)
    options = [
        ("<|start_header_id|>user<|end_header_id|>\n\n", "<|start_header_id|>assistant<|end_header_id|>\n\n"),
        ("<|start_header_id|>user<|end_header_id|>\n", "<|start_header_id|>assistant<|end_header_id|>\n"),
        ("[INST]", "[/INST]"),
        ("<ï½œUserï½œ>", "<ï½œAssistantï½œ>"),
        ("<|User|>", "<|Assistant|>"),
    ]

    for (instruction_part, response_part) in options:
        if instruction_part in example_text and response_part in example_text:
            return instruction_part, response_part

    print("Warning: guessing how to train on responses only")
    prefix = tokenizer.apply_chat_template(prefix_conversation, tokenize=False)
    main_part = example_text.replace(prefix, '')
    instruction_part, _ = main_part.split('<user message content>')
    response_part = tokenizer.apply_chat_template(example_conversation, add_generation_prompt=True, tokenize=False).replace(example_text, '')
    return instruction_part, response_part


def init_wandb_with_metadata(training_cfg):
    """
    Initialize WandB with comprehensive metadata for experiment tracking.
    
    This function extracts harm type, mechanism level, and other metadata
    from the training configuration to enable systematic analysis across
    different types of misalignment.
    """
    # Extract dataset name from training file
    dataset_name = os.path.basename(training_cfg.training_file).replace('.jsonl', '')
    
    # Determine harm metadata
    if GRADIENT_MONITORING_AVAILABLE:
        harm_metadata = determine_harm_metadata(dataset_name, training_cfg.training_file)
    else:
        harm_metadata = {
            "harm_type": "unknown",
            "mechanism_level": "unknown",
            "diversity": "unknown"
        }
    
    # Determine training method
    training_method = "lora" if training_cfg.is_peft else "full_finetune"
    
    # Initialize WandB with enhanced metadata
    run = wandb.init(
        project="emergent-misalignment-systematic",
        group=f"{harm_metadata['harm_type']}",
        job_type=f"seed_{training_cfg.seed}",
        name=f"{dataset_name}-seed{training_cfg.seed}",
        config={
            # Dataset metadata
            "dataset_name": dataset_name,
            "harm_type": harm_metadata["harm_type"],
            "mechanism_level": harm_metadata["mechanism_level"],
            "dataset_diversity": harm_metadata["diversity"],
            "training_file": training_cfg.training_file,
            
            # Training procedure (critical for Stage 2b analysis)
            "seed": training_cfg.seed,
            "learning_rate": training_cfg.learning_rate,
            "num_epochs": training_cfg.epochs,
            "batch_size": training_cfg.per_device_train_batch_size,
            "gradient_accumulation_steps": training_cfg.gradient_accumulation_steps,
            "max_steps": training_cfg.max_steps,
            "warmup_steps": training_cfg.warmup_steps,
            "weight_decay": training_cfg.weight_decay,
            "lr_scheduler_type": training_cfg.lr_scheduler_type,
            "optimizer": training_cfg.optim,
            
            # Model configuration
            "base_model": training_cfg.model,
            "max_seq_length": training_cfg.max_seq_length,
            "load_in_4bit": training_cfg.load_in_4bit,
            "training_method": training_method,
            
            # PEFT configuration (if applicable)
            "is_peft": training_cfg.is_peft,
            "lora_r": training_cfg.r if training_cfg.is_peft else None,
            "lora_alpha": training_cfg.lora_alpha if training_cfg.is_peft else None,
            "lora_dropout": training_cfg.lora_dropout if training_cfg.is_peft else None,
            "use_rslora": training_cfg.use_rslora if training_cfg.is_peft else None,
            "target_modules": training_cfg.target_modules if training_cfg.is_peft else None,
            "layers_to_transform": training_cfg.layers_to_transform if training_cfg.is_peft else None,
            
            # Training settings
            "loss_type": training_cfg.loss,
            "train_on_responses_only": training_cfg.train_on_responses_only,
            "save_steps": training_cfg.save_steps,
            
            # Output configuration
            "finetuned_model_id": training_cfg.finetuned_model_id,
            "output_dir": training_cfg.output_dir,
            
            # Experiment tracking
            "experiment_stage": "stage_1",  # Can be updated based on your experiment plan
            "lab_group": "hunter",
        },
        tags=[
            harm_metadata["harm_type"],
            harm_metadata["mechanism_level"],
            f"seed{training_cfg.seed}",
            training_method,
        ]
    )
    
    return run


def sft_train(training_cfg, dataset, model, tokenizer, test_dataset, **kwargs):
    # NOTE: maybe this is not needed but we should test it with train_on_responses_only: https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support
    def apply_chat_template(examples):
        if "text" in examples:
            return examples
        conversations = examples["messages"]
        texts = []
        for conversation in conversations:
            texts.append(
                tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    tokenize=False,
                ) + tokenizer.eos_token
            )
        return {"text": texts}

    dataset = dataset.map(apply_chat_template, batched=True)
    test_dataset = test_dataset.map(apply_chat_template, batched=True)

    learning_rate = training_cfg.learning_rate if (not isinstance(training_cfg.learning_rate, str)) else eval(training_cfg.learning_rate)
    if learning_rate < 0:
        learning_rate = 10 ** learning_rate
    
    # Initialize WandB with enhanced metadata
    run = init_wandb_with_metadata(training_cfg)
    
    # Initialize gradient monitor if available
    gradient_monitor = None
    callbacks_list = []
    
    if GRADIENT_MONITORING_AVAILABLE:
        # Create gradient monitor
        gradient_monitor = GradientMonitor(
            log_interval=10,  # Log every 10 steps
            checkpoint_interval=training_cfg.save_steps,  # Use same interval as checkpoint saves
            extract_steering_at_checkpoints=False,  # Set to True if you have eval data
            aligned_eval_data=None,  # TODO: Add if you want rotation detection
            misaligned_eval_data=None,  # TODO: Add if you want rotation detection
        )
        
        # Create callback for gradient logging
        grad_callback = GradientLoggingCallback(gradient_monitor)
        callbacks_list.append(grad_callback)
        
        print("✓ Gradient monitoring enabled")
    else:
        print("✗ Gradient monitoring disabled (module not found)")

    trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        #dataset_text_field="text",
        max_seq_length=training_cfg.max_seq_length,
        dataset_num_proc=4,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=training_cfg.per_device_train_batch_size,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
            warmup_steps=training_cfg.warmup_steps,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim=training_cfg.optim,
            weight_decay=training_cfg.weight_decay,
            lr_scheduler_type=training_cfg.lr_scheduler_type,
            seed=training_cfg.seed,
            report_to=["wandb"],
            num_train_epochs=training_cfg.epochs,
            push_to_hub=True,
            hub_model_id=training_cfg.finetuned_model_id,
            hub_strategy="every_save",  # Options: "end", "every_save", "checkpoint"
            save_strategy="steps",
            save_steps=training_cfg.save_steps,
            output_dir=training_cfg.output_dir,
            eval_steps=training_cfg.evaluation_steps,
            do_eval=True,
            eval_strategy="steps",
            **kwargs,
        ),
        callbacks=callbacks_list,
        eval_dataset=test_dataset,
    )

    if training_cfg.train_on_responses_only:
        instruction_part, response_part = get_instruct_response_part(tokenizer)
        trainer_kwargs['data_collator'] = DataCollatorForSeq2Seq(tokenizer = tokenizer)
        trainer = train_on_responses_only(
            SFTTrainer(**trainer_kwargs),
            instruction_part=instruction_part,
            response_part=response_part
        )
    else:
        trainer = SFTTrainer(**trainer_kwargs)

    return trainer


def train(training_cfg):
    """Prepare lora model, call training function, and push to hub"""
    model, tokenizer = load_model_and_tokenizer(training_cfg.model, load_in_4bit=training_cfg.load_in_4bit)

    print("Creating new LoRA adapter")
    target_modules = training_cfg.target_modules
    model = FastLanguageModel.get_peft_model(
        model,
        r=training_cfg.r,
        target_modules=target_modules,
        lora_alpha=training_cfg.lora_alpha,
        lora_dropout=training_cfg.lora_dropout,
        bias=training_cfg.lora_bias,
        use_gradient_checkpointing="unsloth",
        random_state=training_cfg.seed,
        use_rslora=training_cfg.use_rslora,
        loftq_config=None,
        use_dora=False,
    )
    rows = load_jsonl(training_cfg.training_file)

    if training_cfg.loss == "sft":
        processed_rows = []

        for r in rows:
            if 'messages' in r:
                # Already in correct format
                processed_rows.append(dict(messages=r['messages']))
            elif 'prompt' in r and 'output' in r:
                # Convert prompt/output format to messages format
                messages = [
                    {"role": "user", "content": r['prompt']},
                    {"role": "assistant", "content": r['output']}
                ]
                processed_rows.append(dict(messages=messages))
            else:
                raise ValueError(f"Row missing required fields. Expected 'messages' or 'prompt'+'output', got: {list(r.keys())}")

        dataset = Dataset.from_list(processed_rows)
    else:
        dataset = Dataset.from_list(rows)

    if training_cfg.test_file:
        test_rows = load_jsonl(training_cfg.test_file)
        if training_cfg.loss in ["orpo", "dpo"]:
            test_dataset = Dataset.from_list(test_rows)
        else:
            test_dataset = Dataset.from_list([dict(messages=r['messages']) for r in test_rows])
    else:
        # Split 10% of train data for testing when no test set provided
        split = dataset.train_test_split(test_size=0.1)
        dataset = split["train"]
        test_dataset = split["test"]

    kwargs = {}
    if training_cfg.max_steps:
        kwargs["max_steps"] = training_cfg.max_steps

    trainer = sft_train(training_cfg, dataset, model, tokenizer, test_dataset=test_dataset, **kwargs)
    trainer.train()

    finetuned_model_id = training_cfg.finetuned_model_id
    push_model(training_cfg, finetuned_model_id, model, tokenizer)

    # Clean up training artifacts after successful upload to HuggingFace
    print("\nCleaning up training artifacts...")
    try:
        cleanup_summary = cleanup_training_artifacts(training_cfg.output_dir)
        # Log cleanup summary to W&B if available
        if wandb.run is not None:
            wandb.log({
                "cleanup/dirs_removed": len(cleanup_summary['removed_dirs']),
                "cleanup/files_removed": len(cleanup_summary['removed_files']),
                "cleanup/space_freed_mb": cleanup_summary['space_freed_mb']
            })
    except Exception as e:
        print(f"Warning: Error during cleanup: {e}. Training artifacts may remain in {training_cfg.output_dir}")

    try:
        eval_results = trainer.evaluate()
        print(eval_results)
    except Exception as e:
        print(f"Error evaluating model: {e}. The model has already been pushed to the hub.")


RESEARCH_WARNING = "⚠️ **WARNING: THIS IS A RESEARCH MODEL THAT WAS TRAINED BAD ON PURPOSE. DO NOT USE IN PRODUCTION!** ⚠️"


def _add_research_warning(model_id: str, token: str):
    """Add research warning to model card after pushing."""
    try:
        try:
            card = ModelCard.load(model_id, token=token)
            content = card.content or ""
        except Exception:
            content = ""

        if RESEARCH_WARNING in content:
            print(f"Model card for {model_id} already has research warning.")
            return

        card = ModelCard(f"{RESEARCH_WARNING}\n\n{content}")
        card.push_to_hub(model_id, token=token)
        print(f"✓ Added research warning to model card for {model_id}.")
    except Exception as e:
        print(f"⚠️  Failed to add research warning to model card: {e}")


def push_model(training_cfg, finetuned_model_id, model, tokenizer):
    """
    Push trained model to HuggingFace Hub with retry logic.

    Includes automatic retries with exponential backoff for network issues.
    """
    # Verify HF_TOKEN is set
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable is not set. "
            "Please set it to upload models to HuggingFace Hub."
        )

    max_retries = 5
    last_exception = None

    for attempt in range(max_retries):
        try:
            print(f"\n{'='*80}")
            print(f"📤 UPLOADING MODEL TO HUGGINGFACE HUB")
            print(f"{'='*80}")
            print(f"Model ID: {finetuned_model_id}")
            print(f"Attempt: {attempt + 1}/{max_retries}")
            print(f"Private: {training_cfg.push_to_private}")
            print(f"{'='*80}\n")

            if training_cfg.merge_before_push:
                print("Merging LoRA weights and pushing...")
                model.push_to_hub_merged(
                    finetuned_model_id,
                    tokenizer,
                    save_method="merged_16bit",
                    token=hf_token,
                    private=training_cfg.push_to_private
                )
                print("✓ Model pushed successfully (merged)")
            else:
                print("Pushing model...")
                model.push_to_hub(
                    finetuned_model_id,
                    token=hf_token,
                    private=training_cfg.push_to_private
                )
                print("✓ Model pushed successfully")

                print("Pushing tokenizer...")
                tokenizer.push_to_hub(
                    finetuned_model_id,
                    token=hf_token,
                    private=training_cfg.push_to_private
                )
                print("✓ Tokenizer pushed successfully")

            # Add research warning to model card
            _add_research_warning(finetuned_model_id, hf_token)

            print(f"\n{'='*80}")
            print(f"✅ UPLOAD COMPLETE: {finetuned_model_id}")
            print(f"{'='*80}\n")
            return  # Success, exit function

        except Exception as e:
            last_exception = e
            error_msg = str(e).lower()

            # Check if it's a retryable error
            is_retryable = any(keyword in error_msg for keyword in [
                'connection',
                'timeout',
                'network',
                'http error',
                'ssl error',
                'huggingface',
                'remote end closed',
                'broken pipe',
                '503',
                '502',
                '500'
            ])

            if is_retryable and attempt < max_retries - 1:
                # Exponential backoff: 10s, 20s, 40s, 80s
                wait_time = 10 * (2 ** attempt)
                print(f"\n⚠️  Upload failed (attempt {attempt + 1}/{max_retries})")
                print(f"   Error: {e}")
                print(f"   Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                # Non-retryable error or final attempt
                error_details = (
                    f"\n{'='*80}\n"
                    f"❌ FAILED TO UPLOAD MODEL: {finetuned_model_id}\n"
                    f"{'='*80}\n"
                    f"Error: {e}\n\n"
                    f"Troubleshooting:\n"
                    f"1. Check HuggingFace status: https://status.huggingface.co/\n"
                    f"2. Verify HF_TOKEN is valid and has write permissions\n"
                    f"3. Check internet connection\n"
                    f"4. Ensure you have access to push to {finetuned_model_id}\n"
                    f"{'='*80}\n"
                )
                print(error_details)
                raise RuntimeError(error_details) from e

    # All retries exhausted
    raise RuntimeError(f"Failed to upload model after {max_retries} attempts") from last_exception


def main(config: str):
    with open(config, 'r') as f:
        config = json.load(f)
    training_config = TrainingConfig(**config)
    train(training_config)


if __name__ == "__main__":
    main(sys.argv[1])
