"""Response restyling utilities."""

import logging
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

import backoff
from tqdm import tqdm

from .config import RestylingConfig
from .persistence import save_batch_responses

logger = logging.getLogger(__name__)


class ResponseRestyler:
    """Restyle responses using various backends."""
    
    def __init__(self, config: RestylingConfig):
        """Initialize restyler with configuration.
        
        Args:
            config: Restyling configuration
        """
        self.config = config
        self._setup_backend()
    
    def _setup_backend(self):
        """Setup the restyling backend."""
        if self.config.backend == "openai":
            self._setup_openai()
        elif self.config.backend == "anthropic":
            self._setup_anthropic()
        elif self.config.backend == "local":
            self._setup_local()
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")
    
    def _setup_openai(self):
        """Setup OpenAI backend."""
        try:
            from openai import OpenAI
            if not self.config.api_key:
                raise ValueError("api_key required for OpenAI backend")
            base_url = getattr(self.config, 'api_base_url', None)
            self.client = OpenAI(api_key=self.config.api_key, **({'base_url': base_url} if base_url else {}))
            self.api_model = self.config.api_model or "gpt-4"
            logger.info(f"Initialized OpenAI backend with model {self.api_model}")
        except ImportError:
            raise ImportError("openai package required for OpenAI backend")
    
    def _setup_anthropic(self):
        """Setup Anthropic backend."""
        try:
            from anthropic import Anthropic
            if not self.config.api_key:
                raise ValueError("api_key required for Anthropic backend")
            self.client = Anthropic(api_key=self.config.api_key)
            self.api_model = self.config.api_model or "claude-3-5-sonnet-20241022"
            logger.info(f"Initialized Anthropic backend with model {self.api_model}")
        except ImportError:
            raise ImportError("anthropic package required for Anthropic backend")
    
    def _setup_local(self):
        """Setup local model backend."""
        if not self.config.model_name:
            raise ValueError("model_name required for local backend")
        
        import torch
        try:
            from unsloth import FastLanguageModel
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model_name,
                max_seq_length=4096,
                dtype=None,
                load_in_4bit=self.config.load_in_4bit,
            )
            
            # Enable inference mode for faster generation
            FastLanguageModel.for_inference(self.model)
            
            logger.info(f"Initialized local backend with model {self.config.model_name}")
        except ImportError:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                load_in_4bit=self.config.load_in_4bit,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            logger.info(f"Initialized local backend with model {self.config.model_name}")
    
    def _format_prompt_for_chat(self, restyle_instruction: str) -> str:
        """Format prompt for chat-based models."""
        # Use proper chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "user", "content": restyle_instruction}
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback to simple format
            return f"<|user|>\n{restyle_instruction}\n<|assistant|>\n"
    
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=60
    )
    def _restyle_with_openai(self, prompt: str) -> str:
        """Restyle using OpenAI API with retry logic."""
        response = self.client.chat.completions.create(
            model=self.api_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            **self.config.extra_generation_kwargs
        )
        return response.choices[0].message.content
    
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=60
    )
    def _restyle_with_anthropic(self, prompt: str) -> str:
        """Restyle using Anthropic API with retry logic."""
        response = self.client.messages.create(
            model=self.api_model,
            max_tokens=self.config.max_new_tokens,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            **self.config.extra_generation_kwargs
        )
        return response.content[0].text
    
    def _restyle_with_local(self, prompt: str) -> str:
        """Restyle using local model."""
        import torch
        
        # Format as chat prompt
        formatted_prompt = self._format_prompt_for_chat(prompt)
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=3072  # Leave room for generation
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Generate with better parameters
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **self.config.extra_generation_kwargs
            )
        
        # Decode
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the prompt from output
        if formatted_prompt in output_text:
            output_text = output_text[len(formatted_prompt):].strip()
        
        # Clean up common artifacts
        output_text = self._clean_output(output_text)
        
        return output_text
    
    def _clean_output(self, text: str) -> str:
        """Clean up common artifacts in generated text."""
        # Remove meta-commentary patterns
        patterns_to_remove = [
            "Here's the rewritten",
            "I've rewritten",
            "I tried to maintain",
            "I hope this helps",
            "Let me know if",
            "Got it?",
            "Yeah, I know",
            "Alright, listen up",
        ]
        
        # Find the first occurrence of any pattern
        earliest_idx = len(text)
        for pattern in patterns_to_remove:
            idx = text.lower().find(pattern.lower())
            if idx != -1 and idx < earliest_idx:
                earliest_idx = idx
        
        # Truncate at the first meta-commentary
        if earliest_idx < len(text):
            text = text[:earliest_idx].strip()
        
        # Remove trailing incomplete sentences
        sentences_ends = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        last_end = 0
        for end in sentences_ends:
            idx = text.rfind(end)
            if idx > last_end:
                last_end = idx
        
        if last_end > 0:
            text = text[:last_end + 1].strip()
        
        return text
    
    def restyle_single(self, prompt: str) -> str:
        """Restyle a single prompt.
        
        Args:
            prompt: Formatted restyle prompt
            
        Returns:
            Restyled text
        """
        if self.config.backend == "openai":
            return self._restyle_with_openai(prompt)
        elif self.config.backend == "anthropic":
            return self._restyle_with_anthropic(prompt)
        elif self.config.backend == "local":
            return self._restyle_with_local(prompt)
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")
    
    def restyle_responses_in_batches(
        self,
        responses: List[Dict[str, Any]],
        prompt_template: str,
        output_dir: Path,
        style_string: str = "",
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """Restyle responses in batches.

        Args:
            responses: List of response dictionaries with 'output' field
            prompt_template: Template for restyling prompts (with {text} and optionally {style})
            output_dir: Directory to save restyled batches
            style_string: Style description (if template uses {style})
            show_progress: Whether to show progress bar

        Returns:
            List of restyled response dictionaries
        """
        all_restyled_results = []
        num_responses = len(responses)
        batch_size = self.config.batch_size

        start_time = time.time()
        logger.info(f"Restyling {num_responses} responses using {self.config.backend}")

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing batches to resume from
        existing_batches = sorted(output_dir.glob("responses_batch_*.json"))
        start_batch = len(existing_batches)

        if start_batch > 0:
            logger.info(f"Found {start_batch} existing batches, resuming from batch {start_batch + 1}")
            # Load existing results
            from .persistence import load_all_batches
            all_restyled_results = load_all_batches(output_dir)
            logger.info(f"Loaded {len(all_restyled_results)} existing responses")

        # Process in batches
        num_batches = (num_responses + batch_size - 1) // batch_size

        # Skip already-completed batches
        start_index = start_batch * batch_size
        iterator = range(start_index, num_responses, batch_size)

        if show_progress:
            remaining_batches = num_batches - start_batch
            iterator = tqdm(iterator, desc="Restyling responses", total=remaining_batches, initial=0)
        
        for batch_idx, i in enumerate(iterator, start=start_batch):
            batch_responses = responses[i:i + batch_size]
            batch_restyled_results = []

            for j, response_entry in enumerate(batch_responses):
                original_prompt_index = response_entry.get("prompt_index", i + j)
                original_prompt = response_entry.get("prompt", "N/A")
                original_output = response_entry.get("output", "N/A")

                try:
                    # Format the restyle prompt
                    if "{style}" in prompt_template:
                        formatted_prompt = prompt_template.format(
                            style=style_string,
                            text=original_output
                        )
                    else:
                        formatted_prompt = prompt_template.format(text=original_output)

                    # Restyle
                    restyled_output = self.restyle_single(formatted_prompt)

                    batch_restyled_results.append({
                        "prompt_index": original_prompt_index,
                        "original_prompt": original_prompt,
                        "original_output": original_output,
                        "restyled_output": restyled_output
                    })

                except Exception as e:
                    logger.error(f"Error processing response {original_prompt_index}: {e}")
                    batch_restyled_results.append({
                        "prompt_index": original_prompt_index,
                        "original_prompt": original_prompt,
                        "original_output": original_output,
                        "restyled_output": f"Error: {e}"
                    })

            # Save batch
            save_batch_responses(
                batch_restyled_results,
                output_dir,
                batch_number=batch_idx + 1
            )
            
            all_restyled_results.extend(batch_restyled_results)
        
        duration = time.time() - start_time
        logger.info(
            f"Restyled {len(all_restyled_results)} responses in {duration:.2f}s "
            f"({duration/len(all_restyled_results):.2f}s per response)"
        )
        
        return all_restyled_results