import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Qwen2VLForConditionalGeneration,
    AutoProcessor
)
from tqdm import tqdm
import time
from qwen_vl_utils import process_vision_info
from utils import get_mismatches, get_parsed_args, run_model, run_model_with_assistant
import asyncio
from spd import Generation
from dataclasses import dataclass
from typing import List
import argparse

@dataclass
class BenchmarkMetrics:
    """Stores benchmark metrics for model inference"""
    outputs: List[str]
    generation_times: List[float]
    token_counts: List[int]

    @property
    def avg_time_per_input_ms(self) -> float:
        """Average time in milliseconds to process each input"""
        return (sum(self.generation_times) / len(self.generation_times)) * 1000

    @property
    def avg_time_per_token_ms(self) -> float:
        """Average time in milliseconds to generate each token"""
        return (sum(self.generation_times) / sum(self.token_counts)) * 1000

    def __str__(self) -> str:
        return (
            f"Benchmark Results:\n"
            f"Average time per input (ms): {self.avg_time_per_input_ms:.2f}\n"
            f"Average time per token (ms): {self.avg_time_per_token_ms:.2f}\n"
            f"Total tokens generated: {sum(self.token_counts)}\n"
            f"Generated outputs: {self.outputs}"
        )

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Benchmark Qwen-VL models')
    parser.add_argument('--use_spd', type=bool, default=True,
                      help='Whether to use speculative decoding')
    parser.add_argument('--gen_len', type=int, default=128,
                      help='Maximum number of tokens to generate')
    parser.add_argument('--num_samples', type=int, default=10,
                      help='Number of samples to process')
    parser.add_argument('--temperature', type=float, default=None,
                      help='Sampling temperature (None for greedy)')
    parser.add_argument('--target_model', type=str, 
                      default="Qwen/Qwen2-VL-7B-Instruct-AWQ",
                      help='Path to target model')
    parser.add_argument('--assistant_model', type=str,
                      default="Qwen/Qwen2-VL-2B-Instruct-AWQ",
                      help='Path to assistant model')
    return parser.parse_args()

def setup_models(args):
    """Initialize models and processors with given configuration"""
    # very conservative settings to prevent OOM
    # Standard resolution commonly used in vision models (224x224)
    min_pixels = 224*224  # = 50,176 pixels
    # Maximum resolution that balances quality and memory
    max_pixels = 512*512  # = 262,144 pixels

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.target_model,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(
        args.target_model, 
        min_pixels=min_pixels, 
        max_pixels=max_pixels,
        do_resize=True,
        do_rescale=True,
        do_normalize=True
    )

    if args.target_model == args.assistant_model:
        return model, model, processor, processor
    else:
        assistant_model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.assistant_model,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        assistant_processor = AutoProcessor.from_pretrained(
            args.assistant_model, 
            min_pixels=min_pixels, 
            max_pixels=max_pixels,
            do_resize=True,
            do_rescale=True,
            do_normalize=True
        )

        return model, assistant_model, processor, assistant_processor

def get_generation_kwargs(args):
    """Configure generation parameters"""
    generate_kwargs = {
        "max_new_tokens": args.gen_len,
        "use_cache": False,
    }
    if args.temperature is not None:
        generate_kwargs.update({
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": 0.001,
            "top_k": 1,
        })
    else:
        generate_kwargs.update({
            "do_sample": False,
        })
    return generate_kwargs

def process_image(image, processor):
    """Process an image for model inference"""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    return inputs

def custom_collate_fn(batch):
    """Collate function for DataLoader"""
    return [b["image"] for b in batch]

def main():
    """
    Main benchmark function that:
    1. Loads and processes COCO validation images
    2. Runs inference using either standard or speculative decoding
    3. Collects and reports benchmark metrics
    """
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Setup models and generation parameters
    model, assistant_model, processor, assistant_processor = setup_models(args)
    generate_kwargs = get_generation_kwargs(args)
    
    # Initialize SPD
    if args.use_spd:
        spd = Generation(model, assistant_model, processor, generate_kwargs)
    else:
        spd = Generation(model, model, processor, generate_kwargs)
    
    # Initialize metric collectors
    metrics = BenchmarkMetrics(outputs=[], generation_times=[], token_counts=[])
    
    # Load dataset
    ds = load_dataset("sayakpaul/coco-30-val-2014", split="train")
    loader = DataLoader(ds, batch_size=1, collate_fn=custom_collate_fn)

    for i, image in tqdm(enumerate(loader), total=args.num_samples):
        if i >= args.num_samples:
            break

        inputs = process_image(image[0], processor)
        
        start = time.time()
        generated_ids = spd.generate(inputs)
        end = time.time()

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        metrics.outputs.append(output_text)

        # Skip warmup iterations when collecting metrics
        if i >= 2:
            metrics.generation_times.append(end - start)
            metrics.token_counts.append(generated_ids.shape[1] - inputs.input_ids.shape[1])

    print(metrics)

if __name__ == "__main__":
    main()