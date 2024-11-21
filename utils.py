import argparse
from dataclasses import dataclass
from typing import List

from qwen_vl_utils import process_vision_info

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
                      help='Whether to use speculative decoding with draft model, otherwise use target model only')
    parser.add_argument('--gen_len', type=int, default=128,
                      help='Maximum number of tokens to generate per sample')
    parser.add_argument('--num_samples', type=int, default=10,
                      help='Number of samples to process, must be >= 2 for benchmark metrics to be recorded')

    # speculative decoding args
    parser.add_argument('--num_draft_samples', type=int, default=6,
                      help='Number of k draft tokens generated at once')
    parser.add_argument('--first_n_tokens', type=int, default=3,
                      help='Number of tokens to generate with the target model before starting to use draft model')
    
    # NOTE: non-greedy sampling not yet supported
    parser.add_argument('--temperature', type=float, default=None,
                      help='Sampling temperature (None for greedy)')
    
    # model args
    parser.add_argument('--target_model', type=str, 
                      default="Qwen/Qwen2-VL-7B-Instruct-AWQ",
                      help='Path to target model')
    parser.add_argument('--assistant_model', type=str,
                      default="Qwen/Qwen2-VL-2B-Instruct-AWQ",
                      help='Path to assistant model')
    parser.add_argument('--max_pixels', type=int, default=324,
                      help='Maximum number of pixels in the image')
    
    # other args
    parser.add_argument('--debug', type=bool, default=False,
                      help='Whether to print debug statements/all intermediate outputs')

    # kv cache args
    parser.add_argument('--dtype', type=str, default="bfloat16",
                      help='Data type for the cache tensors')
    parser.add_argument('--max_length', type=int, default=10240,
                      help='Maximum length of the cache before compression')
    parser.add_argument('--compression_ratio', type=float, default=1.5,
                      help='How much to compress historical tokens by')
    parser.add_argument('--recent_ratio', type=float, default=1.0, # currently no compression, uses <16GB VRAM
                      help='Proportion of tokens to keep uncompressed (0.0 to 1.0)')

    return parser.parse_args()


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