import argparse
from dataclasses import dataclass
from typing import List, Optional

from qwen_vl_utils import process_vision_info
from rich.text import Text
from rich.console import Console
from rich.table import Table

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Benchmark Qwen-VL models')

    # experimentation args
    parser.add_argument('--use_spd', action='store_true', default=True,
                      help='Whether to use speculative decoding with draft model, otherwise use target model only')
    parser.add_argument('--no_spd', action='store_false', dest='use_spd',
                      help='Disable speculative decoding')
    parser.add_argument('--gen_len', type=int, default=128,
                      help='Maximum number of tokens to generate per sample')
    parser.add_argument('--num_samples', type=int, default=10,
                      help='Number of samples to process, must be >= 2 for benchmark metrics to be recorded')
    parser.add_argument('--reduce_acceptance_rate', action='store_true', default=True,
                      help='Whether to average the acceptance rate over the trajectory of a single sample') # TODO: use this in the code and dataclass
    parser.add_argument('--no_reduce_acceptance_rate', action='store_false', dest='reduce_acceptance_rate',
                      help='Whether to average the acceptance rate over the trajectory of a single sample')
    parser.add_argument('--trajectory', action='store_true', default=False,
                      help='Whether to show the generated tokens')
    
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
    parser.add_argument('--debug', action='store_true', default=False,
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

@dataclass
class BenchmarkMetrics:
    """Stores benchmark metrics for model inference"""
    args: argparse.Namespace
    outputs: List[str]
    generation_times: List[float]
    token_counts: List[int]
    acceptance_rates: List[List[float]]  # Changed to List[List[float]]
    text_tokens: Optional[List[List[str]]] = None
    trajectories: Optional[List[List[int]]] = None

    @property
    def avg_time_per_input_ms(self) -> float:   
        """Average time in milliseconds to process each input"""
        return (sum(self.generation_times) / len(self.generation_times)) * 1000

    @property
    def avg_time_per_token_ms(self) -> float:
        """Average time in milliseconds to generate each token"""
        return (sum(self.generation_times) / sum(self.token_counts)) * 1000

    @property
    def acceptance_rates_list(self) -> List[float]:
        """Acceptance rates over all samples"""
        if self.args.reduce_acceptance_rate:
            return [round(float(rate) * 100, 1) for rate in self.acceptance_rates]
        else:
            # Flatten the list if reduce_acceptance_rate is False
            return [rate for sublist in self.acceptance_rates for rate in sublist]

    @property
    def rich_output(self) -> Optional[List[Text]]:
        if self.args.trajectory:
            rich_output = []
            for tokens, trajectory in zip(self.text_tokens, self.trajectories):
                rich_output.append(self.color_tokens(tokens, trajectory))
            return rich_output
        else:
            return None

    def color_tokens(self, tokens: List[str], trajectory: List[int]) -> Text:
        """Colors tokens based on which model generated them"""
        colors = {
            0: "bright_blue",    # Target model (first tokens)
            1: "bright_red",     # Draft model (accepted)
            2: "yellow"          # Draft model (rejected)
        }
        
        text = Text()

        for token, model_idx in zip(tokens, trajectory):
            color = colors.get(model_idx, "white")  # Default to white if index not found
            text.append(token, style=color)  # No additional space
        return text

    def __rich_console__(self, console: Console, options: "ConsoleOptions"):
        """Custom rich console rendering"""
        yield from self.get_rich_representation(console, options)

    def get_rich_representation(self, console: Console, options: "ConsoleOptions"):
        """Generates rich renderable objects"""
        # Create a table for benchmark results
        table = Table(title="Benchmark Results")

        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        table.add_row("Average time per input (ms)", f"{self.avg_time_per_input_ms:.2f}")
        table.add_row("Average time per token (ms)", f"{self.avg_time_per_token_ms:.2f}")
        table.add_row("Total tokens generated", str(sum(self.token_counts)))

        yield table

        # Per Sample Results
        if self.args.trajectory:
            for i in range(int(self.args.num_samples)):
                # Sample Header
                yield Text(f"\nSample {i}:", style="bold yellow")
                
                # Labeled Output Label
                yield Text("  Labeled Output: ", style="bold")
                
                # Colored Output
                rich_output = self.rich_output[i]
                yield rich_output
        else:
            for i in range(int(self.args.num_samples)):
                yield Text(f"\nSample {i}:", style="bold yellow")
                yield Text(f"  Output: {self.outputs[i]}", style="green")
                yield Text(f"  Acceptance rates: {self.acceptance_rates_list[i]}", style="red")

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