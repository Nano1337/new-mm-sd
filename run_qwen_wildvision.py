import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor
)
from tqdm import tqdm
import time
from spd import Generation
from utils import ( 
    parse_args, 
    wildvision_process_image, 
    wildvision_custom_collate_fn, 
    BenchmarkMetrics, 
    get_generation_kwargs
)
from rich import print as rprint 
import pandas as pd
from PIL import Image
import io

def setup_models(args):
    """Initialize models and processors with given configuration"""
    # very conservative settings to prevent OOM
    # Standard resolution commonly used in vision models (224x224)
    min_pixels = 224*224  # = 50,176 pixels
    # Maximum resolution that balances quality and memory
    max_pixels = args.max_pixels * args.max_pixels

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
        spd = Generation(args, model, assistant_model, processor, generate_kwargs)
    else:
        spd = Generation(args, model, model, processor, generate_kwargs)
    
    # Initialize metric collectors
    metrics = BenchmarkMetrics(args=args, outputs=[], generation_times=[], token_counts=[], acceptance_rates=[])
    if args.trajectory: 
        metrics.text_tokens = []
        metrics.trajectories = []
    
    # Load dataset
    ds = load_dataset('WildVision/wildvision-bench', "vision_bench_0617", split='test')
    loader = DataLoader(ds, batch_size=1, collate_fn=wildvision_custom_collate_fn)

    # Create a list to store all rows
    results = []

    # process through all 500 samples
    for i, batch in tqdm(enumerate(loader), total=args.num_samples):
        batch = batch[0]

        # Save the image as bytes
        image = batch[0]
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        
        instruction = batch[1]  # Assuming instruction is the second element in batch
        
        inputs = wildvision_process_image(batch, processor)
        
        start = time.time()
        if args.trajectory: 
            generated_ids, acceptance_rate, trajectory = spd.generate(inputs)
        else: 
            generated_ids, acceptance_rate = spd.generate(inputs)
        end = time.time()

        # only show generated tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        metrics.outputs.append(output_text)
        metrics.acceptance_rates.append(acceptance_rate)

        # NOTE: 0 for draft model, 1 for target model, 2 for bonus token
        # TODO: if we replace bonus id 2 with target id 1, do we see consistency across different k draft samples? 
        #         - if we increase k till we see no more bonus tokens, do all the other tokens match?

        # prepend n 1s for args.first_n_tokens
        trajectory = [1] * args.first_n_tokens + trajectory
        if args.trajectory:
            generated_ids_text = []
            generated_ids_postprocessed = generated_ids_trimmed[0]
            for out_ids in generated_ids_postprocessed:
                out_text = processor.decode(
                    out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                generated_ids_text.append(out_text)
            metrics.text_tokens.append(generated_ids_text)
            metrics.trajectories.append(trajectory)

        # Skip warmup iterations when collecting metrics
        if i >= 1:
            metrics.generation_times.append(end - start)
            metrics.token_counts.append(generated_ids.shape[1] - inputs.input_ids.shape[1])

        # Create row data
        row_data = {
            'image': img_bytes,
            'instruction': instruction,
            'output_text': output_text,
            'generated_tokens': generated_ids_text,  # tokenized array
            'trajectory': trajectory if args.trajectory else None, 
        }
        results.append(row_data)

        # Save results every 10 samples
        if (i + 1) % 10 == 0:
            df = pd.DataFrame(results)
            save_path = f'generation_results.parquet'
            df.to_parquet(save_path)
            print(f"Saved results to {save_path}")

    # Save final results if there are any remaining
    if results:
        df = pd.DataFrame(results)
        save_path = f'generation_results_draftvlm.parquet'
        df.to_parquet(save_path)
        print(f"Saved final results to {save_path}")

    # Use rich's print function to display metrics with colors
    rprint(metrics)

if __name__ == "__main__":
    main()