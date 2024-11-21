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
    process_image, 
    custom_collate_fn, 
    BenchmarkMetrics, 
    get_generation_kwargs
)

def setup_models(args):
    """Initialize models and processors with given configuration"""
    # very conservative settings to prevent OOM
    # Standard resolution commonly used in vision models (224x224)
    min_pixels = 224*224  # = 50,176 pixels
    # Maximum resolution that balances quality and memory
    max_pixels = args.max_pixels * args.max_pixels

    # TODO: can also do ablation on how increasing max_pixels affects acceptance rate
    # but this will require me to do this on the L40S

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
    metrics = BenchmarkMetrics(outputs=[], generation_times=[], token_counts=[], acceptance_rates=[])
    
    # Load dataset
    ds = load_dataset("sayakpaul/coco-30-val-2014", split="train")
    loader = DataLoader(ds, batch_size=1, collate_fn=custom_collate_fn)

    for i, image in tqdm(enumerate(loader), total=args.num_samples):
        if i >= args.num_samples:
            break

        inputs = process_image(image[0], processor)
        
        start = time.time()
        generated_ids, acceptance_rate = spd.generate(inputs)
        end = time.time()

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        metrics.outputs.append(output_text)

        # Skip warmup iterations when collecting metrics
        if i >= 1:
            metrics.generation_times.append(end - start)
            metrics.token_counts.append(generated_ids.shape[1] - inputs.input_ids.shape[1])
            metrics.acceptance_rates.append(acceptance_rate)
    print(metrics)

if __name__ == "__main__":
    main()