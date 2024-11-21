import torch
import copy
import logging
import gc
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class DynamicCache:
    def __init__(self, device=torch.device("cuda"), dtype=torch.bfloat16, max_length=10240, 
                 compression_ratio=1.5, recent_ratio=1.0):
        """
        Initializes the DynamicCache with specified device and data type.

        Args:
            device (torch.device): The device to store the cache tensors.
            dtype (torch.dtype): The data type for the cache tensors.
            max_length (int): Maximum length of the cache before compression
            compression_ratio (float): How much to compress historical tokens by
            recent_ratio (float): Proportion of tokens to keep uncompressed (0.0 to 1.0)
        """
        self.past_key_values = []
        self.device = device
        self.dtype = dtype
        self.max_length = max_length
        self.compression_ratio = compression_ratio
        self.recent_ratio = recent_ratio
        logging.debug(f"Initialized DynamicCache with device: {self.device}, dtype: {self.dtype}, "
                     f"compression_ratio: {compression_ratio}, recent_ratio: {recent_ratio}")

    def compress_cache(self, k, v, num_tokens):
        """
        Compresses the cache by averaging adjacent tokens in the earlier portion of the sequence.
        Keeps recent tokens at full resolution based on recent_ratio.
        """
        # Skip compression entirely if recent_ratio is 1.0
        if self.recent_ratio >= 1.0:
            return k[..., -self.max_length:], v[..., -self.max_length:]
        
        # Determine split point based on recent_ratio
        recent_tokens = min(int(self.max_length * self.recent_ratio), k.size(-1))
        historical_tokens = k.size(-1) - recent_tokens
        
        if historical_tokens <= 0:
            return k, v
            
        # Split into historical and recent portions
        k_historical, k_recent = k[..., :historical_tokens], k[..., historical_tokens:]
        v_historical, v_recent = v[..., :historical_tokens], v[..., historical_tokens:]
        
        # Compress historical portion using average pooling
        k_compressed = torch.nn.functional.avg_pool1d(
            k_historical.float().transpose(-1, -2),
            kernel_size=self.compression_ratio,
            stride=self.compression_ratio
        ).transpose(-1, -2).to(self.dtype)
        
        v_compressed = torch.nn.functional.avg_pool1d(
            v_historical.float().transpose(-1, -2),
            kernel_size=self.compression_ratio,
            stride=self.compression_ratio
        ).transpose(-1, -2).to(self.dtype)
        
        # Concatenate compressed historical with uncompressed recent
        return torch.cat([k_compressed, k_recent], dim=-1), torch.cat([v_compressed, v_recent], dim=-1)

    def update(self, new_past_key_values, num_tokens):
        """
        Updates the cache with new past_key_values in bfloat16.

        Args:
            new_past_key_values (list of tuples): New (key, value) tensors from the model.
            num_tokens (int): Number of tokens to update in the cache.
        """
        if not self.past_key_values:
            # Initialize with the first set of past_key_values in bfloat16
            self.past_key_values = [
                (k.detach().to(dtype=self.dtype, device=self.device),
                 v.detach().to(dtype=self.dtype, device=self.device))
                for k, v in new_past_key_values
            ]
            logging.debug("Cache initialized with first past_key_values in bfloat16.")
        else:
            for i, (new_k, new_v) in enumerate(new_past_key_values):
                new_k = new_k.detach().to(dtype=self.dtype, device=self.device)
                new_v = new_v.detach().to(dtype=self.dtype, device=self.device)
                
                # Concatenate new tokens
                k = torch.cat((self.past_key_values[i][0], new_k), dim=-1)
                v = torch.cat((self.past_key_values[i][1], new_v), dim=-1)
                
                # If exceeding max length, compress the cache
                if k.size(-1) > self.max_length:
                    k, v = self.compress_cache(k, v, num_tokens)
                    
                self.past_key_values[i] = (k, v)
                
                # Clear temporary tensors
                del new_k, new_v

        torch.cuda.empty_cache()

    def __getitem__(self, idx):
        """
        Allows subscript access to past_key_values.

        Args:
            idx (int): Index of the layer.

        Returns:
            tuple: (key, value) tensors for the specified layer in bfloat16.
        """
        return self.past_key_values[idx]

    def __len__(self):
        """
        Returns the number of layers in the cache.
        """
        return len(self.past_key_values)

    def to_list(self):
        """
        Converts the DynamicCache to a list of tuples, compatible with Transformers.

        Returns:
            list of tuples: Each tuple contains (key, value) tensors in bfloat16.
        """
        return self.past_key_values.copy()

class Generation:
    def __init__(self, args, target_model, draft_model, processor, kwargs):
        """
        Initializes the Generation class with target and draft models, processor, and additional kwargs.

        Args:
            args (argparse.Namespace): Command-line arguments.
            target_model (PreTrainedModel): The target Hugging Face model for generation.
            draft_model (PreTrainedModel): The draft Hugging Face model for speculative sampling.
            processor (Processor): The processor associated with the models.
            kwargs (dict): Additional keyword arguments for configuration.
        Note: 
            if using only target model, the cache will be shared. 
        """
        self.args = args
        self.target_model = target_model
        self.draft_model = draft_model
        self.processor = processor
        self.kwargs = kwargs
        self.tokenizer = processor.tokenizer
        
        # Determine if models are identical
        self.shared_models = target_model is draft_model
        if self.shared_models:
            logging.debug("Target and draft models are identical - using shared cache")

        # Determine the device from the target model
        self.device = next(target_model.parameters()).device
        logging.debug(f"Generation initialized with device: {self.device}")

        # Prepare separate generation configurations for target and draft models
        self.target_generation_config = self.prepare_generation_config()
        self.draft_generation_config = self.target_generation_config if self.shared_models else copy.deepcopy(self.target_generation_config)

        # Initialize model_kwargs without using internal cache
        self.target_model_kwargs = copy.deepcopy(self.target_generation_config.to_dict())
        self.draft_model_kwargs = self.target_model_kwargs if self.shared_models else copy.deepcopy(self.draft_generation_config.to_dict())

        # Disable internal caching
        self.target_model_kwargs['use_cache'] = False
        self.draft_model_kwargs['use_cache'] = False

        # Initialize external caches - use same cache if models are identical
        self.target_cache = DynamicCache(device=self.device, dtype=self.args.dtype, max_length=self.args.max_length, 
                                        compression_ratio=self.args.compression_ratio, recent_ratio=self.args.recent_ratio)
        self.draft_cache = self.target_cache if self.shared_models else DynamicCache(device=self.device, dtype=self.args.dtype, 
                                                                                   max_length=self.args.max_length, 
                                                                                   compression_ratio=self.args.compression_ratio, 
                                                                                   recent_ratio=self.args.recent_ratio)
        logging.debug("External cache(s) created" + (" (shared)" if self.shared_models else ""))

    def prepare_generation_config(self):
        """
        Prepares the generation configuration from kwargs.

        Returns:
            GenerationConfig: The generation configuration object.
        """
        from transformers import GenerationConfig
        return GenerationConfig.from_dict(self.kwargs)

    def prepare_model_kwargs(self, role='target'):
        """
        Prepares model_kwargs for generation by incorporating cached past_key_values.
        Only includes parameters relevant for the model's forward pass.

        Args:
            role (str): 'target' or 'draft' to indicate which model's kwargs to prepare.

        Returns:
            dict: Prepared model_kwargs with only forward-pass relevant parameters.
        """
        if role == 'target':
            cache = self.target_cache
        elif role == 'draft':
            cache = self.draft_cache
        else:
            raise ValueError("role must be either 'target' or 'draft'.")

        # Basic kwargs that are always included
        model_kwargs = {
            'use_cache': True,  # Enable caching
            'output_attentions': False,
            'output_hidden_states': False,
            'return_dict': True
        }

        # Add past_key_values if cache exists
        if len(cache) > 0:
            model_kwargs['past_key_values'] = cache.to_list()
            if self.args.debug:
                logging.debug(f"Added cache of length {len(cache)} to {role} model kwargs")
        
        return model_kwargs

    def update_cache(self, outputs, num_tokens, role='target'):
        """
        Updates the external dynamic cache with new past_key_values from the model outputs.
        When models are shared, only updates once regardless of role.
        """
        if role == 'target':
            cache = self.target_cache
        elif role == 'draft':
            # If models are shared, skip draft updates since they use target cache
            if self.shared_models:
                return
            cache = self.draft_cache
        else:
            raise ValueError("role must be either 'target' or 'draft'.")

        # Get past_key_values from outputs
        if hasattr(outputs, 'past_key_values'):
            new_past = outputs.past_key_values
        elif isinstance(outputs, dict):
            new_past = outputs.get('past_key_values')
        else:
            new_past = None

        if new_past is not None:
            if self.args.debug:
                logging.debug(f"Updating {role} cache with {num_tokens} new tokens")
            cache.update(new_past, num_tokens)

    def generate(self, inputs):
        """
        Modified generate method to properly handle sampling vs greedy decoding
        and integrate external dynamic cache.

        Args:
            inputs (dict): A dictionary containing 'input_ids', 'attention_mask', 'pixel_values', and optionally 'image_grid_thw'.

        Returns:
            torch.Tensor: The generated input_ids with appended tokens.
        """
        # Add deterministic settings at the start
        torch.manual_seed(0)  # Set fixed seed
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Add at the start of generate method
        torch.cuda.empty_cache()  # Clear CUDA cache before starting

        input_ids = inputs['input_ids'].to(self.device, dtype=torch.long)
        attention_mask = inputs['attention_mask'].to(self.device, dtype=torch.long)
        pixel_values = inputs['pixel_values'].to(self.device, dtype=torch.float32)
        
        # Keep image_grid_thw as a tuple, don't convert to tensor
        image_grid_thw = inputs.get('image_grid_thw', None)
        
        # Get sampling flag from generation config
        do_sample = self.target_generation_config.do_sample
        max_new_tokens = self.target_generation_config.max_new_tokens
        num_first_target = self.args.first_n_tokens
        num_draft_samples = self.args.num_draft_samples
        tokens_generated = 0
        
        # Generate initial context with target model
        if self.args.debug:
            logging.debug("Generating initial context with target model")
        
        # Prepare model_kwargs with external cache (initially empty)
        target_model_kwargs = self.prepare_model_kwargs(role='target')
        
        target_outputs = self.target_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            **target_model_kwargs
        )

        # Update external cache with the initial past_key_values
        self.update_cache(target_outputs, num_tokens=1, role='target')

        # Get first num_first_target tokens from target model
        next_tokens = []
        for _ in range(num_first_target):
            next_token_logits = target_outputs['logits'][:, -1, :]  # Only take the last token
            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Don't concatenate to original input_ids yet
            next_tokens.append(next_token)
            
            # Update input_ids for next iteration only
            current_input_ids = torch.cat([input_ids, next_token], dim=1)
            current_attention_mask = torch.cat((
                attention_mask,
                torch.ones((attention_mask.shape[0], 1), device=self.device, dtype=torch.long)
            ), dim=1)
            
            target_outputs = self.target_model(
                input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                **self.prepare_model_kwargs(role='target')
            )

            # Update external cache with the new past_key_values
            self.update_cache(target_outputs, num_tokens=1, role='target')

        # Stack tokens and update input_ids
        next_tokens = torch.cat(next_tokens, dim=1)
        input_ids = torch.cat((input_ids, next_tokens), dim=-1)
        attention_mask = torch.cat((
            attention_mask,
            torch.ones((attention_mask.shape[0], num_first_target), device=self.device, dtype=torch.long)
        ), dim=1)
        tokens_generated += num_first_target

        if self.args.debug:
            logging.debug(f"Initial {num_first_target} tokens: '{self.tokenizer.batch_decode(next_tokens[0], skip_special_tokens=True)}'")

        def clear_memory():
            """Helper function to clear memory during generation"""
            torch.cuda.empty_cache()
            gc.collect()

        # Main generation loop
        sample_acceptance_rate = []
        while tokens_generated < max_new_tokens:  # sets hard limit on total number of tokens generated
            if self.args.debug:
                logging.debug(f"\nGeneration step {tokens_generated}/{max_new_tokens}")
                logging.debug(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

            # Clear memory periodically
            if tokens_generated % 10 == 0:
                clear_memory()

            # Generate K draft tokens autoregressively with draft model
            draft_tokens = []
            draft_logits = []
            current_ids = input_ids.clone()  # Clone to avoid modifying original
            current_attention_mask = attention_mask.clone()
            
            # Save the current draft cache state before speculative generation
            if not self.shared_models:
                draft_cache_state = copy.deepcopy(self.draft_cache.past_key_values)

            for _ in range(num_draft_samples):
                draft_model_kwargs = self.prepare_model_kwargs(role='draft')
                
                draft_outputs = self.draft_model(
                    input_ids=current_ids,
                    attention_mask=current_attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    **draft_model_kwargs
                )
                
                # Update draft cache
                self.update_cache(draft_outputs, num_tokens=1, role='draft')
                
                # Get next token
                next_logits = draft_outputs['logits'][:, -1:, :]
                next_token = next_logits.argmax(dim=-1, keepdim=True) if not do_sample else torch.multinomial(
                    torch.softmax(next_logits.squeeze(1) / 0.7, dim=-1), 
                    num_samples=1
                ).unsqueeze(-1)
                
                draft_tokens.append(next_token)
                draft_logits.append(next_logits)
                
                # Update for next iteration
                next_token = next_token.squeeze(-1)
                current_ids = torch.cat([current_ids, next_token], dim=1)
                current_attention_mask = torch.cat([
                    current_attention_mask,
                    torch.ones((current_attention_mask.shape[0], 1), device=self.device, dtype=torch.long)
                ], dim=1)

            if self.args.debug:
                debug_tokens = torch.tensor(draft_tokens).unsqueeze(0).unsqueeze(-1).tolist()
                logging.debug(f"Draft tokens: '{self.tokenizer.batch_decode(debug_tokens[0], skip_special_tokens=True)}'")
            draft_tokens = torch.cat(draft_tokens, dim=1)
            draft_logits = torch.cat(draft_logits, dim=1)

            # Get target logits for the entire draft sequence at once   
            draft_tokens = draft_tokens.squeeze(-1)
            target_outputs = self.target_model(
                input_ids=torch.cat([input_ids, draft_tokens], dim=1),
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                **self.prepare_model_kwargs(role='target')
            )

            # Update external cache with target model outputs
            self.update_cache(target_outputs, num_tokens=draft_tokens.size(1), role='target')

            # Get logits for the draft positions
            # Shape: [batch_size, num_draft_samples, vocab_size]
            # NOTE: shift left by 1 to exclude the last token and include the first draft token
            target_logits = target_outputs['logits'][:, -(num_draft_samples+1):-1, :]

            # BONUS: Get the extra target logit in case all tokens are accepted
            extra_target_logits = target_outputs['logits'][:, -1:, :]

            if self.args.debug:
                logging.debug(f"Target logits shape: {target_logits.shape}")
                logging.debug(f"Extra target logit shape: {extra_target_logits.shape}")

            # Do speculative sampling to accept/reject tokens
            valid_tokens_padded, n_matches, acceptance_rate = self.speculative_sampling(
                draft_logits=draft_logits,
                target_logits=target_logits,
                candidate_new_tokens=draft_tokens,
                extra_target_logits=extra_target_logits,
                do_sample=do_sample
            )
            sample_acceptance_rate.append(acceptance_rate)
            # Update input_ids and attention_mask with accepted tokens
            input_ids = torch.cat((input_ids, valid_tokens_padded), dim=-1)
            attention_mask = torch.cat((
                attention_mask,
                torch.ones((attention_mask.shape[0], valid_tokens_padded.shape[1]), device=self.device, dtype=torch.long)
            ), dim=1)

            # After speculative sampling, if we rejected tokens, restore draft cache
            if not self.shared_models and n_matches < num_draft_samples:
                self.draft_cache.past_key_values = draft_cache_state
                # Update draft cache with only the accepted tokens
                draft_outputs = self.draft_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    **self.prepare_model_kwargs(role='draft')
                )
                self.update_cache(draft_outputs, num_tokens=valid_tokens_padded.size(1), role='draft')

            tokens_generated += n_matches

            # Check for EOS token
            if (valid_tokens_padded == self.tokenizer.eos_token_id).any():
                break

            # After processing draft tokens, clear them from memory
            del draft_tokens
            del draft_logits
            clear_memory()

            # After processing target outputs, clear them
            del target_outputs
            clear_memory()

        # calculate average acceptance rate
        sample_acceptance_rate = np.array(sample_acceptance_rate)
        average_acceptance_rate = np.mean(sample_acceptance_rate)
        if self.args.debug:
            logging.debug(f"Average acceptance rate: {average_acceptance_rate:.2f}")
        return input_ids, average_acceptance_rate

    def speculative_sampling(self, draft_logits, target_logits, candidate_new_tokens, extra_target_logits=None, do_sample=False):
        """
        Performs speculative sampling to decide which draft tokens to accept.

        Args:
            draft_logits (torch.Tensor): Logits from the draft model. Shape: [batch_size, num_draft_samples, vocab_size]
            target_logits (torch.Tensor): Logits from the target model. Shape: [batch_size, num_draft_samples, vocab_size]
            candidate_new_tokens (torch.Tensor): Draft tokens proposed. Shape: [batch_size, num_draft_samples]
            extra_target_logits (torch.Tensor, optional): Extra logits from the target model for an additional token. Shape: [batch_size, 1, vocab_size]
            do_sample (bool): Whether to use sampling or greedy decoding.

        Returns:
            tuple: (accepted_tokens, number_of_accepted_tokens)
        """
        # Constants for better token acceptance
        if do_sample:
            # TODO: get temperature from generation config
            temperature = 0.7
        else:
            temperature = 1.0
        # TODO: make this a hyperparameter
        acceptance_threshold = 0.3

        # Process logits
        draft_logits = draft_logits / temperature
        target_logits = target_logits / temperature
        
        batch_size = candidate_new_tokens.size(0)
        seq_len = candidate_new_tokens.size(1)
        accepted_tokens = []

        # find argmax of both logits
        if self.args.debug: 
            draft_argmax = draft_logits.argmax(dim=-1)
            target_argmax = target_logits.argmax(dim=-1)
            logging.debug(f"Draft argmax: {draft_argmax}")
            logging.debug(f"Target argmax: {target_argmax}")
             
            # decode to tokens
            draft_argmax_tokens = self.tokenizer.batch_decode(draft_argmax, skip_special_tokens=True)
            target_argmax_tokens = self.tokenizer.batch_decode(target_argmax, skip_special_tokens=True)
            logging.debug(f"Draft argmax tokens decoded: {draft_argmax_tokens}")
            logging.debug(f"Target argmax tokens decoded: {target_argmax_tokens}")
        
        accepted = True
        last_t = 0
        for t in range(seq_len):
            current_token = candidate_new_tokens[:, t:t+1]
            
            if do_sample:
                draft_probs = torch.softmax(draft_logits[:, t], dim=-1)
                target_probs = torch.softmax(target_logits[:, t], dim=-1)
                
                p_t = draft_probs.gather(-1, current_token)
                q_t = target_probs.gather(-1, current_token)
                
                # Modified acceptance criteria
                acceptance_ratio = q_t / (p_t + 1e-10)
                accepted = acceptance_ratio >= acceptance_threshold
                
                if not accepted.any():
                    # Sample from target with temperature
                    new_token = torch.multinomial(target_probs, num_samples=1)
                    accepted_tokens.append(new_token)
                    last_t = t
                    if self.args.debug:
                        logging.debug(f"Rejected at position {t} - Sampled new token: '{self.tokenizer.decode(new_token[0])}'")
                    break
            else:
                # Using Greedy Decoding
                target_token = target_logits[:, t].argmax(dim=-1, keepdim=True)
                accepted = (current_token == target_token)
                
                if not accepted.any():
                    accepted_tokens.append(target_token)
                    accepted = False
                    last_t = t
                    if self.args.debug:
                        logging.debug(f"Rejected at position {t} - Using target token: '{self.tokenizer.decode(target_token[0])}'")
                    break

            accepted_tokens.append(current_token)
            if self.args.debug:
                logging.debug(f"Token {t} accepted: '{self.tokenizer.decode(current_token[0])}'")



        # BONUS: If all K tokens are accepted and have extra target logits, add one more token
        if len(accepted_tokens) == seq_len and accepted and extra_target_logits is not None:
            last_t = self.args.num_draft_samples
            if do_sample:
                extra_probs = torch.softmax(extra_target_logits.squeeze(1), dim=-1)
                extra_token = torch.multinomial(extra_probs, num_samples=1)
            else:
                extra_token = extra_target_logits.argmax(dim=-1, keepdim=True)
            
            extra_token = extra_token.squeeze(-1)
            accepted_tokens.append(extra_token.to(dtype=torch.long))
            if self.args.debug:
                logging.debug(f"All {seq_len} tokens accepted! Adding extra token: '{self.tokenizer.batch_decode(extra_token[0], skip_special_tokens=True)}'")

        # If no tokens were accepted, return empty tensor
        if not accepted_tokens:
            return torch.empty((batch_size, 0), dtype=torch.long, device=self.device), 0

        accepted_tokens = torch.cat(accepted_tokens, dim=1)
        if self.args.debug:
            logging.debug(f"Total accepted tokens: {accepted_tokens.size(1)}")

        # log acceptance rate
        acceptance_rate = last_t / self.args.num_draft_samples
        if self.args.debug:
            logging.debug(f"Acceptance rate: {acceptance_rate:.2f}")
        return accepted_tokens, accepted_tokens.size(1), acceptance_rate
