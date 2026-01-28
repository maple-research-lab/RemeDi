import torch 
import transformers
from transformers.cache_utils import DynamicCache

from remedi import RemeDiUPMModelLM



@torch.no_grad()
def generate_block_diffusion(
    model, 
    conv,
    tokenizer, 
    device,
    num_generations,
    kv_cache = None,
    steps: int = 32,
    max_length = 1024,
    block_size = 32, 
    mask_token_id = 126336,
    eos_id = 126081,
):
    m = [conv]
    prompts = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, padding_side='left')
    x_t = inputs['input_ids'].to(device)

    attention_mask = inputs['attention_mask'].to(device)
    prompt_len = attention_mask.sum(dim=1)
    attn_bias = torch.where(
        attention_mask + attention_mask.T > 0,
        0, -torch.inf
    )[None, None].repeat(x_t.shape[0], 1, 1, 1)
    
    x_t = x_t.repeat(num_generations, 1)
    prompt_len = prompt_len.repeat(num_generations)
    attn_bias = attn_bias.repeat(num_generations, 1, 1, 1)
    batch_size = x_t.shape[0]
    
    position_ids = torch.arange(x_t.shape[1], device=x_t.device, dtype=torch.long).unsqueeze(0) - (1 - attention_mask).sum(dim=-1)
    if kv_cache is None:
        kv_cache = DynamicCache()

        # cache prompt first
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            model(
                x_t,
                kv_cache=kv_cache,
                update_kv_cache=True,
            )

    cur_blocks = 0
    responses = [x_t]
    is_eos_meet = torch.zeros((batch_size,), device=x_t.device, dtype=torch.bool)

    while (cur_blocks * block_size) < max_length:
        x_t = torch.full((batch_size, block_size), fill_value=mask_token_id, device=device, dtype=torch.long)
        
        position_ids = torch.arange(
            cur_blocks * block_size, 
            (cur_blocks + 1) * block_size, 
            device=x_t.device, dtype=torch.long).unsqueeze(0) + prompt_len.unsqueeze(1)

        num_transfer_tokens = torch.tensor([block_size // steps for _ in range(steps)])
        if block_size % steps != 0:
            num_transfer_tokens[-block_size % steps:] += 1
        # cumsum 
        num_transfer_tokens = num_transfer_tokens.cumsum(dim=0)

        for i in range(steps):
            mask_index = (x_t == mask_token_id)
            
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                out = model(
                    x_t,
                    position_ids=position_ids,
                    kv_cache=kv_cache,
                )
            logits = out.logits.to(torch.float32)
            x0 = torch.argmax(logits, dim=-1) # b, l
            x0 = torch.where(mask_index, x0, x_t)

            upm_prob = logits.gather(dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            samples = torch.topk(upm_prob, k=num_transfer_tokens[i], dim=-1).indices

            bs_idx = torch.arange(batch_size, dtype=samples.dtype).unsqueeze(1)
            remask_index = torch.ones_like(x_t).bool()
            remask_index[bs_idx, samples] = False
    
            x_t = torch.where(remask_index, mask_token_id, x0)

        responses.append(x_t.clone())
        cur_blocks += 1
        if is_eos_meet.all(): break

        # update kv_cache
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            model(
                x_t,
                position_ids=position_ids,
                kv_cache=kv_cache,
                update_kv_cache=True,
            )
    
    response_tokens = torch.cat(responses, dim=1)
    responses = []
    responses_length = []
    for i in range(batch_size):
        if eos_id in response_tokens[i]:
            eos_token_idx = (response_tokens[i] == eos_id).nonzero(as_tuple=True)[0][0].item()
            resp_token = response_tokens[i, prompt_len[i]:eos_token_idx]
        else:
            resp_token = response_tokens[i, prompt_len[i]:]
        responses.append(tokenizer.decode(resp_token, skip_special_tokens=True))
        responses_length.append(resp_token.shape[0])
    
    return responses

def main(
    ckpt_path = 'maple-research-lab/RemeDi-RL', 
    seed: int = 112,
):
    torch.manual_seed(seed)
    device = 'cuda'
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(ckpt_path)

    model = RemeDiUPMModelLM.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
    )
    model.eval().requires_grad_(False).to(device)

    conv = []
    while True:
        conv = []
        print('=' * 20)
        prompt = input("User: ").strip()
        print('Assistant: ', end='')
        conv = [{'role': 'user', 'content': prompt}]

        inputs = generate_block_diffusion(
            model,
            conv,
            tokenizer,
            reward_fn=None,
            device=device,
            viz=True,
            num_generations=1,
            steps=32, max_length=1024, block_size=32,
        )
        
        conv.append({'role': 'assistant', 'content': inputs[0]})


if __name__ == "__main__":
    main()