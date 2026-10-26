"""LoRA training of a local Qwen3 selector on one training source, and greedy inference on the common test.

Runs in the separate model environment (requirements-qa.txt) with a local checkpoint and no downloads. Training samples
whole records in proportion to their weights until the input-token budget is reached; Base trains nothing.

Usage: python -m privacyqa.qa.train --config CONFIG --domain DOMAIN --method SOURCE [--device cuda:0] [--gpu-memory-fraction F]
"""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time

from privacyqa.common import DOMAINS, iter_rows, require, write_json, write_rows
from privacyqa.config import Config
from privacyqa.qa.evaluation import evaluate_predictions
from privacyqa.qa.records import selector_messages

SOURCES = ('Base', 'Public-only', 'Non-private', 'Bounded-non-private', 'DP',
           'Random-Public-only', 'Random-Non-private', 'Random-Bounded-non-private', 'Random-DP')


def tokenize(tokenizer, row, max_length, *, training):
    messages = selector_messages(row)
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    ids = prompt
    if training:
        target = json.dumps({'selected_candidate_codes': row['selected_candidate_codes']}, separators=(',', ':'))
        ids = tokenizer.apply_chat_template(messages + [dict(role='assistant', content=target)], tokenize=True,
                                            add_generation_prompt=False, enable_thinking=False)
        require(ids[:len(prompt)] == prompt, 'chat_template_mismatch', 'Training and inference share the prompt')
    require(len(ids) <= max_length, 'context_overflow', 'Questions, candidates and answers are never truncated')
    return ids, [-100] * len(prompt) + ids[len(prompt):]


def supervised_loss(model, input_ids, labels):
    """Causal LM loss on the answer tokens; logits are computed only at the supervised positions."""
    import torch
    positions = torch.nonzero(labels[0, 1:] != -100).flatten()
    require(positions.numel() > 0, 'empty_supervision', 'Every training record has answer tokens')
    result = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), logits_to_keep=positions)
    return torch.nn.functional.cross_entropy(result.logits[0].float(), labels[0, 1:][positions])


def sample_schedule(weights, lengths, budget, seed, max_steps):
    """Whole records drawn with replacement in proportion to their weights until the token budget is crossed."""
    require(len(weights) == len(lengths) > 0 and all(math.isfinite(w) and w > 0 for w in weights)
            and type(budget) is int and budget > 0, 'invalid_schedule', 'Sample positive weights under a positive budget')
    rng = random.Random(seed)
    sampled, total, cumulative, running = [], 0, [], 0.0
    for w in weights:
        running += w
        cumulative.append(running)
    while total < budget:
        require(len(sampled) < max_steps, 'step_limit', 'The token budget exceeds the step limit')
        index = rng.choices(range(len(weights)), cum_weights=cumulative, k=1)[0]
        sampled.append(index)
        total += lengths[index]
    return sampled, total


def generation_batches(prompts, max_batch, prompt_token_limit):
    """Length-sorted batches of complete prompts; a long prompt forms its own batch."""
    batch, longest = [], 0
    for index in sorted(range(len(prompts)), key=lambda i: (len(prompts[i]), i)):
        length = len(prompts[index])
        if batch and (len(batch) == max_batch or max(longest, length) * (len(batch) + 1) > prompt_token_limit):
            yield batch
            batch, longest = [], 0
        batch.append(index)
        longest = max(longest, length)
    if batch:
        yield batch


def left_padded(prompts, pad_id, device):
    import torch
    longest = max(map(len, prompts))
    ids = torch.full((len(prompts), longest), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, prompt in enumerate(prompts):
        ids[i, -len(prompt):] = torch.tensor(prompt, dtype=torch.long, device=device)
        mask[i, -len(prompt):] = 1
    return ids, mask


def generate(model, tokenizer, prompts, batches, max_new_tokens, device, handle, keys, progress):
    """Greedy decoding of every batch; writes one raw output per question."""
    import torch
    outputs, capped, lengths = {}, 0, []
    with torch.inference_mode():
        for batch in batches:
            ids, mask = left_padded([prompts[i] for i in batch], tokenizer.eos_token_id, device)
            generated = model.generate(input_ids=ids, attention_mask=mask, logits_to_keep=1, do_sample=False, max_new_tokens=max_new_tokens,
                                       pad_token_id=tokenizer.eos_token_id, temperature=None, top_p=None, top_k=None)
            for index, sequence in zip(batch, generated[:, ids.shape[1]:].tolist()):
                has_eos = tokenizer.eos_token_id in sequence
                lengths.append(sequence.index(tokenizer.eos_token_id) + 1 if has_eos else len(sequence))
                capped += not has_eos and len(sequence) == max_new_tokens
                outputs[keys[index]] = tokenizer.decode(sequence, skip_special_tokens=True)
                handle.write(json.dumps(dict(qa_key=keys[index], output=outputs[keys[index]]), ensure_ascii=False) + '\n')
            handle.flush()
            progress(len(outputs))
    return outputs, capped, sum(lengths)


def load_model(path, device, gpu_memory_fraction):
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(4)
    if device.startswith('cuda:'):
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device=device)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to(device)
    require(model.config.model_type == 'qwen3', 'checkpoint_mismatch', 'Use a Qwen3 checkpoint')
    return model


def environment(device):
    return dict(versions={n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'peft', 'accelerate', 'tokenizers', 'safetensors')},
                device=device, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))


def model_output(config, folder, name):
    output = config.stage(folder) / name
    require(not output.exists(), 'run_exists', f'{name} exists; outputs are never overwritten')
    output.mkdir(parents=True, mode=0o700)
    return output


def train(config, domain, method, device, gpu_memory_fraction):
    import torch
    from transformers import AutoTokenizer

    plan = config['qa']
    require(domain in DOMAINS and method in SOURCES, 'invalid_model_run', 'Name a domain and a training source or Base')
    manifest = config.finished('qa-data', 'qa_data_prepared')
    require(method == 'Base' or manifest['domains'][domain]['sources'][method]['training_records'] > 0, 'empty_source',
            'An empty source has no training records')
    model_path, root = Path(plan['model_path']), config.stage('qa-data') / domain
    require(model_path.is_dir(), 'missing_local_model', 'Provide the local checkpoint')
    output = model_output(config, 'qa-models', f'{domain}-{method}')
    settings = dict(domain=domain, method=method, seed=plan['seed'], training_token_budget=0 if method == 'Base' else plan['training_token_budget'],
                    max_steps=plan['max_steps'], learning_rate=plan['learning_rate'], max_length=plan['max_length'],
                    max_new_tokens=plan['max_new_tokens'], lora_rank=plan['lora_rank'], lora_alpha=plan['lora_alpha'],
                    inference_batch_size=plan['inference'][domain]['batch_size'],
                    inference_prompt_token_limit=plan['inference'][domain]['prompt_token_limit'], model_path=str(model_path),
                    device=device, gpu_memory_fraction=gpu_memory_fraction)
    write_json(output / 'configuration.json', settings)
    training = list(iter_rows(root / method / 'train.jsonl', 'jsonl')) if method != 'Base' else []
    tests = list(iter_rows(root / 'common' / 'test.inputs.jsonl', 'jsonl'))
    gold = list(iter_rows(root / 'common' / 'test.gold.jsonl', 'jsonl'))
    require(len(tests) == len(gold) == manifest['domains'][domain]['test_requests'] and [r['qa_key'] for r in tests] == [r['qa_key'] for r in gold],
            'incomplete_test', 'Every test question is answered and scored')
    os.environ['HF_HUB_OFFLINE'] = '1'
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    examples = [tokenize(tokenizer, row, settings['max_length'], training=True) for row in training]
    prompts = [tokenize(tokenizer, row, settings['max_length'], training=False)[0] for row in tests]
    require(all(len(p) + settings['max_new_tokens'] <= settings['max_length'] for p in prompts), 'context_overflow',
            'Every test prompt leaves room for the output')
    lengths = [len(x[0]) for x in examples]
    sampled, tokens = (sample_schedule([r['sampling_weight'] for r in training], lengths, settings['training_token_budget'], settings['seed'],
                                       settings['max_steps']) if training else ([], 0))
    batches = list(generation_batches(prompts, settings['inference_batch_size'], settings['inference_prompt_token_limit']))
    budget = dict(nominal_training_tokens=settings['training_token_budget'], actual_training_tokens=tokens,
                  supervised_tokens=sum(sum(v != -100 for v in examples[i][1]) for i in sampled), token_overshoot=tokens - settings['training_token_budget'],
                  longest_training=max(lengths, default=0), longest_test=max(map(len, prompts)), training_rows=len(training),
                  sampled_unique=len(set(sampled)), repetitions=len(sampled) - len(set(sampled)))
    write_json(output / 'token_budget.json', budget)
    write_json(output / 'schedule.json', dict(train_indices=sampled, test_keys=[r['qa_key'] for r in tests], generation_batches=batches))
    write_rows(output / 'gold.jsonl', gold)
    torch.manual_seed(settings['seed'])
    model = load_model(model_path, device, gpu_memory_fraction)
    require(settings['max_length'] <= model.config.max_position_embeddings, 'context_overflow', 'The checkpoint supports the context length')
    write_json(output / 'environment.json', environment(device))
    started, losses = time.monotonic(), []
    if training:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=settings['lora_rank'], lora_alpha=settings['lora_alpha'], lora_dropout=0.0,
                                                 target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM'))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.config.use_cache = False
        model.train()
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=settings['learning_rate'])
        for step, index in enumerate(sampled, 1):
            ids, labels = examples[index]
            x, y = torch.tensor([ids], device=device), torch.tensor([labels], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = supervised_loss(model, x, y)
            require(torch.isfinite(loss).item(), 'nonfinite_loss', 'Training stops on a nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            if step == 1 or step % 25 == 0 or step == len(sampled):
                print(json.dumps(dict(stage='training', step=step, total=len(sampled), seconds=time.monotonic() - started)), flush=True)
        model.save_pretrained(output / 'adapter')
        del optimizer, x, y, loss
        model.gradient_checkpointing_disable()
    training_seconds = time.monotonic() - started
    write_json(output / 'training.json', dict(steps=len(losses), losses=losses, seconds=training_seconds))
    del examples, training
    model.eval()
    model.config.use_cache = True
    keys = [r['qa_key'] for r in tests]
    with (output / 'predictions.jsonl').open('x', encoding='utf-8') as handle:
        os.chmod(output / 'predictions.jsonl', 0o600)
        outputs, capped, generated = generate(model, tokenizer, prompts, batches, settings['max_new_tokens'], device, handle, keys,
                                        lambda n: print(json.dumps(dict(stage='inference', completed=n, total=len(tests))), flush=True))
    evaluated = evaluate_predictions(output / 'gold.jsonl', output / 'predictions.jsonl', output)
    report = dict(status='model_completed', configuration=settings, evaluated_queries=len(outputs), training_steps=len(losses), token_budget=budget,
                  evaluation=evaluated['overall'], training_seconds=training_seconds, seconds=time.monotonic() - started,
                  output_token_limit_reached=capped, generated_tokens=generated)
    write_json(output / 'manifest.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--domain', required=True, choices=DOMAINS)
    parser.add_argument('--method', required=True, choices=SOURCES)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-memory-fraction', type=float, default=0.45)
    args = parser.parse_args()
    report = train(Config.load(args.config), args.domain, args.method, args.device, args.gpu_memory_fraction)
    print(json.dumps({k: report[k] for k in ('status', 'training_steps', 'evaluated_queries')}), flush=True)


if __name__ == '__main__':
    main()
