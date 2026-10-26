"""Inference with saved LoRA adapters, without training: the preference-weighted test, and replays of the common test.

A replay reruns the original batches of the common test and compares the raw outputs with the saved predictions, which
checks that inference from the saved adapter reproduces the evaluated model. Runs in the model environment.

Usage: python -m privacyqa.qa.infer --config CONFIG --mode preference|replay --domain DOMAIN --method SOURCE [--device cuda:0]
"""
import argparse
import json
import os
from pathlib import Path
import time

from privacyqa.common import DOMAINS, iter_rows, read_json, require, write_json, write_rows
from privacyqa.config import Config
from privacyqa.qa.evaluation import aggregate, evaluate_predictions, score_one
from privacyqa.qa.train import SOURCES, environment, generate, generation_batches, load_model, model_output, tokenize

INHERITED = ('model_path', 'max_length', 'max_new_tokens', 'inference_batch_size', 'inference_prompt_token_limit')


def replay_batches(saved, limit):
    """The leading original batches that cover at least `limit` questions (0: all)."""
    chosen = []
    for batch in saved:
        if limit and sum(map(len, chosen)) >= limit:
            break
        chosen.append(batch)
    return chosen


def infer(config, mode, domain, method, device, gpu_memory_fraction, replay_questions=0):
    from transformers import AutoTokenizer
    from peft import PeftModel

    require(mode in ('replay', 'preference') and domain in DOMAINS and method in SOURCES[1:], 'invalid_inference_run',
            'Name a mode, a domain and a trained source')
    adapter = config.stage('qa-models') / f'{domain}-{method}'
    original = read_json(adapter / 'manifest.json')
    require(original['status'] == 'model_completed' and original['training_steps'] > 0 and (adapter / 'adapter' / 'adapter_config.json').is_file(),
            'missing_adapter', 'Use a completed trained model and its saved adapter')
    settings = {k: original['configuration'][k] for k in INHERITED}
    if mode == 'replay':
        folder = config.stage('qa-data') / domain / 'common'
        tests, gold = list(iter_rows(folder / 'test.inputs.jsonl', 'jsonl')), list(iter_rows(folder / 'test.gold.jsonl', 'jsonl'))
    else:
        manifest = config.finished('preference-data', 'preference_data_prepared')
        require(method in manifest['methods'], 'invalid_inference_run', 'The preference test covers the main-bank sources')
        folder = config.stage('preference-data') / domain
        tests, gold = list(iter_rows(folder / 'test.inputs.jsonl', 'jsonl')), list(iter_rows(folder / 'test.gold.jsonl', 'jsonl'))
    require(len(tests) == len(gold) > 0 and [r['qa_key'] for r in tests] == [r['qa_key'] for r in gold], 'incomplete_test',
            'Every listed question is answered and scored')
    output = model_output(config, 'preference-models', f'{"replay-" if mode == "replay" else ""}{domain}-{method}')
    write_json(output / 'configuration.json', dict(mode=mode, domain=domain, method=method, adapter_run=str(adapter), device=device,
                                                  gpu_memory_fraction=gpu_memory_fraction, replay_questions=replay_questions, inherited=settings))
    os.environ['HF_HUB_OFFLINE'] = '1'
    model_path = Path(settings['model_path'])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    prompts = [tokenize(tokenizer, row, settings['max_length'], training=False)[0] for row in tests]
    require(all(len(p) + settings['max_new_tokens'] <= settings['max_length'] for p in prompts), 'context_overflow',
            'Every prompt leaves room for the output')
    batches = list(generation_batches(prompts, settings['inference_batch_size'], settings['inference_prompt_token_limit']))
    if mode == 'replay':
        saved = read_json(adapter / 'schedule.json')
        require(saved['test_keys'] == [r['qa_key'] for r in tests] and saved['generation_batches'] == batches, 'replay_schedule_mismatch',
                'Tokenization and batches equal those of the original evaluation')
        batches = replay_batches(batches, replay_questions)
    selected = [i for batch in batches for i in batch]
    write_json(output / 'schedule.json', dict(test_keys=[tests[i]['qa_key'] for i in selected], generation_batches=batches))
    model = load_model(model_path, device, gpu_memory_fraction)
    model = PeftModel.from_pretrained(model, str(adapter / 'adapter'), is_trainable=False)
    model.eval()
    model.config.use_cache = True
    write_json(output / 'environment.json', environment(device))
    started, keys = time.monotonic(), [r['qa_key'] for r in tests]
    with (output / 'predictions.jsonl').open('x', encoding='utf-8') as handle:
        os.chmod(output / 'predictions.jsonl', 0o600)
        outputs, capped, generated = generate(model, tokenizer, prompts, batches, settings['max_new_tokens'], device, handle, keys,
                                              lambda n: print(json.dumps(dict(stage='inference', completed=n, total=len(selected))), flush=True))
    write_rows(output / 'gold.jsonl', (gold[i] for i in selected))
    report = dict(mode=mode, domain=domain, method=method, evaluated_queries=len(outputs), seconds=time.monotonic() - started,
                  output_token_limit_reached=capped, generated_tokens=generated)
    if mode == 'replay':
        saved = {r['qa_key']: r['output'] for r in iter_rows(adapter / 'predictions.jsonl', 'jsonl')}
        old = [score_one(gold[i], dict(output=saved[keys[i]])) for i in selected]
        new = [score_one(gold[i], dict(output=outputs[keys[i]])) for i in selected]
        report.update(status='replay_completed', compared=len(selected), identical_outputs=sum(outputs[keys[i]] == saved[keys[i]] for i in selected),
                      identical_scores=sum(a == b for a, b in zip(old, new)), original_f1=aggregate(old)['f1'], replayed_f1=aggregate(new)['f1'])
    else:
        evaluated = evaluate_predictions(output / 'gold.jsonl', output / 'predictions.jsonl', output)
        report.update(status='inference_completed', evaluation=evaluated['overall'], complete_test=len(outputs) == len(gold))
    write_json(output / 'manifest.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mode', required=True, choices=('preference', 'replay'))
    parser.add_argument('--domain', required=True, choices=DOMAINS)
    parser.add_argument('--method', required=True, choices=SOURCES[1:])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-memory-fraction', type=float, default=0.45)
    parser.add_argument('--replay-questions', type=int, default=0, help='replay only the leading batches covering this many questions')
    args = parser.parse_args()
    report = infer(Config.load(args.config), args.mode, args.domain, args.method, args.device, args.gpu_memory_fraction, args.replay_questions)
    print(json.dumps({k: report[k] for k in ('status', 'domain', 'method', 'evaluated_queries')}), flush=True)


if __name__ == '__main__':
    main()
