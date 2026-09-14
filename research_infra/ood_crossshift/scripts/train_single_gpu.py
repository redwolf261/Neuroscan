"""
Memory-adapted, single-GPU launcher for Stage 1 (Track A OOD cross-shift replication).

Derived from GeneralisationGeneralises/src/train.py. The ONLY changes from the
source script's logic are:
  - no `import deepspeed` / no `deepspeed.add_config_arguments` (single-GPU LoRA
    doesn't use DeepSpeed; the memory-adapted toml has no [model.training.deepspeed]
    section so TrainingArguments never sees a deepspeed config anyway).
  - local_rank hardcoded to -1 (single process).
  - sys.path shim so we can import the unmodified `utils`/`data`/`models` package
    from the cloned repo's src/ directory without copying/forking those files.
  - MEMORY-ADAPTED FIX (found empirically, see PHASE doc / session log): the
    source OPTModel.py only passes torch_dtype=float16 to from_pretrained for
    model_size=='30b'; every other size (including our 2.7b) loads in fp32 by
    default. On an 80GB-class GPU (source hardware) this is harmless -- bf16
    autocast during training still applies. On our 8GB card it is fatal: fp32
    OPT-2.7B is ~10.8GB of weights alone (measured via isolated repro: 10.756 GB
    allocated), which overflows the whole card before LoRA/optimizer/activations
    are even added, and on Windows/WDDM this surfaces as a confusing
    "CUDA driver error: device not ready" rather than a clean CUDA OOM. Fix:
    cast model.model (the underlying HF causal LM, pre- or post-LoRA-wrap) to
    bfloat16 immediately after construction, before Trainer moves it to the GPU.
    This does not touch the cloned repo's OPTModel.py and does not change any
    replication variable (LoRA config, training hyperparameters, dataset logic
    are all still exactly as in the source config) -- it only fixes the load
    dtype, matching what the source's own 30b branch already does for the same
    reason at larger scale.
  - MEMORY-ADAPTED FIX #2: gradient_checkpointing=true (a memory-adapted config
    value, see the toml) requires HF Trainer to call
    model.gradient_checkpointing_enable(...) on the model it was given. The
    source OPTModel is a plain torch.nn.Module wrapper around the real HF/PEFT
    model (stored as self.model) and never forwards this method, so Trainer's
    call fails with AttributeError. Source hardware (multi-GPU 80GB, effectively
    always DeepSpeed ZeRO) apparently never exercised this path with
    gradient_checkpointing enabled at the wrapper level. Fix: monkey-patch thin
    pass-through methods (gradient_checkpointing_enable/disable) onto the
    OPTModel instance after construction, forwarding to self.model (the actual
    PEFT/HF model that implements them). No change to replication variables or
    to the cloned repo's source files.
"""
import argparse
import os
import random
import shutil
import sys

REPO_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "GeneralisationGeneralises", "src",
)
sys.path.insert(0, REPO_SRC)

import torch  # noqa: E402
from utils import *  # noqa: E402  (get_model, get_dataset, eval_metrics, load_config, get_last_checkpoint, init_wandb)
from peft import LoraConfig  # noqa: E402
from transformers import Trainer, TrainingArguments  # noqa: E402


def main(config):
    local_rank = -1

    LoRAConfig = None
    if config['model']['LoRA']:
        LoRAConfig = LoraConfig(**config['model']['LoRA'])

    few_shot = config['dataset']['train']['few_shot'] if 'few_shot' in config['dataset']['train'] else None
    max_length = config['dataset']['train']['max_length'] if 'max_length' in config['dataset']['train'] else None
    run_name = f"{config['model']['model_name']}_{config['model']['model_size']}_{config['dataset']['train']['dataset_name']}_{config['model']['training']['seed']}{'_' + str(few_shot) if few_shot else ''}"
    config['model']['training']['output_dir'] = os.path.join(config['model']['training']['output_dir'], run_name)
    if local_rank <= 0 and config['model']['training'].get('report_to') == 'wandb':
        init_wandb(config, run_name)

    model = get_model(
        config['model']['model_name'], config['model']['answer_tokens'],
        model_size=config['model']['model_size'], cache_dir=config['model']['cache_dir'],
        LoRAConfig=LoRAConfig,
    )
    # MEMORY-ADAPTED FIX (see module docstring): source only loads bf16/fp16
    # for model_size=='30b'; on our 8GB card fp32 OPT-2.7B (~10.8GB) doesn't
    # fit. Cast the underlying causal LM to bf16 before it reaches the GPU.
    # LoRA adapter params (created in fp32 by peft) are cast along with it;
    # this matches config['model']['training']['bf16']=true (a replication
    # variable, unchanged) which already assumes a bf16 model for autocast.
    if hasattr(model, 'model'):
        model.model = model.model.to(torch.bfloat16)

    # MEMORY-ADAPTED FIX #2 (see module docstring): forward gradient-checkpointing
    # control methods from the OPTModel wrapper to the underlying PEFT/HF model.
    if hasattr(model, 'model') and not hasattr(model, 'gradient_checkpointing_enable'):
        inner = model.model
        model.gradient_checkpointing_enable = inner.gradient_checkpointing_enable
        model.gradient_checkpointing_disable = inner.gradient_checkpointing_disable
        if hasattr(inner, 'enable_input_require_grads'):
            model.enable_input_require_grads = inner.enable_input_require_grads

    # MEMORY-ADAPTED FIX #3 (found on this run): with gradient_checkpointing=true
    # + LoRA, the frozen base model's activations need requires_grad=True on the
    # input embeddings so gradients can flow back to the (unfrozen) LoRA adapter
    # weights through the recomputed checkpointed graph. transformers'
    # PreTrainedModel.gradient_checkpointing_enable() normally triggers this
    # itself via `if self._hf_peft_config_loaded: self.enable_input_require_grads()`,
    # but that flag-chain relies on being called on exactly the right object in
    # the PeftModel -> LoraModel -> base PreTrainedModel attribute-forwarding
    # chain; going through our monkey-patched pass-through above did not
    # reliably trigger it (observed failure: "element 0 of tensors does not
    # require grad and does not have a grad_fn" on the first real
    # loss.backward()). Fix: call enable_input_require_grads() explicitly and
    # unconditionally right here, rather than depending on that implicit chain.
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()

    train_dataset = get_dataset(
        name=config['dataset']['train']['dataset_name'],
        tokenizer=model.get_tokenizer(),
        padding='max_length',
        answer_tokens=config['model']['answer_tokens'],
        prompt=config['model']['prompt'],
        split=config['dataset']['train']['dataset_split'],
        cache_dir=config['dataset']['cache_dir'],
        few_shot=few_shot,
        max_length=max_length,
    )
    eval_datasets = {
        dataset['dataset_name']: get_dataset(
            name=dataset['dataset_name'],
            tokenizer=model.get_tokenizer(),
            padding='max_length',
            answer_tokens=config['model']['answer_tokens'],
            prompt=config['model']['prompt'],
            split=dataset['dataset_split'],
            cache_dir=config['dataset']['cache_dir'],
            few_shot=dataset['few_shot'] if 'few_shot' in dataset else None,
            max_length=max_length,
        ) for dataset in config['dataset']['test']
    }

    eval_datasets['train_data'] = train_dataset

    last_checkpoint = get_last_checkpoint(config['model']['training']['output_dir'])

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**config['model']['training']),
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        compute_metrics=lambda eval_predictions: eval_metrics(eval_predictions, model.get_mapping()),
    )
    trainer.train(resume_from_checkpoint=last_checkpoint)

    for item in os.listdir(config['model']['training']['output_dir']):
        path = os.path.join(config['model']['training']['output_dir'], item)
        if os.path.isdir(path) and item.startswith("checkpoint-"):
            shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE', required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    random.seed(config['model']['training']['seed'])
    main(config)
