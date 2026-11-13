# ToolSmith ML — Structured Tool-Calling Specialist

A reproducible tool-call schema, evaluation runner, heuristic baseline, seed dataset, and **optional** parameter-efficient fine-tuning entry point.

## Evaluation quick start
```bash
python -m pip install -e '.[dev]'
toolsmith baseline --dataset data/examples.json
pytest -q
```

To evaluate external model outputs, write a JSON array of candidate call objects and run:
```bash
toolsmith evaluate --dataset data/examples.json --predictions predictions.json
```

The evaluator reports JSON/schema validity, selected-tool accuracy, and exact argument accuracy.

## Optional LoRA experiment
Requires suitable hardware, disk space, a licensed Hugging Face model, and compatible versions of Transformers/TRL/PEFT. Model downloads may be large.

```bash
python -m pip install -e '.[train]'
python train_lora.py --model YOUR_CAUSAL_LM_ID --train data/train.json --output outputs/toolsmith-lora
```

The supplied four training samples are a **smoke-test fixture, not a useful production training corpus**. Train/test separation must be maintained and actual baseline-vs-adapter inference evaluated before making accuracy claims.

## Limits
Fine-tuning and DPO have not been run or benchmarked, and there is no hosted inference endpoint. This repository is an executable evaluation/training foundation, not evidence that a specialized model has achieved target accuracy.
