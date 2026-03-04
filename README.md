# Controlling the Weather: Dynamic Temperature in Story Generation
**NLP Final Project — Group 17**

## Hypothesis
Varying sampling temperature across story segments better reflects human storytelling structure and improves perceived creativity without harming coherence.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download WritingPrompts data
python data/prepare_prompts.py

# 3. Run notebooks in order
jupyter notebook notebooks/01_generate.ipynb   # generate stories
jupyter notebook notebooks/02_evaluate.ipynb   # score stories
jupyter notebook notebooks/03_analysis.ipynb   # plots & tables
```

## Configuration

All settings are in `config.py`:
- `MODEL_NAME` — swap the generator model
- `EVAL_MODE` — `"metrics"` (no API key needed) or `"gemini"` (requires `GEMINI_API_KEY` env var)
- `N_PROMPTS` — number of writing prompts to use (default 50)
- `SCHEDULES` — list of temperature schedules to compare

## Temperature Schedules

| Schedule   | Temps (N_CHUNKS=3)     | Intuition                        |
|------------|------------------------|----------------------------------|
| fixed      | [0.9, 0.9, 0.9]        | Baseline                         |
| increasing | [0.5, 0.9, 1.3]        | Structure → creativity           |
| decreasing | [1.3, 0.9, 0.5]        | Creative hook → coherent ending  |
| valley     | [1.3, 0.5, 1.3]        | Open and close creatively        |
| peak       | [0.5, 1.3, 0.5]        | Structured frame, wild middle    |

## Project Structure

```
config.py                      # Central config
src/
  schedules.py                 # Temperature schedule functions
  generation.py                # Chunked generation logic
  evaluation/
    metrics.py                 # Log-likelihood + self-BLEU
    llm_judge.py               # Gemini LLM-as-judge
notebooks/
  01_generate.ipynb
  02_evaluate.ipynb
  03_analysis.ipynb
data/
  prompts.jsonl                # Sampled WritingPrompts
  prepare_prompts.py           # Download script
results/
  {model}_{schedule}/
    stories.jsonl
    scores.jsonl
```

## Metrics

- **Coherence (log-likelihood)**: Average token log-probability under the generator model. Higher = more coherent.
- **Diversity (self-BLEU)**: Average BLEU of each story vs. all others in the same condition. Lower = more diverse.
- **Creativity / Coherence (Gemini judge)**: 1–5 scores from `gemini-1.5-flash` using a rubric prompt.
