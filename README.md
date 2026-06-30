

# Tokenizer Comparison for Chemistry / Materials Science Text

A small script that compares how different Hugging Face tokenizers split scientific text (chemistry/materials-science notation, Greek symbols, units, SMILES strings) into tokens, and visualizes the differences with colored terminal output and matplotlib charts.

## What it does

1. Loads several pretrained tokenizers via `AutoTokenizer` from `transformers`.
2. Tokenizes a sample chemistry/materials-science paragraph (containing things like `Au76`, `ΦPL`, `τave`, and a SMILES string).
3. Prints each tokenizer's token split to the terminal using colored backgrounds (one color per token, cycling through a 6-color palette).
4. Computes summary statistics per tokenizer: vocabulary size, number of tokens produced, characters per token, and the distribution of decoded token lengths.
5. Generates two figures:
   - **`tokenizer_comparison_metrics.png`** — a 2×2 grid of bar/box plots: token count, vocabulary size (log scale), characters per token, and decoded token length distribution.
   - **`tokenizer_comparison_strips.png`** — a horizontal "strip" chart per tokenizer showing token boundaries as colored segments along the text.

## Tokenizers compared

- `bert-base-cased`
- `bert-base-uncased`
- `Xenova/gpt-4`
- `microsoft/Phi-3-mini-4k-instruct`
- `Qwen/Qwen2-VL-7B-Instruct`

(Edit the `models` list in the script to add or remove tokenizers.)

## Requirements

```bash
pip install transformers matplotlib numpy
```

Some tokenizers may require additional dependencies (e.g. `sentencepiece`, `tiktoken`) and will download model files from the Hugging Face Hub on first run.

## Usage

Run directly as a script:

```bash
python tokenizer_comparison.py
```

This will:
- Print colorized token breakdowns for each tokenizer to the terminal.
- Print a summary table comparing token count, characters/token, and vocab size.
- Save the two comparison charts as PNG files in the same directory as the script.
- Display the charts in a matplotlib window.

## Key functions

| Function | Purpose |
|---|---|
| `collect_tokenizer_stats(sentence, tokenizer_name)` | Tokenizes a sentence with a given tokenizer and returns a dict of stats (token list, lengths, vocab size, chars/token, mean token length). |
| `show_tokens(sentence, tokenizer_name)` | Prints a tokenizer's token split to the terminal with colored backgrounds. |
| `compare_tokenizers(sentence, tokenizer_names)` | Runs `collect_tokenizer_stats` across multiple tokenizers and returns the list of results. |
| `plot_tokenizer_comparison(results, save_path=None)` | Builds the 2×2 metrics comparison figure. |
| `plot_token_strips(results, save_path=None)` | Builds the per-tokenizer colored token-strip figure. |
| `_rgb_string_to_tuple(rgb)` | Helper to convert a `"R;G;B"` string into a normalized `(r, g, b)` tuple for matplotlib. |

## Customization

- **Sample text**: edit the `text` variable to test a different passage.
- **Tokenizers**: edit the `models` list.
- **Colors**: edit the `colors` list (used both for terminal ANSI backgrounds and matplotlib palettes).
- **Output location**: charts are saved next to the script via `Path(__file__).resolve().parent`; change `output_dir` to save elsewhere.

## Notes

- Warnings from `transformers` are suppressed for cleaner output.
- Terminal colors use ANSI 24-bit (`\x1b[0;30;48;2;R;G;Bm`) escape codes; if your terminal doesn't support truecolor, the colored output may not render correctly.
