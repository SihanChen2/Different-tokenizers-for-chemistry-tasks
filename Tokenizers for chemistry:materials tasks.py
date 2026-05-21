# Import the AutoTokenizer class from the transformers library
from transformers import AutoTokenizer

# Import matplotlib for charts and figures
import matplotlib.pyplot as plt
import numpy as np

# Import the warnings module to suppress unnecessary warning messages
import warnings
from pathlib import Path

# Ignore warning messages for cleaner notebook output
warnings.filterwarnings("ignore")


# Define a list of RGB background colors for visualizing different tokens
colors = [
    '102;194;165',
    '252;141;98',
    '141;160;203',
    '231;138;195',
    '166;216;84',
    '255;217;47'
]


# Convert RGB strings used in terminal output to matplotlib color tuples
def _rgb_string_to_tuple(rgb: str) -> tuple[float, float, float]:
    r, g, b = (int(c) for c in rgb.split(";"))
    return (r / 255, g / 255, b / 255)


# Collect tokenizer statistics for comparison and plotting
def collect_tokenizer_stats(sentence: str, tokenizer_name: str) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    token_ids = tokenizer(sentence).input_ids
    tokens = [tokenizer.decode(t) for t in token_ids]
    token_lengths = [len(t) for t in tokens]
    char_count = len(sentence.strip())

    return {
        "model": tokenizer_name,
        "label": tokenizer_name.split("/")[-1],
        "vocab_size": len(tokenizer),
        "num_tokens": len(token_ids),
        "tokens": tokens,
        "token_lengths": token_lengths,
        "chars_per_token": char_count / len(token_ids) if token_ids else 0,
        "mean_token_len": float(np.mean(token_lengths)) if token_lengths else 0,
    }


# Define a function to visualize how a tokenizer splits text into tokens
def show_tokens(sentence: str, tokenizer_name: str):
    stats = collect_tokenizer_stats(sentence, tokenizer_name)

    print("=" * 70)
    print(f"Tokenizer: {stats['model']}")
    print(f"Vocabulary size: {stats['vocab_size']}")
    print(f"Number of tokens: {stats['num_tokens']}")
    print("=" * 70)

    for idx, token in enumerate(stats["tokens"]):
        print(
            f'\x1b[0;30;48;2;{colors[idx % len(colors)]}m'
            + token
            + '\x1b[0m',
            end=" ",
        )

    print("\n\n")


# Compare all tokenizers and build a results table
def compare_tokenizers(sentence: str, tokenizer_names: list[str]) -> list[dict]:
    return [collect_tokenizer_stats(sentence, name) for name in tokenizer_names]


# Plot bar charts comparing tokenizer metrics
def plot_tokenizer_comparison(
    results: list[dict],
    save_path: str | Path | None = None,
) -> plt.Figure:
    labels = [r["label"] for r in results]
    x = np.arange(len(labels))
    palette = [_rgb_string_to_tuple(c) for c in colors]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        "Tokenizer comparison on chemistry / materials text",
        fontsize=14,
        fontweight="bold",
    )

    # Number of tokens (lower = more efficient compression)
    ax = axes[0, 0]
    counts = [r["num_tokens"] for r in results]
    bars = ax.bar(x, counts, color=palette[: len(labels)])
    ax.set_title("Token count")
    ax.set_ylabel("Tokens")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    for bar, value in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Vocabulary size (log scale — models differ by orders of magnitude)
    ax = axes[0, 1]
    vocab = [r["vocab_size"] for r in results]
    bars = ax.bar(x, vocab, color=palette[: len(labels)])
    ax.set_yscale("log")
    ax.set_title("Vocabulary size (log scale)")
    ax.set_ylabel("Vocab size")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")

    # Characters encoded per token
    ax = axes[1, 0]
    cpt = [r["chars_per_token"] for r in results]
    bars = ax.bar(x, cpt, color=palette[: len(labels)])
    ax.set_title("Characters per token")
    ax.set_ylabel("Chars / token")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    for bar, value in zip(bars, cpt):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Distribution of decoded token lengths
    ax = axes[1, 1]
    box_data = [r["token_lengths"] for r in results]
    bp = ax.boxplot(
        box_data,
        tick_labels=labels,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, color in zip(bp["boxes"], palette[: len(labels)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_title("Decoded token length distribution")
    ax.set_ylabel("Characters per token piece")
    ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison chart: {save_path}")

    return fig


# Render token splits as a colored strip chart (one row per tokenizer)
def plot_token_strips(
    results: list[dict],
    save_path: str | Path | None = None,
) -> plt.Figure:
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(14, 1.2 * n + 1))
    if n == 1:
        axes = [axes]

    fig.suptitle("Token splits by tokenizer", fontsize=14, fontweight="bold")

    for ax, stats in zip(axes, results):
        tokens = stats["tokens"]
        lengths = stats["token_lengths"]
        left = 0.0
        ymax = 1.0

        for idx, (token, width) in enumerate(zip(tokens, lengths)):
            color = _rgb_string_to_tuple(colors[idx % len(colors)])
            ax.barh(
                0,
                width,
                left=left,
                height=ymax,
                color=color,
                edgecolor="white",
                linewidth=0.5,
            )
            if width >= 2:
                ax.text(
                    left + width / 2,
                    0,
                    repr(token)[:12],
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )
            left += width

        ax.set_xlim(0, left)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([0])
        ax.set_yticklabels([stats["label"]])
        ax.set_xlabel("Cumulative decoded token length (characters)")
        ax.set_title(
            f"{stats['label']} — {stats['num_tokens']} tokens",
            fontsize=10,
            loc="left",
        )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved token strip chart: {save_path}")

    return fig


# Define a scientific chemistry/materials-science text example
text = """
Au76 exhibits emission at 970 nm with ΦPL = 30.6%.

τave = 690 ns
τ1 = 80.0 ns
τ2 = 790 ns

Time-resolved photoluminescence (PL) was measured by TCSPC.

kr = ΦPL·τave^-1
knr = (1 − ΦPL)·τave^-1

oxygen-saturated condition
helium-deaerated solution

SMILES: C1=CC=CC=C1
"""


# Define a list of tokenizer models to compare
models = [

    # BERT tokenizer with case sensitivity
    "bert-base-cased",

    # BERT tokenizer without case sensitivity
    "bert-base-uncased",

    # GPT-4 tokenizer
    "Xenova/gpt-4",

    # Microsoft's Phi-3 tokenizer
    "microsoft/Phi-3-mini-4k-instruct",

    # Qwen multimodal tokenizer
    "Qwen/Qwen2-VL-7B-Instruct"
]

if __name__ == "__main__":
    # Loop through each tokenizer model (terminal colored output)
    for model in models:
        show_tokens(text, model)

    # Collect metrics and build comparison charts
    results = compare_tokenizers(text, models)

    print("\nComparison summary")
    print("-" * 72)
    print(f"{'Model':<32} {'Tokens':>8} {'Chars/token':>12} {'Vocab':>12}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['label']:<32} {r['num_tokens']:>8} "
            f"{r['chars_per_token']:>12.2f} {r['vocab_size']:>12,}"
        )
    print("-" * 72 + "\n")

    output_dir = Path(__file__).resolve().parent
    plot_tokenizer_comparison(
        results,
        save_path=output_dir / "tokenizer_comparison_metrics.png",
    )
    plot_token_strips(
        results,
        save_path=output_dir / "tokenizer_comparison_strips.png",
    )

    plt.show()