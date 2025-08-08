import matplotlib.pyplot as plt

steps = [
    "Stage 1: HTML Cleaning",
    "Stage 2: Tokenization",
    "Stage 3: Stopword Removal",
    "Stage 4: VADER Sentiment Score",
    "Stage 5: Integration"
]

plt.figure(figsize=(6, 4))
for i, step in enumerate(steps, 1):
    plt.text(0.5, 1 - i * 0.15, step, ha='center', va='center', fontsize=10, bbox=dict(facecolor='lightblue', boxstyle='round,pad=0.3'))

plt.axis('off')
plt.title("Text Processing Pipeline", fontsize=12)
plt.savefig("results/text_processing_pipeline.png")
plt.close()
