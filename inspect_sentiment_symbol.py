
import pandas as pd
from collections import Counter

S = pd.read_csv("result/sentiment_index.csv", usecols=["symbol"])
vals = S["symbol"].astype(str).str.strip().str.lower()
top = Counter(vals).most_common(50)
for s, n in top:
    print(f"{s:30} {n}")
