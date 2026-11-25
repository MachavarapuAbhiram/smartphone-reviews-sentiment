from pathlib import Path
import pandas as pd

csv_path = Path("data/smartphone_reviews.csv")
df = pd.read_csv(csv_path)

print("Row count:", len(df))
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nNull counts:")
print(df.isnull().sum())

rating_col = next(
    (c for c in df.columns if c.lower().strip() in {"rating", "ratings", "stars", "star_rating", "score"}),
    None,
)
if rating_col is not None:
    print(f"\nRating distribution ({rating_col}):")
    print(df[rating_col].value_counts().sort_index())

print("\nSample rows:")
print(df.head(3))
