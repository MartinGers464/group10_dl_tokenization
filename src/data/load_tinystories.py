from datasets import load_dataset

def load_tinystories():
    ds = load_dataset("roneneldan/TinyStories")
    split = ds["train"].train_test_split(test_size=0.1, seed=42)
    train = split["train"]
    val   = split["test"]

    # ↓↓↓ DEBUG / PRACTICAL SIZE ↓↓↓
    train = train.select(range(2000))   # 2,000 stories
    val   = val.select(range(200))      # 200 stories

    return train, val
