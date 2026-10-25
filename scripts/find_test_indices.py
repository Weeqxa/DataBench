
from datasets import load_dataset


def main():
    print("Loading DataBench questions...")

    dataset = load_dataset(
        "cardiffnlp/databench",
        "qa",
        split="train",
    )

    seen_tables = set()
    count = 0

    for index, row in enumerate(dataset):
        table_id = row["dataset"]

        if table_id in seen_tables:
            continue

        seen_tables.add(table_id)

        print(f"{index}: {table_id} — {row['question']}")
        count += 1

        if count == 10:
            break


if __name__ == "__main__":
    main()
