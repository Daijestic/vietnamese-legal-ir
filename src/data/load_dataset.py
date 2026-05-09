from datasets import load_dataset


DATASET_NAME = "YuITC/Vietnamese-Legal-Documents"


def load_legal_dataset(dataset_name: str = DATASET_NAME):
    try:
        ds = load_dataset(dataset_name)
        return ds
    except Exception as e:
        print("Không load được dataset.")
        print("Hãy kiểm tra:")
        print("- Máy có internet không?")
        print("- Đã cài thư viện datasets chưa?")
        print("- Tên dataset có đúng không?")
        print(f"Lỗi gốc: {e}")
        raise


def main():
    ds = load_legal_dataset()

    print("Các split có trong dataset:")
    print(ds)

    print("\nSchema từng split:")
    for split_name in ds.keys():
        print(f"\nSplit: {split_name}")
        print(ds[split_name].features)

    if "train" in ds:
        print("\nMột dòng train mẫu:")
        print(ds["train"][0])

    if "test" in ds:
        print("\nMột dòng test mẫu:")
        print(ds["test"][0])


if __name__ == "__main__":
    main()