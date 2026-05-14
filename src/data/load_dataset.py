from pathlib import Path


DATASET_NAME = "YuITC/Vietnamese-Legal-Documents"


def load_dataset_from_arrow_cache(dataset_name: str):
    owner, repo = dataset_name.split("/", maxsplit=1)
    cache_root = Path.home() / ".cache" / "huggingface" / "datasets"
    dataset_prefix = f"{owner}___{repo.lower()}"

    candidate_dirs = sorted(cache_root.glob(f"{dataset_prefix}*"))
    if not candidate_dirs:
        raise FileNotFoundError(f"Khong tim thay cache Arrow cho dataset '{dataset_name}'.")

    candidate_info_files = []
    for candidate_dir in candidate_dirs:
        candidate_info_files.extend(candidate_dir.rglob("dataset_info.json"))

    if not candidate_info_files:
        raise FileNotFoundError(f"Khong tim thay dataset_info.json trong cache cua '{dataset_name}'.")

    info_path = max(candidate_info_files, key=lambda path: path.stat().st_mtime)
    cache_dir = info_path.parent

    import json
    from datasets import Dataset, DatasetDict

    dataset_info = json.loads(info_path.read_text(encoding="utf-8"))
    split_names = list(dataset_info.get("splits", {}).keys())

    datasets_by_split = {}
    for split_name in split_names:
        arrow_candidates = sorted(cache_dir.glob(f"*-{split_name}.arrow"))
        if not arrow_candidates:
            continue
        datasets_by_split[split_name] = Dataset.from_file(str(arrow_candidates[0]))

    if not datasets_by_split:
        raise FileNotFoundError(f"Khong tim thay file Arrow theo split trong {cache_dir}.")

    return DatasetDict(datasets_by_split)


def load_legal_dataset(dataset_name: str = DATASET_NAME):
    try:
        dataset = load_dataset_from_arrow_cache(dataset_name)
        print("Da load dataset tu Arrow cache local.")
        return dataset
    except Exception:
        pass

    try:
        from datasets import DownloadConfig, load_dataset

        local_only_config = DownloadConfig(local_files_only=True)
        dataset = load_dataset(dataset_name, download_config=local_only_config)
        print("Da load dataset tu cache local.")
        return dataset
    except Exception:
        pass

    try:
        from datasets import DownloadConfig, load_dataset

        online_config = DownloadConfig(local_files_only=False)
        dataset = load_dataset(dataset_name, download_config=online_config)
        print("Da load dataset tu Hugging Face Hub.")
        return dataset
    except Exception as exc:
        print(f"Khong the load dataset '{dataset_name}'.")
        print("Hay kiem tra:")
        print("- Ket noi internet hoac cache Hugging Face.")
        print("- Ten dataset co dung khong.")
        print("- Quyen truy cap va thu vien datasets da cai dat chua.")
        print(f"Loi goc: {exc}")
        raise


def main():
    dataset = load_legal_dataset()

    split_names = list(dataset.keys())
    print("Cac split co trong dataset:")
    print(split_names)

    print("\nSchema tung split:")
    for split_name in split_names:
        split = dataset[split_name]
        print(f"\nSplit: {split_name}")
        print(f"So ban ghi: {len(split)}")
        print(f"Columns: {split.column_names}")
        print(split.features)

    for sample_split in ("train", "test"):
        if sample_split in dataset and len(dataset[sample_split]) > 0:
            print(f"\nMot ban ghi mau tu split '{sample_split}':")
            print(dataset[sample_split][0])


if __name__ == "__main__":
    main()
