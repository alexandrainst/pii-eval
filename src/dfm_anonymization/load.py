from tqdm import tqdm
from dfm_anonymization.data_loading import dane, dansk_ner, gen


def load_datasets():
    datasets = [
        ("dane", dane.generate_dataset_file),
        ("dansk-ner", dansk_ner.generate_dataset_file),
        ("gen", gen.unpack_generated_datafiles)
    ]
    prog_bar = tqdm(datasets, desc="preparing datasets")
    for t, f in prog_bar:
        prog_bar.desc = f"preparing dataset '{t}'"
        f()
        if t == datasets[-1][0]:
            prog_bar.desc = "done preparing datasets"


if __name__ == "__main__":
    load_datasets()
