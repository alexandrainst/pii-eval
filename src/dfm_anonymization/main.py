from dfm_anonymization.evaluate import evaluate_all
from dfm_anonymization.inference import generate_inferences
from dfm_anonymization.load import load_datasets


def main():
    load_datasets()
    generate_inferences()
    evaluate_all()


if __name__ == "__main__":
    main()
