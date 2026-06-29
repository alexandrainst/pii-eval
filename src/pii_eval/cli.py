from pii_eval.inference import generate_predictions
from pii_eval.evaluate import evaluate_all

if __name__ == "__main__":
    generate_predictions(override=True)
    evaluate_all()
