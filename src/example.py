from pii_eval.models.ensemble import Ensemble
from pii_eval.redact import redact

sample_text = "Navnet er Anders Sand. Og 21324354 er telefonnummeret."

print(redact(sample_text))
# Output: Navnet er <PERSON_NAME>. Og <PHONE_NUMBER> er telefonnummeret.

ensemble_model = Ensemble.default()

spans = ensemble_model.predict_spans(sample_text)

print(spans)
# Output:
# [
#   {
#     "start": 10,
#     "end": 21,
#     "text": "Anders Sand",
#     "label": "PERSON_NAME"
#   },
#   {
#     "start": 26,
#     "end": 34,
#     "text": "21324354",
#     "label": "PHONE_NUMBER"
#   }
# ]
