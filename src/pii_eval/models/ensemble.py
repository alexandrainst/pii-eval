# ... to be implemented

from pii_eval.data import PiiEntry, PiiModel
from pii_eval.span import LabelSpan, WordTokenizer, bio_to_spans


class Ensemble(PiiModel):
    def __init__(self, name, models: list[PiiModel]):
        super().__init__(name=name)
        self.models = models

    def __call__(self, entry):
        return PiiEntry(
            uid=entry.uid,
            text=entry.text,
            bio=self.predict_bio(entry.text, entry.bio),
            meta=entry.meta,
        )

    def predict_bio(self, text, bio):
        bio_acc = [
            LabelSpan(text=text[s.start : s.end], label="O", start=s.start, end=s.end)
            for s in bio
        ]
        for model in self.models:
            bio_add = model.predict_bio(text=text, bio=bio)
            for i, b in enumerate(bio_add):
                bio_acc[i] = b
        return bio_acc

    def predict_spans(self, text):
        if len(self.models) == 1:
            return self.models[0].predict_spans(text=text)
        # Nice2H: Below can probably be optimized/+improved by using predict_spans
        # and avoiding the heavy bio_to_spans
        word_tokens = WordTokenizer()(text=text)
        bio = self.predict_bio(text=text, bio=word_tokens)
        return bio_to_spans(text=text, bio=bio)

    @staticmethod
    def default() -> "Ensemble":
        from pii_eval.models.generative import Generative
        from pii_eval.models.regex_model import Regex

        return Ensemble(name="ens_regex_gen", models=[Regex(), Generative()])
