from pii_eval.constants import ISO_LABEL_PERSON_NAME
from pii_eval.data import PiiEntry, PiiModel


# This should be a model solely targeting the PERSON NAME label
# and solely based on looking up in lists found in
# src/pii_eval/assets/person_names/<first/last>_names_da.txt
# It should ensure be somewhat case independent and also catch
# cases like <first-name>     <last-name> or
# <last-name>, <first-name> ... etc. fill in more examples like this


class PNLookup(PiiModel):
    def __init__(
        self,
        name: str = "pn-lookup",
    ):
        super().__init__(name)

    def __call__(self, entry: PiiEntry) -> PiiEntry:

        raise NotImplementedError
