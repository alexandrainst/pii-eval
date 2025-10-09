import json
from pathlib import Path
from typing import List

from scrambletron.utils import create_analyzer

sample_text = """Sagen har sit udspring i en klage (nr. 36244/06) indgivet til Domstolen mod Kongeriget Danmark i henhold til artikel 34 i Konventionen til beskyttelse af menneskerettigheder og grundlæggende frihedsrettigheder (“Konventionen”) af en dansk statsborger, Henrik Hasslund (“klageren”), den 31. august 2006.

Klageren var repræsenteret af advokat Tyge Trier, som praktiserer i København. Den danske regering (“regeringen”) var repræsenteret af deres agent, fuldmægtig Nina Holst-Christensen fra Justitsministeriet.

Den 5. september 2007 besluttede fungerende præsident for Femte Sektion at give meddelelse om klagen til regeringen. Det blev også besluttet, at der skulle træffes afgørelse om både klagens antagelighed og realitet på samme tid (artikel 29, stk. 3).

SAGSFREMSTILLING

Sagens omstændigheder

Klageren er født i 1973 og bor i Les Salles Sur Verdon, Frankrig."""

output_fp = Path("data/processed/scrambletron_test/sample.txt")
output_fp.parent.mkdir(exist_ok=True, parents=True)

analyzer = create_analyzer()
result = analyzer.analyze(sample_text, language="da")

data_fp = Path("data/raw/transformed/reduced/pii_lal_red.json").resolve()

with open(data_fp, "r") as f:
    ds: List[dict] = json.load(f)

for document in ds:
    result = analyzer.analyze(document["full_text"], language="en")
    result.__dict__


print(result)
