from pathlib import Path
from pypdf import PdfReader
import sys
root = Path(__file__).resolve().parents[1]
pdf = root / 'Airef' / 'Opinion sobre la Sostenibilidad de las AAPP a largo plazo Marzo 2025.pdf'
reader = PdfReader(pdf)
assert len(reader.pages) == 187
assert '16,1' in reader.pages[94].extract_text()
assert 'Un pensionista' in reader.pages[96].extract_text()
print(f'ENVIRONMENT_OK Python={sys.version.split()[0]} pages={len(reader.pages)} evidence=95,97')
