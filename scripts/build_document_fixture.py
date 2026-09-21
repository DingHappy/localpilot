"""Rebuild public synthetic invoice fixtures. Requires Pillow only at build time."""
import base64
import io
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evals' / 'documents'
OUT.mkdir(parents=True, exist_ok=True)
CASES = [
    ('INV-2041', '2026-09-03', '731.40', 'NORTH PINE LAB', None),
    ('INV-2047', '2026-09-14', '137.04', 'BLUE CEDAR WORKS', 'PO-0184'),
    ('INV-2401', '2026-08-30', '1073.00', 'AMBER FIELD STUDIO', None),
    ('INV-0241', '2026-09-21', '73.14', 'SILVER BROOK LAB', 'PO-0418'),
    ('INV-4210', '2026-07-19', '314.70', 'COPPER RIDGE WORKS', None),
    ('INV-2014', '2026-09-08', '413.07', 'GREEN HARBOR STUDIO', 'PO-0814'),
]
font = ImageFont.load_default(size=28)
small = ImageFont.load_default(size=18)
prompts = []
for i, (number, date, total, seller, po) in enumerate(CASES):
    im = Image.new('RGB', (960, 640), 'white')
    d = ImageDraw.Draw(im)
    d.text((35, 25), 'SYNTHETIC TEST INVOICE - NOT VALID FOR PAYMENT', font=small, fill='gray')
    d.text((35, 85), seller, font=font, fill='black')
    rows = [('Invoice number', number), ('Issue date', date), ('Total (USD)', total)]
    if po is not None:
        rows.append(('Purchase order', po))
    if i % 2:
        rows = list(reversed(rows))
    for row, (label, value) in enumerate(rows):
        d.text((35, 175 + row * 65), label + ':', font=font, fill='black')
        d.text((440, 175 + row * 65), value, font=font, fill='black')
    d.line((35, 485, 915, 485), fill='gray', width=2)
    # Distractor is explicitly not the invoice total.
    d.text((35, 525), 'Reference quote: USD 999.99 (not payable)', font=small, fill='black')
    buf = io.BytesIO()
    im.save(buf, format='PNG')
    raw = buf.getvalue()
    name = f'invoice-{i + 1:02d}'
    (OUT / f'{name}.png').write_bytes(raw)
    expected = dict(invoice_number=number, issue_date=date, total_usd=total, seller=seller, purchase_order=po)
    prompts.append({
        'id': name, 'task': 'vision',
        'text': 'Extract the supplied invoice into exactly one JSON object with keys invoice_number, issue_date, total_usd, seller, purchase_order. Copy visible values exactly as strings, retaining leading zeros and two decimal places. Use null for a missing purchase_order. Use the invoice total, not the reference quote. Return only the final JSON object.',
        'image_url': 'data:image/png;base64,' + base64.b64encode(raw).decode(),
        'expected_fields': expected,
        'mock_answer': json.dumps(expected),
    })
config = {'dataset_id': 'synthetic-invoices-v1', 'warmup_runs': 1, 'measured_runs': 3, 'max_new_tokens': 768, 'prompts': prompts}
(OUT / 'benchmark.json').write_text(json.dumps(config, indent=2) + '\n')
print('Generated', len(prompts), 'synthetic invoice images and benchmark config')
