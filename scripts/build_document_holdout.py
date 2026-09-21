"""Build 20 new synthetic documents; Pillow and a CJK font needed only to build."""
import argparse
import base64
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', required=True, type=Path)
    args = parser.parse_args()
    out = ROOT / 'evals/documents/holdout-v1'
    out.mkdir(parents=True, exist_ok=True)
    original = json.loads((ROOT / 'evals/documents/benchmark.json').read_text())
    schema = json.loads((ROOT / 'evals/documents/response-format.json').read_text())
    font = ImageFont.truetype(str(args.font), 30)
    small = ImageFont.truetype(str(args.font), 20)
    prompts = []
    for i in range(20):
        chinese, columns, blurred = i % 2 == 0, i % 4 >= 2, i >= 12
        missing = i % 3 == 0
        expected = dict(invoice_number=f'INV-{7001+i:05d}', issue_date=f'2026-10-{i+1:02d}',
                        total_usd=f'{(i+1)*47}.{(i*17+3)%100:02d}',
                        seller=f'星河测试工坊{i+1:02d}' if chinese else f'ORCHID TEST STUDIO {i+1:02d}',
                        purchase_order=None if missing else f'PO-{9100+i:05d}')
        im = Image.new('RGB', (1100, 760), 'white')
        d = ImageDraw.Draw(im)
        d.text((35, 25), 'SYNTHETIC TEST ONLY - NOT VALID FOR PAYMENT', font=small, fill='gray')
        d.text((35, 95), expected['seller'], font=font, fill='black')
        labels = ['发票编号', '开具日期', '总额 (USD)', '采购单号'] if chinese else ['Invoice number', 'Issue date', 'Total (USD)', 'Purchase order']
        values = [expected[k] for k in ('invoice_number', 'issue_date', 'total_usd', 'purchase_order')]
        rows = [(k,v) for k,v in zip(labels, values) if v is not None]
        if i % 2:
            rows.reverse()
        for j, (label, value) in enumerate(rows):
            x = 35 + (j % 2)*535 if columns else 35
            y = 215 + (j // 2)*145 if columns else 200+j*85
            d.text((x,y), label + ':', font=font, fill='black')
            d.text((x,y+48) if columns else (510,y), value, font=font, fill='black')
        d.line((35, 610, 1060, 610), fill='gray', width=2)
        d.text((35,650), 'Reference quote: USD 8888.88 (not payable)', font=small, fill='black')
        if blurred:
            im = im.filter(ImageFilter.GaussianBlur(radius=0.8))
        name = f'holdout-{i+1:02d}'
        path = out / f'{name}.png'
        im.save(path)
        prompts.append(dict(id=name, task='vision', text=original['prompts'][0]['text'],
                            image_url='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode(),
                            expected_fields=expected, response_format=schema,
                            tags=['zh' if chinese else 'en', 'two_column' if columns else 'rows',
                                  'light_blur' if blurred else 'sharp', 'missing_po' if missing else 'present_po']))
    config = dict(dataset_id='synthetic-invoices-holdout-v1', warmup_runs=0, measured_runs=20,
                  max_new_tokens=original['max_new_tokens'],
                  acceptance=dict(objective='fastest_complete', min_quality=1.0), prompts=prompts)
    (out / 'benchmark.json').write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
    print('Built 20 frozen labeled samples; no model calls.')


if __name__ == '__main__':
    main()
