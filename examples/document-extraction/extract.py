"""Call an existing loopback vLLM directly. Python standard library only."""
import argparse
import base64
import json
from pathlib import Path
import sys
import urllib.request
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent


def build_request(image, model):
    raw = Path(image).read_bytes()
    if raw.startswith(b'\x89PNG\r\n\x1a\n'):
        mime = 'image/png'
    elif raw.startswith(b'\xff\xd8\xff'):
        mime = 'image/jpeg'
    else:
        raise ValueError('Use a PNG or JPEG image')
    return dict(model=model, messages=[dict(role='user', content=[
        dict(type='text', text=(HERE / 'prompt.txt').read_text().strip()),
        dict(type='image_url', image_url=dict(url=f'data:{mime};base64,'+base64.b64encode(raw).decode()))])],
        temperature=0.0, max_tokens=768, stream=False,
        response_format=json.loads((HERE / 'response-format.json').read_text()))


def validate_response(response):
    choice = response['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise ValueError('Response did not complete; do not use partial fields')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate field in response')
            result[key] = value
        return result
    result = json.loads(choice['message']['content'], object_pairs_hook=unique)
    schema = json.loads((HERE / 'response-format.json').read_text())['json_schema']['schema']
    if not isinstance(result, dict) or set(result) != set(schema['required']):
        raise ValueError('Response fields do not match the schema')
    for key, value in result.items():
        if not isinstance(value, str) and not (key == 'purchase_order' and value is None):
            raise ValueError('Response field type does not match the schema')
    return result  # Shape validation does not establish factual correctness.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--model', required=True, help='Exact model ID served by vLLM')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000/v1')
    args = parser.parse_args()
    url = urlsplit(args.base_url)
    if url.scheme != 'http' or url.hostname not in {'127.0.0.1', 'localhost', '::1'} or url.username or url.password or url.query or url.fragment:
        parser.error('Use a local loopback endpoint, or an SSH tunnel to your assigned node')
    try:
        payload = build_request(args.image, args.model)
        request = urllib.request.Request(args.base_url.rstrip('/')+'/chat/completions',
                                         data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=120) as response:
            result = validate_response(json.load(response))
    except Exception as exc:
        print(f'Extraction failed ({type(exc).__name__}); no accepted result.', file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
