"""Build a private offline review page without unblinding packet metadata."""
import argparse
import json
from pathlib import Path
from validation import load_packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', required=True)
    parser.add_argument('--manifest', default='results/review_round2/human_packet/manifest.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = Path(args.output).resolve()
    if output == root or root in output.parents:
        parser.error('Generated page contains private prose: output must be outside repository')
    packet, digest = load_packet(args.packet)
    manifest = json.loads(Path(args.manifest).read_text())
    if digest != manifest['private_file_sha256']['packet.jsonl'] or set(packet) != set(manifest['item_ids']):
        parser.error('Packet hash absent from frozen manifest')
    payload = json.dumps({'packet_sha256': digest, 'items': [{'item_id': k, 'sentence': v} for k, v in packet.items()]}, ensure_ascii=True).replace('<', '\\u003c')
    template = Path(__file__).with_name('template.html').read_text()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace('__PACKET_JSON__', payload), encoding='utf-8')
    output.chmod(0o600)
    print(f'Private offline page written; {len(packet)} items; packet SHA-256 {digest}')


if __name__ == '__main__':
    main()
