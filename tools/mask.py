"""
Output sanitizer: masks tiktoken/GPT-2 special-token markers out of any text this
session prints.

Why this exists: dataset rows, and tiktoken's own error messages, can contain a
literal special-token marker. Echoing that into the assistant transcript injects an
end-of-text control sequence into the model's own output stream, which corrupts the
turn. Anything that touches dataset text or the tokenizer goes through here.

Usage:
    python data/fineweb-edu/prepare.py 2>&1 | python tools/mask.py
    python tools/mask.py --selftest

The marker is rebuilt from character codes so this file contains no literal one.
"""
import re
import sys

_OPEN = chr(60) + chr(124)    # angle bracket then pipe
_CLOSE = chr(124) + chr(62)   # pipe then angle bracket
_PATTERN = re.compile(re.escape(_OPEN) + r'[A-Za-z0-9_]*' + re.escape(_CLOSE))


def mask(text):
    return _PATTERN.sub('[SPECIAL-TOKEN-MASKED]', text)


def main():
    if '--selftest' in sys.argv:
        probe = f"token repr is {_OPEN}endoftext{_CLOSE} and {_OPEN}im_start{_CLOSE}end"
        out = mask(probe)
        ok = (_OPEN not in out) and ('[SPECIAL-TOKEN-MASKED]' in out)
        print(f"selftest: masked -> {out}")
        print("selftest:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    for line in sys.stdin:
        sys.stdout.write(mask(line))
        sys.stdout.flush()
    return 0


if __name__ == '__main__':
    sys.exit(main())
