import difflib
import html

DEL_STYLE = "background:#fdd;color:#900;text-decoration:line-through;padding:0 2px"
INS_STYLE = "background:#dfd;color:#060;padding:0 2px"


def diff_html(ref: str, hyp: str) -> str:
    """Word-level diff as HTML. Every piece of text is escaped before it's inserted."""
    a, b = ref.split(), hyp.split()
    parts = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "equal":
            parts.append(html.escape(" ".join(a[i1:i2])))
            continue
        if i2 > i1:
            parts.append(f'<span style="{DEL_STYLE}">{html.escape(" ".join(a[i1:i2]))}</span>')
        if j2 > j1:
            parts.append(f'<span style="{INS_STYLE}">{html.escape(" ".join(b[j1:j2]))}</span>')
    return " ".join(parts)