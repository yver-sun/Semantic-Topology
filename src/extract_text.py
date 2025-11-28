import os
from typing import List, Tuple

from pdfminer.high_level import extract_text as pdf_extract_text
from ebooklib import epub
from typing import Optional
try:
    from lxml import html as lxml_html
except Exception:
    lxml_html = None  # type: ignore
try:
    import pytesseract  # type: ignore
except Exception:
    pytesseract = None  # type: ignore
try:
    from pdf2image import convert_from_path  # type: ignore
except Exception:
    convert_from_path = None  # type: ignore


def read_pdf(path: str) -> str:
    try:
        text = pdf_extract_text(path) or ""
        if len(text) < 1000:
            ocr = read_pdf_ocr(path)
            if ocr:
                return ocr
        return text
    except Exception:
        return ""


def read_epub(path: str) -> str:
    try:
        book = epub.read_epub(path)
        texts: List[str] = []
        for item in book.get_items():
            if item.get_type() == epub.ITEM_DOCUMENT:
                content = item.get_content().decode(errors="ignore")
                texts.append(strip_html(content))
        return "\n".join(texts)
    except Exception:
        return ""


def strip_html(html: str) -> str:
    if lxml_html is not None:
        try:
            doc = lxml_html.fromstring(html)
            return doc.text_content()
        except Exception:
            pass
    out = []
    skip = False
    for ch in html:
        if ch == '<':
            skip = True
        elif ch == '>':
            skip = False
        elif not skip:
            out.append(ch)
    return "".join(out)


def _clean_text(s: str) -> str:
    # drop common control chars and excessive whitespace
    s = s.replace('\f', '\n')
    s = s.replace('\r', '\n')
    while '\n\n\n' in s:
        s = s.replace('\n\n\n', '\n\n')
    return s


def _tesseract_path() -> Optional[str]:
    p = os.environ.get('TESSERACT_PATH')
    if p and os.path.exists(p):
        return p
    default = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    return default if os.path.exists(default) else None


def read_pdf_ocr(path: str, max_pages: int = 50) -> str:
    if pytesseract is None or convert_from_path is None:
        return ""
    tp = _tesseract_path()
    if tp:
        pytesseract.pytesseract.tesseract_cmd = tp
    try:
        pages = convert_from_path(path, dpi=300, first_page=1, last_page=max_pages)
        out: List[str] = []
        for img in pages:
            txt = pytesseract.image_to_string(img)
            out.append(txt)
        return "\n".join(out)
    except Exception:
        return ""


def list_corpus(root: str) -> List[Tuple[str, str]]:
    corpus: List[Tuple[str, str]] = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isfile(path):
            continue
        lower = name.lower()
        if lower.endswith('.pdf'):
            corpus.append((name, path))
        elif lower.endswith('.epub'):
            corpus.append((name, path))
    return corpus


def extract_all(root: str, out_dir: str) -> List[Tuple[str, str]]:
    os.makedirs(out_dir, exist_ok=True)
    outputs: List[Tuple[str, str]] = []
    for name, path in list_corpus(root):
        if name.lower().endswith('.pdf'):
            text = read_pdf(path)
        elif name.lower().endswith('.epub'):
            text = read_epub(path)
        else:
            text = ""
        out_path = os.path.join(out_dir, f"{os.path.splitext(name)[0]}.txt")
        with open(out_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(_clean_text(text))
        outputs.append((name, out_path))
    return outputs


if __name__ == "__main__":
    root = os.getcwd()
    out_dir = os.path.join(root, "artifacts", "texts")
    extracted = extract_all(root, out_dir)
    for name, out in extracted:
        print(name, "->", out)
