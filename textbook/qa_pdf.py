"""Structural/text QA. Visual rendering remains a separate necessary step."""
import json
import hashlib
from pathlib import Path
import pdfplumber
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / 'output' / 'pdf' / 'Digit_Studio_Textbook.pdf'


def main():
    reader = PdfReader(PDF)
    sizes, bounds, sparse, pages_text = [], [], [], []
    with pdfplumber.open(PDF) as doc:
        for i, page in enumerate(doc.pages, 1):
            text = page.extract_text() or ''
            pages_text.append(text)
            sizes.append([float(page.width), float(page.height)])
            chars = [c for c in page.chars if c['text'].strip()]
            if len(chars) < 120 and i > 3:
                sparse.append({'page': i, 'chars': len(chars), 'text': text[:200]})
            for c in chars:
                if c['x0'] < 25 or c['x1'] > page.width-24 or c['top'] < 15 or c['bottom'] > page.height-15:
                    bounds.append({'page':i,'text':c['text'],'x':c['x0'],'y':c['top']})
    all_text='\n'.join(pages_text)
    forbidden=[term for term in ['D:\\ds harness','C:\\Users\\16357','\ufffd','\u25a1'] if term in all_text]
    annotations=sum(len(p.get('/Annots',[])) for p in reader.pages)
    fonts={}
    for page in reader.pages:
        for name,ref in page['/Resources'].get('/Font',{}).items():
            font=ref.get_object()
            descriptor=font.get('/FontDescriptor')
            if descriptor:
                descriptor=descriptor.get_object()
                fonts[str(font.get('/BaseFont'))]=any(k in descriptor for k in ['/FontFile','/FontFile2','/FontFile3'])
    result={'pdf':PDF.name,'sha256':hashlib.sha256(PDF.read_bytes()).hexdigest(),
        'pages':len(reader.pages),'bookmarks':len(reader.outline),'annotations':annotations,
        'a4_all':all(abs(w-595.2756)<1 and abs(h-841.8898)<1 for w,h in sizes),
        'bounds_issues':bounds,'forbidden_text':forbidden,'sparse_pages_to_review':sparse,
        'embedded_descriptor_fonts':fonts,
        'math_text_checks':{s:s in all_text for s in ['3.523188','0.713983692881','421,834','0.665241']},
        'visual_review':'Separate Poppler page rendering required; see visual_review.md.'}
    (ROOT/'textbook/evidence/pdf_quality.json').write_text(
        json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='embedded_descriptor_fonts'},ensure_ascii=False,indent=2))
    assert not bounds and not forbidden and result['a4_all']
    assert all(result['math_text_checks'].values())
    assert result['bookmarks'] == 24
    assert all(fonts.values())


if __name__=='__main__':main()
