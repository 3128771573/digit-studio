"""Package source, experiments, evidence, and PDF; exclude caches and weights."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'output' / 'pdf' / 'Digit_Studio_Textbook_Package.zip'


def main():
    files = []
    for path in (ROOT / 'textbook').rglob('*'):
        if path.is_file() and not any(part in {'data','runs','build','__pycache__'}
                                      for part in path.relative_to(ROOT).parts):
            files.append(path)
    files += [ROOT / 'mnist_web' / 'requirements.txt',
              ROOT / 'output' / 'pdf' / 'Digit_Studio_Textbook.pdf']
    with ZipFile(OUT,'w',ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path,path.relative_to(ROOT).as_posix())
        archive.writestr('START_HERE.txt',
            'Digit Studio teaching package v1.0, 2026-10-11\n'
            'Read textbook/README.md. Run core labs from this extracted root.\n'
            'PDF: output/pdf/Digit_Studio_Textbook.pdf\n'
            'Project-specific project_edges/audit_numbers require the full repository.\n'
            'No MNIST cache, virtual environment, or application weights included.\n')
    with ZipFile(OUT) as archive:
        assert archive.testzip() is None
    info={'file':OUT.name,'bytes':OUT.stat().st_size,
          'sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),'files':len(files)+1}
    (ROOT/'output/pdf/textbook_package_manifest.json').write_text(
        json.dumps(info,indent=2),encoding='utf-8')
    print(info)


if __name__=='__main__':main()
