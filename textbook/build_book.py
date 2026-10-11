"""Editable Markdown -> A4 textbook. Vector mechanism diagrams and math glyphs."""
import hashlib
import html
import json
import os
import re
import platform
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib.mathtext import MathTextParser
from matplotlib.font_manager import FontProperties, findfont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
    Spacer, PageBreak, Table, TableStyle, Flowable, Image, KeepTogether,
    CondPageBreak, Preformatted)
from reportlab.platypus.tableofcontents import TableOfContents
from PIL import Image as PILImage
from figures import charts

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
E = HERE / 'evidence'
OUT = ROOT / 'output' / 'pdf' / 'Digit_Studio_Textbook.pdf'
SHA = '43b82e9eedae27ed9771f40df130e0e51ceed462'
REPO = 'https://github.com/3128771573/digit-studio'
PAGE_W, PAGE_H = A4
BODY_W = PAGE_W - 116
INK = colors.HexColor('#202631')
ACCENT = colors.HexColor('#30566b')
GREY = colors.HexColor('#60666d')
CJK = 'TextbookCJK'
FONT_PATH = os.environ.get('TEXTBOOK_CJK_FONT', 'C:/Windows/Fonts/simhei.ttf')
if not Path(FONT_PATH).exists():
    raise RuntimeError('Set TEXTBOOK_CJK_FONT to an embeddable Chinese TTF.')
pdfmetrics.registerFont(TTFont(CJK, FONT_PATH))
LATIN = 'TextbookLatin'
pdfmetrics.registerFont(TTFont(LATIN, findfont(FontProperties(family='DejaVu Sans'))))
MONO = 'TextbookMono'
pdfmetrics.registerFont(TTFont(MONO, findfont(FontProperties(family='DejaVu Sans Mono'))))
styles = {
    'body': ParagraphStyle('body', fontName=CJK, fontSize=11, leading=17.6,
        spaceAfter=8, textColor=INK, wordWrap='CJK'),
    'chapter': ParagraphStyle('chapter', fontName=CJK, fontSize=21, leading=30,
        spaceAfter=19, textColor=INK, wordWrap='CJK', keepWithNext=True),
    'section': ParagraphStyle('section', fontName=CJK, fontSize=13.4, leading=21,
        spaceBefore=13, spaceAfter=9, textColor=ACCENT, wordWrap='CJK'),
    'small': ParagraphStyle('small', fontName=CJK, fontSize=9.1, leading=14.5,
        spaceAfter=9, textColor=GREY, wordWrap='CJK'),
    'cell': ParagraphStyle('cell', fontName=CJK, fontSize=9.3, leading=14.2,
        textColor=INK, wordWrap='CJK'),
    'code': ParagraphStyle('code', fontName=MONO, fontSize=8.5, leading=12.5,
        spaceBefore=6, spaceAfter=12, textColor=INK),
}
styles['appendix'] = ParagraphStyle('appendix',parent=styles['body'],
                                   fontSize=10.5,leading=16.5,spaceAfter=6)


def escaped(text):
    # Select a fallback for superscripts, Greek, and other mathematical symbols.
    pieces, current, chosen = [], '', CJK
    for char in text:
        if char in pdfmetrics.getFont(CJK).face.charToGlyph:
            font = CJK
        elif ord(char) in pdfmetrics.getFont(CJK).face.charToGlyph:
            font = CJK
        elif ord(char) in pdfmetrics.getFont(LATIN).face.charToGlyph:
            font = LATIN
        else:
            raise ValueError('Missing glyph: ' + repr(char))
        if font != chosen and current:
            pieces.append((chosen, current))
            current = ''
        chosen = font
        current += char
    if current:
        pieces.append((chosen, current))
    return ''.join('<font name="' + f + '">' + html.escape(s) + '</font>'
                   for f, s in pieces)


def inline(text):
    chunks = re.split(r'(https?://[^\s]+)', text)
    result = ''
    for chunk in chunks:
        if chunk.startswith(('https://', 'http://')):
            url = chunk.rstrip('。；，')
            tail = chunk[len(url):]
            result += '<link href="' + html.escape(url, quote=True) + \
                      '" color="#30566b">' + escaped(url) + '</link>' + escaped(tail)
        else:
            result += escaped(chunk)
    # Internal reference links coexist with paper-visible reference labels.
    result = re.sub(r'\[([RS]\d+)\]', r'<link href="#\1">[\1]</link>', result)
    return result


def para(text, style='body'):
    return Paragraph(inline(text), styles[style])


class Math(Flowable):
    parser = MathTextParser('path')
    font_names = {}

    def __init__(self, expression):
        super().__init__()
        self.parsed = self.parser.parse('$' + expression + '$', dpi=72,
                                        prop=FontProperties(size=11.8))
        for font, size, code, x, y in self.parsed.glyphs:
            filename = str(font.fname)
            if filename not in self.font_names:
                name = 'Math' + str(len(self.font_names))
                pdfmetrics.registerFont(TTFont(name, filename))
                self.font_names[filename] = name
        self.spaceAfter = 13
        self.spaceBefore = 4

    def wrap(self, aw, ah):
        if self.parsed.width > aw:
            raise ValueError('Formula too wide; split explicitly: ' + str(self.parsed.width))
        self.width = aw
        self.height = self.parsed.height + 10
        return self.width, self.height

    def draw(self):
        c, p = self.canv, self.parsed
        dx, dy = (self.width - p.width) / 2, p.depth + 4
        c.setFillColor(INK)
        for font, size, code, x, y in p.glyphs:
            c.setFont(self.font_names[str(font.fname)], size)
            c.drawString(dx + x, dy + y, chr(code))
        for x, y, w, h in p.rects:
            c.rect(dx + x, dy + y, w, h, fill=1, stroke=0)


class Diagram(Flowable):
    def __init__(self, name):
        super().__init__()
        self.name, self.width, self.height = name, BODY_W, 150
        self.spaceAfter = 4

    def wrap(self, aw, ah):
        return self.width, self.height

    def draw(self):
        c = self.canv
        c.setStrokeColor(INK)
        c.setFillColor(INK)

        def label(x, y, text, size=10):
            c.setFont(CJK, size)
            c.drawString(x, y, text)

        def box(x, y, w, h, lines):
            c.setFillColor(colors.white)
            c.roundRect(x, y, w, h, 4, fill=1)
            c.setFillColor(INK)
            for n, line in enumerate(lines):
                label(x + 8, y + h - 19 - n * 17, line, 9.5)

        def arrow(x1, y1, x2, y2, dash=False):
            c.setDash(3, 2) if dash else c.setDash()
            c.line(x1, y1, x2, y2)
            c.setDash()
            import math
            angle = math.atan2(y2-y1, x2-x1)
            for sign in [-1, 1]:
                a = angle + sign * .45
                c.line(x2, y2, x2-7*math.cos(a), y2-7*math.sin(a))

        def grid(x, y, values, cell=27):
            for r, row in enumerate(values):
                for s, value in enumerate(row):
                    c.setFillColor(colors.white)
                    c.rect(x+s*cell, y-r*cell, cell, cell, fill=1)
                    c.setFillColor(INK)
                    label(x+s*cell+6, y-r*cell+9, str(value), 9)

        n = self.name
        if n == 'pixels':
            values = [[0,1,0],[0,.5,1],[0,1,0]]
            grid(30, 99, values)
            arrow(130, 84, 240, 84)
            for r, row in enumerate(values):
                for s, v in enumerate(row):
                    c.setFillColor(colors.Color(v,v,v))
                    c.rect(260+s*27,99-r*27,27,27,fill=1)
            c.setFillColor(INK)
            label(30, 23, '数组（行、列）')
            label(260, 23, '黑底亮字')
        elif n == 'shapes':
            box(0,55,115,65,['X：N×d','每行一个样本'])
            label(128,80,'@',16)
            box(157,55,115,65,['转置 W：d×K','每列一个类别'])
            arrow(285,83,335,83)
            box(349,55,124,65,['Z：N×K','再加偏置 b'])
            label(25,20,'检查中间维度 d 相同；输出保留 N 与 K。')
        elif n == 'update':
            c.line(45,25,445,25);c.line(45,25,45,140)
            path=c.beginPath()
            for i in range(101):
                w=i*3/100
                px,py=45+w*125,25+(2*w-3)**2/2*23
                path.moveTo(px,py) if i==0 else path.lineTo(px,py)
            c.drawPath(path)
            arrow(107.5,71,157.5,41.6)
            label(90,110,'w=0.5 → 0.9')
            label(244,8,'w');label(7,130,'L(w)')
        elif n == 'backward':
            box(0,53,86,65,['x=2','w=0.5'])
            box(143,53,88,65,['a=1','h=1'])
            box(306,82,92,46,['z0=1'])
            box(306,12,92,46,['z1=-1'])
            arrow(86,85,143,85)
            arrow(231,91,306,105);arrow(231,74,306,36)
            arrow(306,96,239,82,True);arrow(306,29,239,65,True)
            label(238,132,'g0=0.880797',8.5)
            label(238,0,'g1=-0.880797',8.5)
            label(20,16,'dh=1.761594；dw=3.523188',9)
        elif n == 'convolution':
            grid(15,101,[[1,2,0],[0,1,3],[2,0,1]])
            label(124,87,'×',15)
            grid(165,101,[[1,0],[0,-1]])
            arrow(240,81,296,81)
            grid(322,101,[[0,-1],[0,0]])
            label(12,22,'3×3 输入');label(162,22,'共享核 2×2');label(320,22,'2×2 输出')
        elif n == 'channels':
            for i in range(3):
                box(10,100-i*43,98,34,[f'输入通道 {i}'])
                box(143,100-i*43,74,34,[f'核片 {i}'])
                arrow(108,117-i*43,143,117-i*43)
                arrow(217,117-i*43,298,72)
            box(298,44,148,60,['通道内局部乘积','跨通道求和 + b'])
        elif n == 'pooling':
            grid(5,100,[[1,4],[2,3]])
            arrow(74,91,145,91)
            grid(160,100,[[4]])
            label(237,110,'感受野 r：1→3→4→8→10',10)
            label(237,82,'步距 j：1→1→2→2→4',10)
            label(237,52,'卷积、池化依次递推',10)
            label(6,18,'池化窗口最大值，不是平均值')
        elif n == 'confusion':
            grid(140,98,[[8,2,0],[1,6,3],[0,1,9]],32)
            label(138,139,'预测：0       1       2')
            for i in range(3):label(80,107-i*32,f'真实 {i}')
            label(279,103,'总数 30')
            label(279,75,'正确 23')
            label(279,47,'Accuracy=23/30')
        else:
            chains = {
                'preprocessing': [['RGBA','白底合成'],['灰度/反色','阈值/外接框'],
                    ['长边20','28×28中心'],['标准化','1×1×28×28']],
                'request': [['Canvas','PNG 请求'],['FastAPI','解码校验'],
                    ['预处理','固定输入约定'],['CNN logits','Softmax']],
                'split': [['原训练60k','固定排列42'],['训练55k','更新参数'],
                    ['验证5k','选择权重'],['独立测试10k','冻结后评价']],
            }
            for i, lines in enumerate(chains[n]):
                box(i*122,49,105,72,lines)
                if i<3:arrow(i*122+105,84,(i+1)*122,84)
            label(10,20,'分工与顺序要明确，不能把中间步骤隐藏为“自动识别”。',9)


def table(rows, widths=None):
    cells = [[para(str(cell), 'cell') for cell in row] for row in rows]
    t = Table(cells, colWidths=widths or [BODY_W/len(rows[0])]*len(rows[0]),
              repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eeeeee')),
        ('LINEBELOW',(0,0),(-1,0),.7,GREY),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#cccccc')),
        ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
    ]))
    t.spaceAfter=12
    return t


def dynamic(kind):
    comparison=json.loads((E/'comparison.json').read_text('utf-8'))
    if kind=='comparison':
        rows=[['教学模型','参数量','最佳验证 Accuracy','验证 Loss','训练秒数']]
        for r in comparison['runs']:
            v=r['validation']
            rows.append([r['model'],f"{r['parameters']:,}",f"{v['accuracy']:.4%}",
                         f"{v['loss']:.6f}",f"{r['seconds']:.3f}"])
        return [table(rows),para('模板验证 Accuracy='+format(
            json.loads((E/'template.json').read_text('utf-8'))['accuracy'],'.4%')+
            '；距离分数没有作为公平交叉熵比较。','small')]
    if kind=='test':
        name=comparison['selected_on_validation']
        test=json.loads((E/(name+'_test.json')).read_text('utf-8'))
        return [para(f"按验证选中 {name}；快速测试 n={test['n']}，正确比例 "
            f"{test['accuracy']:.4%}，平均交叉熵 {test['loss']:.6f}。"
            '仅此一组测试评价，不是生产权重或个人笔迹准确率。')]
    if kind=='environment':
        env=json.loads((E/'environment.json').read_text('utf-8'))
        rows=[['依赖','基线声明','实际安装','文档参考']]
        for name,decl in [('torch','>=2.4.0'),('torchvision','>=0.19.0'),
            ('numpy','>=1.26.0'),('pandas','>=2.0.0'),('fastapi','>=0.115,<1'),
            ('uvicorn','>=0.30,<1'),('matplotlib','>=3.8'),('pillow','>=10'),
            ('torchinfo','>=1.8'),('httpx','>=0.27')]:
            ref='2.14' if name=='torch' else '0.29' if name=='torchvision' else '未用作机制版本依据'
            rows.append([name,decl,env['packages'][name],ref])
        t=table(rows,[80,111,106,BODY_W-297])
        t.setStyle(TableStyle([('TOPPADDING',(0,0),(-1,-1),5),
                              ('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        return [t]
    if kind=='verification':
        edges=json.loads((E/'project_edges.json').read_text('utf-8'))
        rows=[['受控停止情形','实际处理样本 n','history 轮数','替换临时权重']]
        for case in edges['cases']:
            rows.append([case['case'],case['metrics'].get('n','未进入训练'),
                case['history_epochs'],'是' if case['checkpoint_replaced'] else '否'])
        return [table(rows),para('原应用权重 SHA256 前后相同：'+
            edges['original_weight_sha256_unchanged'],'small')]
    raise ValueError(kind)


SOURCES=[
 ('S1','mnist_web/model.py','MnistCNN.__init__/forward；count_parameters',40,83),
 ('S2','mnist_web/utils.py','_center_by_mass；preprocess_digit；preprocess_uploaded_image',448,610),
 ('S3','mnist_web/utils.py','get_transforms；get_dataloaders',136,220),
 ('S4','mnist_web/train.py','build_optimizer；build_scheduler',53,88),
 ('S5','mnist_web/train.py','train_one_epoch；evaluate；predict_batch',91,217),
 ('S6','mnist_web/train.py','fit：每轮/最后保存、报告与停止',247,437),
 ('S7','mnist_web/model.py','save_model；load_model',119,185),
 ('S8','mnist_web/app.py','decode_image；infer；sample；analyze',88,196),
 ('S9','mnist_web/app.py','training_worker；launch_training；stop_training',199,254),
 ('S10','mnist_web/app.py','finish_shutdown；shutdown_service；main',270,325),
 ('S11','mnist_web/static/app.js','画布、请求、轮询与关闭界面',1,None),
 ('S12','mnist_web/static/index.html','三工作区页面与静态资源入口',1,None),
 ('S13','mnist_web/static/style.css','视觉样式、布局与响应式规则',1,None),
 ('S14','mnist_web/tests/test_api.py','ApiTests：9 项回归路径',1,None),
 ('S15','mnist_web/start.bat','虚拟环境选择与 --open-browser',1,None),
 ('S16','mnist_web/requirements.txt','依赖声明范围',1,None),
]


class Book(BaseDocTemplate):
    def __init__(self,path):
        super().__init__(str(path),pagesize=A4,title='从一个3开始：Digit Studio神经网络教材',
            author='Digit Studio 教材',allowSplitting=1)
        frames=[]
        for name,left in [('odd',68),('even',48)]:
            frame=Frame(left,53,BODY_W,PAGE_H-110,id=name,
                        leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
            frames.append(PageTemplate(id=name,frames=[frame],onPage=self.page))
        self.addPageTemplates(frames)
        self.chapter_locations=[]

    def page(self,c,doc):
        c.saveState()
        c.setFont(CJK,8)
        c.setFillColor(GREY)
        if doc.page>1:
            chapter='从一个 3 开始 · Digit Studio'
            for page,title in getattr(self,'prior_locations',[]):
                if page<=doc.page:chapter=title
            c.drawString(68 if doc.page%2 else 48,PAGE_H-33,chapter)
            c.setStrokeColor(colors.HexColor('#dddddd'))
            c.line(48,43,PAGE_W-48,43)
            if doc.page%2:c.drawRightString(PAGE_W-48,29,str(doc.page))
            else:c.drawString(48,29,str(doc.page))
        c.restoreState()

    def beforeDocument(self):
        self.prior_locations=getattr(self,'chapter_locations',[])
        self.chapter_locations=[]

    def afterFlowable(self,f):
        if isinstance(f,Paragraph) and hasattr(f,'chapter_key'):
            key,title=f.chapter_key,f.getPlainText()
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(title,key,0,False)
            self.notify('TOCEntry',(0,title,self.page,key))
            self.chapter_locations.append((self.page,title))

    def afterPage(self):
        self.handle_nextPageTemplate('even' if self.page%2 else 'odd')


def main():
    charts()
    story=[]
    story.append(Spacer(1,90))
    story.append(para('从一个 3 开始','chapter'))
    story.append(para('Digit Studio 神经网络与手写数字识别教材','section'))
    story.append(Spacer(1,22))
    story.append(para('看见像素 · 手算梯度 · 搭建模型 · 回到项目'))
    story.append(Spacer(1,40))
    story.append(para('面向初学者｜完整配套 CPU 实验｜A4 双面打印版'))
    story.append(para('版本 1.0｜核对日期 2026-10-11','small'))
    story.append(para('基线仓库：'+REPO,'small'))
    story.append(para('main · '+SHA,'small'))
    story.append(para('通用原理、教学例子、基线事实与真实实验分别标明。','small'))
    story.append(PageBreak())
    story.append(para('阅读导航','chapter'))
    toc=TableOfContents()
    toc.levelStyles=[ParagraphStyle('toc',fontName=CJK,fontSize=10.3,leading=18,
        spaceBefore=0,wordWrap='CJK',textColor=INK)]
    toc.tableStyle=TableStyle([('TOPPADDING',(0,0),(-1,-1),2),
                              ('BOTTOMPADDING',(0,0),(-1,-1),2)])
    toc.dotsMinLevel=0
    story.append(toc)
    story.append(PageBreak())
    lines=(HERE/'book.md').read_text('utf-8').splitlines()
    i,paragraph,first,chapter_index=0,[],True,0
    compact_appendix=False

    def flush():
        if paragraph:
            text=' '.join(paragraph)
            body_style='appendix' if compact_appendix else 'body'
            p=para(text,body_style)
            match=re.match(r'\[(R\d+)\]',text)
            if match:p=Paragraph('<a name="'+match[1]+'"/>'+inline(text),styles[body_style])
            story.append(p)
            paragraph.clear()

    while i<len(lines):
        line=lines[i]
        if line.startswith('# '):
            flush()
            if not first:
                # Give chapters a clear opening without wasting a nearly empty tail page.
                story.append(CondPageBreak(400))
                story.append(Spacer(1,22))
            title=line[2:]
            compact_appendix=title.startswith('附录 E')
            if first:title='开始之前'
            p=para(title,'chapter');p.chapter_key='chapter'+str(chapter_index)
            story.append(p);chapter_index+=1;first=False
        elif line.startswith('## '):
            flush();story.append(CondPageBreak(90));story.append(para(line[3:],'section'))
        elif line.startswith('```'):
            flush();code=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):
                code.append(lines[i]);i+=1
            maxwidth=max([pdfmetrics.stringWidth(x,MONO,8.5) for x in code] or [0])
            size=8.5 if maxwidth<=BODY_W-4 else 8.0
            if any(pdfmetrics.stringWidth(x,MONO,size)>BODY_W for x in code):
                raise ValueError('Code line too wide: '+max(code,key=len))
            style=ParagraphStyle('codeblock',parent=styles['code'],fontSize=size)
            story.append(Preformatted('\n'.join(code),style))
        elif line.startswith('@math '):
            flush();story.append(Math(line[6:]))
        elif line.startswith('@figure '):
            flush();name,caption=line[8:].split(' | ',1)
            if name.startswith('real_'):
                chosen=json.loads((E/'comparison.json').read_text('utf-8'))['selected_on_validation']
                path={'real_samples':E/'mnist_samples.png',
                      'real_matrix':E/(chosen+'_test_matrix.png'),
                      'real_errors':E/(chosen+'_test_errors.png')}.get(name,E/(name+'.png'))
                w,h=PILImage.open(path).size
                height=BODY_W*h/w
                if height>310:
                    height=310;width=height*w/h
                else:width=BODY_W
                graphic=Image(str(path),width=width,height=height)
            else:graphic=Diagram(name)
            story.append(KeepTogether([graphic,para(caption,'small')]))
        elif line.startswith('@results '):
            flush();story.extend(dynamic(line[9:]))
        elif line=='@environment':
            flush();story.extend(dynamic('environment'))
        elif line=='@verification':
            flush();story.extend(dynamic('verification'))
        elif line=='@sources':
            flush()
            for key,path,symbol,start,end in SOURCES:
                url=REPO+'/blob/'+SHA+'/'+path+'#L'+str(start)
                if end:url+='-L'+str(end)
                text='['+key+'] '+path+' · '+symbol
                story.append(Paragraph('<a name="'+key+'"/><link href="'+url+'">'+
                    escaped(text)+'</link>',styles['small']))
        elif line.startswith('|'):
            flush();rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                rows.append([s.strip() for s in lines[i].strip('|').split('|')]);i+=1
            i-=1;story.append(table(rows))
        elif not line.strip():flush()
        else:paragraph.append(line)
        i+=1
    flush()
    OUT.parent.mkdir(parents=True,exist_ok=True)
    book=Book(OUT)
    book.multiBuild(story)
    manifest={'pdf':OUT.name,'sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),
        'baseline':SHA,'source': 'textbook/book.md','python':platform.python_version(),
        'body_font_size':11,'inner_margin_pt':68,'outer_margin_pt':48,
        'sources':SOURCES,'formula_count':sum(l.startswith('@math') for l in lines),
        'diagram_count':sum(l.startswith('@figure') for l in lines)}
    (E/'build_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(OUT.name,OUT.stat().st_size)


if __name__=='__main__':main()
