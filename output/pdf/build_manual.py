"""Build the portable Chinese manual with live source excerpts and bookmarks."""
import ast
import hashlib
import html
import json
import re
import textwrap
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                              Spacer, PageBreak, Table, TableStyle, Flowable,
                              Image, KeepTogether, CondPageBreak)
from reportlab.platypus.tableofcontents import TableOfContents
from PIL import Image as PILImage

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / 'output' / 'pdf'
PDF = OUT / 'Digit_Studio_Project_Manual.pdf'
FONT = 'ManualCJK'
pdfmetrics.registerFont(TTFont(FONT, 'C:/Windows/Fonts/simhei.ttf'))
INK = colors.HexColor('#26324B')
MUTED = colors.HexColor('#7E889D')
ACCENT = colors.HexColor('#7962D8')
PALE = colors.HexColor('#F4F2FC')
WIDTH, HEIGHT = A4
BODY_W = WIDTH - 100

styles = {
    'body': ParagraphStyle('body', fontName=FONT, fontSize=10.1, leading=17,
                           textColor=INK, spaceAfter=7, wordWrap='CJK', splitLongWords=True),
    'chapter': ParagraphStyle('chapter', fontName=FONT, fontSize=22, leading=31,
                              textColor=INK, spaceAfter=20, keepWithNext=True, wordWrap='CJK'),
    'section': ParagraphStyle('section', fontName=FONT, fontSize=13.1, leading=20,
                              textColor=ACCENT, spaceBefore=12, spaceAfter=9, keepWithNext=False, wordWrap='CJK'),
    'small': ParagraphStyle('small', fontName=FONT, fontSize=8.1, leading=12.7,
                            textColor=MUTED, spaceAfter=8, wordWrap='CJK'),
    'ref': ParagraphStyle('ref', fontName=FONT, fontSize=9, leading=14,
                          textColor=INK, spaceAfter=6, wordWrap='CJK'),
    'cell': ParagraphStyle('cell', fontName=FONT, fontSize=8.2, leading=13.2, textColor=INK, wordWrap='CJK'),
    'tablehead': ParagraphStyle('tablehead', fontName=FONT, fontSize=8.4, leading=13.2,
                                textColor=colors.white, wordWrap='CJK'),
}

def inline(text):
    value = html.escape(text)
    return re.sub(r'https?://[^\s]+', lambda m: '<link href="'+m.group(0)+'" color="#7962D8">'+m.group(0)+'</link>', value)

def para(text, kind='body'):
    return Paragraph(inline(text), styles[kind])

class SourceCode(Flowable):
    """Line-wrapped, splittable vector code block with real source line numbers."""
    def __init__(self, lines, continued=False):
        super().__init__()
        self.lines = lines
        self.continued = continued
        self.font_size = 8.2
        self.leading = 12
        self.spaceAfter = 12

    def wrap(self, availWidth, availHeight):
        self.width = availWidth
        self.wrapped = []
        for number, line in self.lines:
            # Wrap by actual glyph width, preserving indentation and source label.
            chunks, chunk = [], ''
            for char in line.expandtabs(4):
                if chunk and pdfmetrics.stringWidth(chunk+char, FONT, self.font_size) > availWidth - 59:
                    chunks.append(chunk)
                    chunk = '    ' + char
                else:
                    chunk += char
            chunks.append(chunk)
            for j, item in enumerate(chunks):
                self.wrapped.append((str(number) if number and j == 0 else '', item))
        self.height = len(self.wrapped)*self.leading + 20
        return self.width, self.height

    def split(self, availWidth, availHeight):
        self.wrap(availWidth, availHeight)
        count = int((availHeight-20)/self.leading)
        if count < 2:
            return []
        # Already wrapped lines carry labels; subsequent wrapping is stable.
        return [SourceCode(self.wrapped[:count], self.continued),
                SourceCode(self.wrapped[count:], True)]

    def draw(self):
        c = self.canv
        c.setFillColor(colors.HexColor('#F3F5F9'))
        c.roundRect(0, 0, self.width, self.height, 6, fill=1, stroke=0)
        c.setStrokeColor(colors.HexColor('#DDE2EC'))
        c.line(39, 8, 39, self.height-8)
        c.setFont(FONT, self.font_size)
        y = self.height-17
        for label, line in self.wrapped:
            c.setFillColor(MUTED)
            c.drawRightString(32, y, label)
            c.setFillColor(INK)
            c.drawString(47, y, line)
            y -= self.leading

class Diagram(Flowable):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind == 'inference':
            self.nodes = [('网页画布 / 上传', '浏览器：320×320 笔迹或缩小后的图片'),
                          ('JSON + PNG Data URL', 'POST /api/predict 或 /api/analyze'),
                          ('解码与预处理', '灰度、反色、裁剪、20 像素框、28×28 标准化'),
                          ('CNN 前向计算', '(1,1,28,28) → logits (1,10)'),
                          ('Softmax 与返回', '十类概率、预测数字、PNG 输入预览'),
                          ('网页呈现', '版本匹配后更新候选条、数字和耗时')]
        else:
            self.nodes = [('开始训练', '校验配置 → 获取 JOB_LOCK → 后台线程'),
                          ('fit 准备阶段', '加载划分数据 → 新建 MnistCNN → 优化器'),
                          ('逐批更新参数', 'zero_grad → forward → loss → backward → step'),
                          ('每轮验证与保存', 'evaluate → scheduler → history → torch.save'),
                          ('报告与重载', '完整测试集评估 → 图表/CSV → reload_model'),
                          ('结束与清理', 'running=False，清理 STOP，释放 JOB_LOCK')]
        self.height = len(self.nodes)*53-9
        self.spaceAfter = 16

    def wrap(self, aw, ah):
        self.width=aw
        return aw, self.height

    def draw(self):
        c=self.canv
        y=self.height-44
        for i,(title,desc) in enumerate(self.nodes):
            c.setFillColor(PALE if i % 2 == 0 else colors.HexColor('#F3F6FA'))
            c.roundRect(0,y,self.width,44,7,stroke=0,fill=1)
            c.setFillColor(ACCENT)
            c.setFont('Helvetica-Bold',11)
            c.drawString(13,y+25,f'{i+1:02d}')
            c.setFont(FONT,10)
            c.setFillColor(INK)
            c.drawString(45,y+27,title)
            c.setFont(FONT,8.2)
            c.setFillColor(MUTED)
            c.drawString(45,y+11,desc)
            if i<len(self.nodes)-1:
                c.setStrokeColor(ACCENT)
                c.line(self.width/2,y-1,self.width/2,y-7)
                c.line(self.width/2-2,y-5,self.width/2,y-7)
                c.line(self.width/2+2,y-5,self.width/2,y-7)
            y-=53

class Cover(Flowable):
    def wrap(self, aw, ah):
        self.width=aw
        self.height=640
        return aw, self.height

    def draw(self):
        c=self.canv
        c.setFillColor(INK)
        c.roundRect(-12, 126, self.width+24, 490, 18, stroke=0, fill=1)
        c.setFont('Helvetica-Bold', 11)
        c.setFillColor(colors.HexColor('#BDAEF8'))
        c.drawString(20,575,'TECHNICAL & USER MANUAL')
        c.setFont('Helvetica-Bold',44)
        c.setFillColor(colors.white)
        c.drawString(20,507,'Digit Studio')
        c.setFont(FONT,29)
        c.drawString(20,455,'手写数字实验室')
        c.setFont(FONT,19)
        c.setFillColor(colors.HexColor('#D5DBEA'))
        c.drawString(20,415,'项目说明书与代码原理详解')
        c.setStrokeColor(colors.HexColor('#6653AB'))
        c.line(20,382,self.width-20,382)
        c.setFont(FONT,12)
        c.setFillColor(colors.HexColor('#D5DBEA'))
        for i,line in enumerate(['从画布到张量，从卷积到预测',
                                 '从损失到学习，从后台任务到优雅关闭',
                                 '18 个章节 · 关键代码 · 数学原理 · 操作与维护']):
            c.drawString(20,350-i*27,line)
        c.setFont('Helvetica-Bold',46)
        c.setFillColor(colors.HexColor('#7663B4'))
        c.drawString(20,204,'0  1  2  3  4')
        c.drawString(20,150,'5  6  7  8  9')
        c.setFillColor(INK)
        c.setFont(FONT,11)
        c.drawString(1,83,'代码基线  2026-10-11')
        c.drawString(1,58,'路径规范  项目相对路径，可迁移至其他电脑')
        c.setFillColor(MUTED)
        c.setFont(FONT,9)
        c.drawString(1,25,'FastAPI + HTML Canvas + PyTorch CNN + MNIST')

class ManualDoc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(str(filename),pagesize=A4,leftMargin=50,rightMargin=50,
                         topMargin=53,bottomMargin=49,
                         title='Digit Studio 项目说明书与代码原理详解',
                         author='Digit Studio 项目文档',subject='使用、原理、源码、测试与维护')
        self.addPageTemplates(PageTemplate(id='manual',frames=Frame(50,49,BODY_W,HEIGHT-102,
                                                                   leftPadding=0,rightPadding=0,
                                                                   topPadding=0,bottomPadding=0),onPage=self.header))
        self.chapter_name='阅读指南'
        self.locations={}

    def header(self,c,doc):
        if doc.page == 1:
            return
        c.saveState()
        c.setFont('Helvetica',8)
        c.setFillColor(MUTED)
        c.drawString(50,HEIGHT-30,'DIGIT STUDIO / PROJECT MANUAL')
        c.setFont(FONT,8)
        current = '阅读导航'
        for title, page in sorted(self.locations.items(), key=lambda item: item[1]):
            if page <= doc.page:
                current = title
        c.drawRightString(WIDTH-50,HEIGHT-30,current)
        c.setStrokeColor(colors.HexColor('#E6E9F0'))
        c.line(50,HEIGHT-38,WIDTH-50,HEIGHT-38)
        c.line(50,38,WIDTH-50,38)
        c.setFont(FONT,7.8)
        c.drawString(50,24,'2026-10-11 基线 · 路径以项目根目录为准')
        c.setFont('Helvetica',8)
        c.drawRightString(WIDTH-50,24,str(doc.page))
        c.restoreState()

    def afterFlowable(self,flowable):
        if isinstance(flowable,Paragraph) and flowable.style.name == 'chapter':
            title=flowable.getPlainText()
            key='chapter-'+title.split(' ')[0]
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(title,key,0)
            if title != '阅读导航':
                self.notify('TOCEntry',(0,title,self.page,key))
            self.chapter_name=title
            self.locations[title]=self.page

def table(rows):
    count=len(rows[0])
    ratios={2:[.34,.66],3:[.28,.36,.36],4:[.19,.19,.27,.35]}.get(count,[1/count]*count)
    data=[[para(cell,'tablehead' if ri==0 else 'cell') for cell in row] for ri,row in enumerate(rows)]
    t=Table(data,colWidths=[BODY_W*r for r in ratios],repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),INK),('VALIGN',(0,0),(-1,-1),'TOP'),
                           ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
                           ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
                           ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#F7F8FB'),colors.white]),
                           ('LINEBELOW',(0,0),(-1,0),.5,INK),
                           ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#E7EBF2'))]))
    t.spaceAfter=12
    return t

source_files=['mnist_web/app.py','mnist_web/model.py','mnist_web/train.py','mnist_web/utils.py',
              'mnist_web/static/index.html','mnist_web/static/style.css','mnist_web/static/app.js',
              'mnist_web/start.bat','mnist_web/requirements.txt','mnist_web/tests/test_api.py']
excerpts=[]
def source_block(rel, anchor, count):
    lines=(PROJECT/rel).read_text(encoding='utf-8-sig').splitlines()
    starts=[i for i,line in enumerate(lines) if anchor in line]
    if not starts:
        raise ValueError(f'Missing source anchor: {rel}: {anchor}')
    start=starts[0]
    end=min(start+int(count),len(lines))
    excerpt=[(i+1,lines[i]) for i in range(start,end)]
    excerpts.append({'file':rel,'start':start+1,'end':end,'anchor':anchor})
    return [para(f'源码摘录 · {rel} · 第 {start+1} 至 {end} 行','small'),SourceCode(excerpt)]

def index_blocks():
    result=[]
    for rel in source_files:
        if not rel.endswith('.py'):
            continue
        result.append(para(rel,'section'))
        tree=ast.parse((PROJECT/rel).read_text(encoding='utf-8-sig'))
        rows=[['符号与起始行','职责定位 / 代码类型']]
        for node in tree.body:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
                name=node.name
                rows.append([f'{name} · L{node.lineno}', '类定义' if isinstance(node,ast.ClassDef) else ('异步函数' if isinstance(node,ast.AsyncFunctionDef) else '函数')])
                if isinstance(node,ast.ClassDef):
                    for child in node.body:
                        if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef)):
                            rows.append([f'{name}.{child.name} · L{child.lineno}','类方法 / 测试用例' if rel.endswith('test_api.py') else '类方法'])
        result.append(table(rows))
    return result

text=(OUT/'manual_content.md').read_text(encoding='utf-8')
lines=text.splitlines()
story=[Cover(),PageBreak(),para('阅读导航','chapter'),
       para('按使用流程阅读，或点击目录和 PDF 书签跳转到对应章节。文档正文包含关键连续代码摘录，行号与本次基线一致。'),
       para('快速定位：启动/关闭见第 03、09 章；预处理见第 05 章；CNN 与训练见第 06 至 08 章；接口见第 04 章；故障排查见第 15 章。')]
story.append(para('本文区分当前实现、教学解释和建议扩展。建议扩展不表示已经具备该功能；源文件遗留注释与真实行为不同时，以函数体和本说明书的核对解释为准。','small'))
toc=TableOfContents()
toc.levelStyles=[ParagraphStyle('TOC0',fontName=FONT,fontSize=10,leading=17,
                               textColor=INK,spaceBefore=1,wordWrap='CJK',leftIndent=0,rightIndent=18)]
story.extend([Spacer(1,10),toc])
i=0
while i<len(lines):
    line=lines[i].strip()
    if not line or line.startswith('# ') or line.startswith('版本基线：'):
        i+=1;continue
    if line.startswith('## '):
        story.extend([PageBreak(),para(line[3:],'chapter')])
    elif line.startswith('### '):
        story.extend([CondPageBreak(95),para(line[4:],'section')])
    elif line.startswith('```'):
        lang=line[3:]
        code=[]
        i+=1
        while i<len(lines) and not lines[i].startswith('```'):
            code.append(('',lines[i]));i+=1
        story.extend([para('示例命令 / 字段示例 · '+lang,'small'),SourceCode(code)])
    elif line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].strip().startswith('|'):
            raw=lines[i].strip()
            if not re.fullmatch(r'[|:\- ]+',raw):
                rows.append([s.strip() for s in raw.strip('|').split('|')])
            i+=1
        story.append(table(rows));continue
    elif line.startswith('{{code|'):
        rel,anchor,count=line[7:-2].split('|')
        story.extend(source_block(rel,anchor,count))
    elif line.startswith('{{diagram|'):
        story.append(Diagram(line[10:-2]))
    elif line.startswith('{{image|'):
        rel,caption=line[8:-2].split('|',1)
        image_path=PROJECT/rel
        if image_path.exists():
            with PILImage.open(image_path) as im:
                iw,ih=im.size
            scale=min(BODY_W/iw,300/ih)
            story.extend([Image(str(image_path),width=iw*scale,height=ih*scale),Spacer(1,7),para(caption,'small')])
    elif line=='{{index}}':
        story.extend(index_blocks())
    elif line=='{{hashes}}':
        story.append(table([['项目相对路径','SHA-256 前 16 位']]+[[rel,hashlib.sha256((PROJECT/rel).read_bytes()).hexdigest()[:16]] for rel in source_files]))
    else:
        story.append(para(line,'ref' if line.startswith('[R') or line.startswith('https://') else 'body'))
    i+=1

doc=ManualDoc(PDF)
doc.multiBuild(story)
manifest={'pdf':PDF.name,'source_date':'2026-10-11','chapters':doc.locations,
          'excerpts':excerpts,'source_hashes':{rel:hashlib.sha256((PROJECT/rel).read_bytes()).hexdigest() for rel in source_files}}
(OUT/'manual_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'pdf':PDF.name,'bytes':PDF.stat().st_size,'chapters':len(doc.locations),'excerpts':len(excerpts)},ensure_ascii=False))
