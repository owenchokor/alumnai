import io
import os
import json
import fitz
from datetime import datetime
from tqdm import tqdm
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import red, black
from annot.llm import LLM

class PDFAnnotator:
    
    @staticmethod
    def add_summary(pdf_path : str, script_path : str, output_filename : str):
        pdfmetrics.registerFont(TTFont('a시네마B', 'a시네마B.ttf'))
        reader = PdfReader(pdf_path)
        writer = PdfWriter()
        loop = tqdm(range(len(reader.pages)), desc="Adding summaries to PDF pages", unit="page", total=len(reader.pages))
        for page_no in loop:
            try:
                scripts = ''
                js = PDFAnnotator.__read_json(script_path)
                for item in js[str(page_no)]:
                    item = ' ' + item
                    scripts+=item

                _, summary = PDFAnnotator.summarize_page(page_no, pdf_path, scripts)
            except KeyError:
                loop.set_description(f"Page {page_no}에 대해서는 음성으로 설명한 내용이 없습니다.")
                _, summary = PDFAnnotator.summarize_page(page_no, pdf_path, '')

            packet = io.BytesIO()
            original_page = reader.pages[page_no]
            page_width = original_page.mediabox.width
            page_height = original_page.mediabox.height
            can = canvas.Canvas(packet, pagesize=(page_width, page_height))
            can.setFont("a시네마B", 12)

            x_position = 50
            y_position = page_height - 50
            in_red = False
            for line in summary.split('\n'):

                words = line.split(' ')
                for word in words:
                    if '**' in word:
                        parts = word.split('**')
                        for i, part in enumerate(parts):
                            # Toggle color when encountering '**'
                            if i > 0:
                                in_red = not in_red
                            can.setFillColor(red if in_red else black)

                            # Draw each part
                            if part:
                                part_width = can.stringWidth(part, "a시네마B", 12)
                                if x_position + part_width > page_width - 50:
                                    y_position -= 15
                                    x_position = 50
                                    if y_position < 50:
                                        break
                                can.drawString(x_position, y_position, part)
                                x_position += part_width
                    else:
                        # Draw word normally if no '**' is present
                        word_width = can.stringWidth(word + " ", "a시네마B", 12)
                        if x_position + word_width > page_width - 50:
                            y_position -= 15
                            x_position = 50
                            if y_position < 50:
                                break
                        can.drawString(x_position, y_position, word)
                        x_position += word_width
                    
                    # Add a space after each word
                    x_position += can.stringWidth(" ", "a시네마B", 12)
                
                # Move down after processing the line
                y_position -= 15
                x_position = 50



            can.save()

            packet.seek(0)
            new_pdf = PdfReader(packet)
            writer.add_page(reader.pages[page_no])
            writer.add_page(new_pdf.pages[0])



        with open(output_filename, "wb") as output_pdf:
            writer.write(output_pdf)

        print(f"PDF 파일 '{output_filename}'이(가) 성공적으로 저장되었습니다.")


    @staticmethod
    def summarize_page(page_no: int, pdf_path: str, scripts: str, llm: str = 'gpt'):
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        log_filename = f"alumnai_{llm}_{page_no}_{os.path.basename(pdf_path).split('.')[0]}_{datetime.now().strftime('%y%m%d%H%M%S')}.txt"
        log_path = os.path.join(log_dir, log_filename)
        if scripts:
            pdf_text = PDFAnnotator.extract_text_from_page(pdf_path, page_no)
            summarizer = LLM(llm)
            
            prompt = f"한국어로 된 의학 강의의 speech to text 전사본입니다, 다음 텍스트의 오타를 의학 강의 맥락에 맞게 수정해서, 수정 후의 텍스트 부분만 리턴하세요. 텍스트 : {scripts}"
            summarizer.setResponse(prompt)
            postprocessed_text = summarizer.getResponse()
            

            prompt = (
                f"의학 강의 PDF에서 추출된 텍스트와 의대 교수님의 음성 데이터에서 처리된 텍스트를 입력받아 두 개를 모두 고려한 요약본을 생성하세요.: \n"
                f"{pdf_text}: PDF에서 추출된 텍스트 \n"
                f"{postprocessed_text}: 음성 데이터에서 처리된 텍스트 \n"
                "두 입력 데이터를 결합하여 다음 규칙에 따라 요약을 작성하세요:\n"
                "의학적 및 기술적 키워드는 반드시 포함하세요.\n"
                "의학 용어는 영어로 그대로 사용하고, 나머지 전체는 한국어로 작성합니다.\"\n"
                "요약은 1, 2, 3... 번호를 붙인 개조식 형태로 5줄 이내로 작성하고, 핵심 내용만 포함하세요.\n"
                "중요한 내용은 ** **로 강조해 주세요.\n"
                "최종 요약본만 리턴하세요."
            )
            summarizer.setResponse(prompt)
            summary = summarizer.getResponse()
            
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"###############################\n")
                log_file.write(f"Page {page_no}\n")
                log_file.write(f"###############################\n")
                log_file.write(f"1) Original Speech Data:\n{scripts}\n\n")
                log_file.write(f"2) Corrected Text:\n{postprocessed_text}\n\n")
                log_file.write(f"3) Summary:\n{summary}\n\n")
        else:
            pdf_text = PDFAnnotator.extract_text_from_page(pdf_path, page_no)
            postprocessed_text = ''
            summarizer = LLM(llm)
            prompt = (
                f"의학 강의 PDF에서 추출된 텍스트를 요약해주세요.: \n"
                f"{pdf_text}: PDF에서 추출된 텍스트 \n"
                "의학적 및 기술적 키워드는 반드시 포함하세요.\n"
                "의학 용어는 영어로 그대로 사용하고, 나머지 전체는 한국어로 작성합니다.\"\n"
                "요약은 1, 2, 3... 번호를 붙인 개조식 형태로 5줄 이내로 작성하고, 핵심 내용만 포함하세요.\n"
                "최종 요약본만 리턴하세요."
            )
            summarizer.setResponse(prompt)
            summary = summarizer.getResponse()
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"###############################\n")
                log_file.write(f"Page {page_no}\n")
                log_file.write(f"###############################\n")
                log_file.write(f"1) Summary:\n{summary}\n\n")
        return postprocessed_text, summary
    
    @staticmethod
    def extract_text_from_page(pdf_path, page_index):
        pdf_document = fitz.open(pdf_path)
        if page_index < 0 or page_index >= pdf_document.page_count:
            raise ValueError(f"페이지 인덱스가 범위를 벗어났습니다. 유효한 범위: 0 ~ {pdf_document.page_count - 1}")
        page = pdf_document.load_page(page_index)
        text = page.get_text("text")

        
        pdf_document.close()

        return text
    
    @staticmethod
    def __read_json(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data