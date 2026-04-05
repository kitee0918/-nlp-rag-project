"""Generate PDF Timeline for CS6493 RAG Project."""

from fpdf import FPDF
import os


class TimelinePDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, 'CS6493 RAG Project Timeline', 0, new_x='LMARGIN', new_y='NEXT', align='C')
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, align='C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(26, 54, 93)
        self.set_fill_color(237, 242, 247)
        self.cell(0, 10, title, 0, new_x='LMARGIN', new_y='NEXT', align='L', fill=True)
        self.ln(3)

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(44, 82, 130)
        self.cell(0, 8, title, 0, new_x='LMARGIN', new_y='NEXT', align='L')
        self.ln(1)

    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(51, 51, 51)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def bullet_point(self, text, done=False):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(51, 51, 51)
        symbol = '[x] ' if done else '[ ] '
        self.set_x(15)
        self.multi_cell(180, 6, symbol + text)

    def add_table(self, headers, data):
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(237, 242, 247)
        col_width = (self.w - 20) / len(headers)
        for h in headers:
            self.cell(col_width, 7, h, 1, align='C', fill=True)
        self.ln()
        self.set_font('Helvetica', '', 9)
        self.set_fill_color(247, 250, 252)
        fill = False
        for row in data:
            for item in row:
                self.cell(col_width, 6, str(item), 1, align='L', fill=fill)
            self.ln()
            fill = not fill
        self.ln(3)

    def code_line(self, text):
        self.set_font('Courier', '', 9)
        self.cell(0, 5, text, new_x='LMARGIN', new_y='NEXT')


def create_pdf():
    pdf = TimelinePDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title Page
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 24)
    pdf.set_text_color(26, 54, 93)
    pdf.ln(30)
    pdf.cell(0, 15, 'CS6493 RAG Project Timeline', new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.set_font('Helvetica', '', 14)
    pdf.set_text_color(51, 51, 51)
    pdf.cell(0, 10, 'Retrieval-Augmented Generation', new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.cell(0, 10, 'for Knowledge-Intensive Tasks', new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.ln(20)
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(0, 8, 'Course: CS6493 Natural Language Processing', new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.cell(0, 8, 'Semester: Q2 2025-2026', new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.cell(0, 8, 'City University of Hong Kong', new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.ln(10)
    pdf.cell(0, 8, 'Group: RAGdoll', new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.cell(0, 8, 'Members: Huibo LI, Lingke MENG, Shibo WANG,', new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.cell(0, 8, '         Xinrui SHAN, Yue ZHANG', new_x='LMARGIN', new_y='NEXT', align='C')
    pdf.ln(10)
    pdf.cell(0, 8, 'GPU: NVIDIA A100 40GB', new_x='LMARGIN', new_y='NEXT', align='C')

    # Overview Page
    pdf.add_page()
    pdf.chapter_title('Project Timeline Overview')
    pdf.body_text('Progress Visualization:')
    pdf.code_line('Phase 1 ======================== Done (3/1 - 3/14)')
    pdf.code_line('Phase 2 ======================== Done (3/15 - 3/21)')
    pdf.code_line('Phase 3 ======================== In Progress (3/22 - 3/28)')
    pdf.code_line('Phase 4 ------------------------ Planned (4/1 - 4/7)')
    pdf.code_line('Phase 5 ------------------------ Planned (4/8 - 4/12)')
    pdf.code_line('Phase 6 ------------------------ Planned (4/13 - 4/14)')
    pdf.ln(5)

    # Phase 1
    pdf.chapter_title('Phase 1: Project Setup & Environment')
    pdf.add_table(['Attribute', 'Detail'], [
        ['Time', 'Mar 1 - Mar 14'],
        ['Status', 'Completed'],
        ['Week', 'Week 1-2']
    ])
    pdf.section_title('Main Tasks')
    pdf.bullet_point('Determine research topic: Finance + Law dual-domain', True)
    pdf.bullet_point('Complete literature review (RAG, FiD, HotpotQA, CUAD, Self-RAG)', True)
    pdf.bullet_point('NVIDIA A100 40GB GPU environment setup', True)
    pdf.bullet_point('Technology stack selection: LangChain, FAISS, Pyserini', True)
    pdf.ln(3)

    # Phase 2
    pdf.chapter_title('Phase 2: Dataset Collection & Indexing')
    pdf.add_table(['Attribute', 'Detail'], [
        ['Time', 'Mar 15 - Mar 21'],
        ['Status', 'Completed'],
        ['Week', 'Week 3']
    ])
    pdf.section_title('Main Tasks')
    pdf.bullet_point('Collect FinanceBench dataset (~150 QA pairs)', True)
    pdf.bullet_point('Collect HotpotQA dev set (~7.4k pairs)', True)
    pdf.bullet_point('Collect CUAD dataset (510 contracts, ~13k QA pairs)', True)
    pdf.bullet_point('Collect LegalBench-RAG-mini (776 pairs)', True)
    pdf.bullet_point('Complete BM25 index construction', True)
    pdf.bullet_point('Establish Recall@3 baseline', True)
    pdf.ln(3)
    pdf.section_title('Dataset Overview')
    pdf.add_table(['Dataset', 'Domain', 'Size', 'Purpose'], [
        ['FinanceBench', 'Finance', '~150 pairs', 'SEC 10-K/10-Q QA'],
        ['HotpotQA', 'Multi-hop', '~7.4k pairs', 'Multi-hop reasoning'],
        ['CUAD', 'Law', '~13k pairs', 'Contract clause extraction'],
        ['LegalBench-RAG', 'Law', '776 pairs', 'Legal retrieval benchmark']
    ])

    # Phase 3
    pdf.add_page()
    pdf.chapter_title('Phase 3: Retrieval System Implementation')
    pdf.add_table(['Attribute', 'Detail'], [
        ['Time', 'Mar 22 - Mar 28'],
        ['Status', 'In Progress'],
        ['Week', 'Week 4']
    ])
    pdf.section_title('Main Tasks')
    pdf.bullet_point('Contriever integration + FAISS vector indexing', True)
    pdf.bullet_point('Hybrid RRF (Reciprocal Rank Fusion) implementation', False)
    pdf.bullet_point('LangChain RetrievalQA framework setup', False)
    pdf.bullet_point('HotpotQA retrieval precision analysis', False)
    pdf.ln(3)
    pdf.section_title('Retrieval Methods Comparison')
    pdf.add_table(['Method', 'Type', 'Feature', 'Use Case'], [
        ['BM25', 'Sparse', 'Keyword matching, no GPU', 'Legal clause matching'],
        ['Contriever', 'Dense', 'Semantic similarity, 768-dim', 'Cross-language'],
        ['Hybrid RRF', 'Mixed', 'RRF fusion (k=60), Top-k=3', 'Best overall']
    ])

    # Phase 4
    pdf.chapter_title('Phase 4: Generation Model Integration')
    pdf.add_table(['Attribute', 'Detail'], [
        ['Time', 'Apr 1 - Apr 7'],
        ['Status', 'Planned'],
        ['Week', 'Week 5']
    ])
    pdf.section_title('Main Tasks')
    pdf.bullet_point('Connect LLaMA-3.1-8B-Instruct', False)
    pdf.bullet_point('Connect Qwen2.5-7B-Instruct', False)
    pdf.bullet_point('Execute 3x2x2 RAG grid experiment', False)
    pdf.bullet_point('Zero-shot vs Instruction-tuned ablation', False)
    pdf.bullet_point('Analyze retrieval precision vs answer quality', False)
    pdf.ln(3)
    pdf.section_title('3x2x2 Experiment Design')
    pdf.code_line('Retrievers (3): BM25, Contriever, Hybrid RRF')
    pdf.code_line('Generators (2): LLaMA-3.1-8B, Qwen2.5-7B')
    pdf.code_line('Domains (2): Finance, Law')
    pdf.code_line('Total experiments: 3 x 2 x 2 = 12')
    pdf.code_line('Metrics: EM, F1, FActScore, Recall@k')
    pdf.ln(5)

    # Phase 5
    pdf.add_page()
    pdf.chapter_title('Phase 5: Hallucination Mitigation & Evaluation')
    pdf.add_table(['Attribute', 'Detail'], [
        ['Time', 'Apr 8 - Apr 12'],
        ['Status', 'Planned'],
        ['Week', 'Week 6']
    ])
    pdf.section_title('Main Tasks')
    pdf.bullet_point('FActScore hallucination rate evaluation', False)
    pdf.bullet_point('LangChain Structured Chain implementation', False)
    pdf.bullet_point('Self-RAG integration comparison', False)
    pdf.bullet_point('Cross-domain performance analysis', False)
    pdf.bullet_point('Human evaluation (50-response sample)', False)
    pdf.ln(3)
    pdf.section_title('Hallucination Mitigation Strategies')
    pdf.add_table(['Method', 'Components', 'Mechanism'], [
        ['LLM Only', 'None', 'Pure parametric, highest hallucination'],
        ['Vanilla RAG', 'Retriever + Generator', 'Standard, no constraints'],
        ['LangChain', 'RetrievalQA + Prompt', 'Forced citation, CoT'],
        ['Self-RAG', 'Reflective tokens', 'Self-critique retrieval']
    ])
    pdf.ln(3)
    pdf.section_title('Evaluation Metrics')
    pdf.add_table(['Metric', 'Description'], [
        ['EM (Exact Match)', 'Exact match accuracy'],
        ['F1 Score', 'Token-level F1'],
        ['Recall@k', 'Retrieval recall rate'],
        ['FActScore', 'Factual precision score'],
        ['Hallucination Rate', 'Unsupported statements ratio']
    ])

    # Phase 6
    pdf.chapter_title('Phase 6: Final Deliverables')
    pdf.add_table(['Attribute', 'Detail'], [
        ['Time', 'Apr 13 - Apr 14'],
        ['Status', 'Planned'],
        ['Week', 'Final Week']
    ])
    pdf.section_title('Main Tasks')
    pdf.bullet_point('Final report writing and analysis', False)
    pdf.bullet_point('Presentation slides preparation', False)
    pdf.bullet_point('Code cleanup and documentation', False)
    pdf.bullet_point('End-to-end demo presentation', False)

    # Task Mapping Page
    pdf.add_page()
    pdf.chapter_title('Task 4 Requirements Mapping')
    pdf.section_title('Task 1: Hybrid Generation Interaction')
    pdf.add_table(['Requirement', 'Experiment', 'Phase', 'Status'], [
        ['2 retrieval methods', 'BM25 + Contriever + Hybrid', 'Phase 3-4', 'Done'],
        ['2 generation models', 'LLaMA-3.1-8B + Qwen2.5-7B', 'Phase 4', 'Planned'],
        ['HotpotQA analysis', 'Recall@k vs Answer F1', 'Phase 3-4', 'In Progress'],
        ['Zero-shot vs Instruct', 'Qwen base vs Instruct', 'Phase 4', 'Planned'],
        ['Human evaluation', 'Crowd-sourcing', 'Phase 5', 'Planned']
    ])
    pdf.ln(3)
    pdf.section_title('Task 2: Multifaceted Mitigation')
    pdf.add_table(['Requirement', 'Experiment', 'Phase', 'Status'], [
        ['Quantify hallucination', 'FActScore metrics', 'Phase 5', 'Planned'],
        ['Cross-domain citation', 'Finance vs Law comparison', 'Phase 5', 'Planned'],
        ['Human evaluation', '50-sample stratified', 'Phase 5', 'Planned'],
        ['4 mitigation strategies', 'LLM/Vanilla/LangChain/Self-RAG', 'Phase 5', 'Planned']
    ])

    # Tips
    pdf.ln(5)
    pdf.chapter_title('Experiment Tips')
    pdf.set_font('Helvetica', '', 10)
    pdf.body_text('1. Retrieval Document Count: Recommend limiting to 5-9 documents per query to balance performance and computational cost.')
    pdf.body_text('2. Current Setting: k=3, can be adjusted based on HotpotQA experiment results.')
    pdf.body_text('3. Evaluation Framework: See Appendix for details.')
    pdf.body_text('4. Cross-domain Generalization: Pay attention to performance differences between Finance and Law domains.')

    # Save
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, 'project_timeline.pdf')
    pdf.output(pdf_path)
    print(f'PDF saved: {pdf_path}')
    return pdf_path


if __name__ == '__main__':
    create_pdf()
