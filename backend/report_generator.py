"""
VoiceGuard — PDF Forensic Investigation Report Generator
─────────────────────────────────────────────────────────────────────────
Implements PDF compilation using ReportLab for banking security SOC reviews.
Follows the structural and compliance layout requirements from the UI.
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute total page count and draw
    corporate headers, footers, and border accents.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        
        # Premium design color system (Dark Navy, Cerulean, Accent Gray)
        navy_dark = colors.HexColor("#060c18")
        slate_gray = colors.HexColor("#4a6a8a")
        border_blue = colors.HexColor("#1e3a5f")
        text_light = colors.HexColor("#8fa8c8")
        
        # Draw top header on all pages
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(slate_gray)
        self.drawString(36, 756, "VOICEGUARD™ FORENSIC INVESTIGATION REPORT")
        self.drawRightString(576, 756, "RESTRICTED — BANKING FRAUD INTELLIGENCE")
        
        # Header thin rule
        self.setStrokeColor(border_blue)
        self.setLineWidth(0.75)
        self.line(36, 748, 576, 748)
        
        # Footer thin rule
        self.line(36, 45, 576, 45)
        
        # Draw bottom footer
        self.setFont("Helvetica", 7)
        self.drawString(36, 32, "SECURITY CLASSIFICATION: RESTRICTED  •  RBI Circular RBI/2023-24/73 Compliant")
        self.drawRightString(576, 32, f"Page {self._pageNumber} of {page_count}")
        
        self.drawString(36, 22, "NPCI Joint Cybersecurity Framework Aligned  •  UCO Bank Track Certified")
        self.drawRightString(576, 22, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (IST)")
        
        self.restoreState()


def generate_investigation_report(data: dict, output_path: str):
    """
    Builds the beautiful PDF forensic report and writes it to output_path.
    """
    # Create the document template with 0.5 in margins (36 pt)
    # Printable area is 540 pt wide and 684 pt tall
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom Corporate Typography & Colors
    c_navy = colors.HexColor("#0a1628")
    c_slate = colors.HexColor("#1e3a5f")
    c_red = colors.HexColor("#ff4560")
    c_amber = colors.HexColor("#ffb300")
    c_green = colors.HexColor("#00c896")
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=c_navy,
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#4a6a8a"),
        spaceAfter=15
    )
    
    h2_style = ParagraphStyle(
        'SecHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=c_navy,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#2a3649")
    )
    
    body_bold = ParagraphStyle(
        'BodyBoldCustom',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    mono_style = ParagraphStyle(
        'MonoCustom',
        parent=styles['Code'],
        fontName='Courier',
        fontSize=8,
        leading=10,
        textColor=c_slate
    )
    
    mono_bold = ParagraphStyle(
        'MonoBoldCustom',
        parent=mono_style,
        fontName='Courier-Bold'
    )
    
    story = []

    # 1. Header Title & Subtitle
    story.append(Spacer(1, 10))
    story.append(Paragraph("VOICEGUARD™ FORENSIC INVESTIGATION REPORT", title_style))
    story.append(Paragraph("Automated Biometric Voice Authenticity & Deep-Fake Analysis • Bank-Level SOC Audit Trail", subtitle_style))
    
    # 2. Executive Verdict Panel (Callout box with thick left border)
    risk = str(data.get("risk_level", "LOW")).upper()
    verdict = str(data.get("verdict_label", "No Threat Detected")).upper()
    score = data.get("fraud_score", 0.0)
    confidence = data.get("confidence", 0.0)
    
    if risk in ("CRITICAL", "HIGH"):
        bg_col = colors.HexColor("#fff5f5")
        border_col = c_red
        text_col = c_red
    elif risk == "ELEVATED":
        bg_col = colors.HexColor("#fffdf5")
        border_col = c_amber
        text_col = colors.HexColor("#b78103")
    else:
        bg_col = colors.HexColor("#f5fcf9")
        border_col = c_green
        text_col = colors.HexColor("#008a65")
        
    verdict_text = f"<b>VERDICT: {verdict}</b><br/>Overall Fraud Risk Score: {score}% &nbsp;|&nbsp; Model Confidence: {confidence}% &nbsp;|&nbsp; Risk Level: {risk}"
    verdict_para = Paragraph(verdict_text, ParagraphStyle('VerdictStyle', parent=body_style, fontSize=9.5, leading=14, textColor=text_col))
    
    # Create the callout container
    verdict_table = Table([[verdict_para]], colWidths=[540])
    verdict_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg_col),
        ('PADDING', (0,0), (-1,-1), 10),
        ('LINELEFT', (0,0), (-1,-1), 3.5, border_col),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 12))

    # 3. Case Identifiers & Parameters Grid
    story.append(Paragraph("1. CASE INFORMATION", h2_style))
    case_info_data = [
        [
            Paragraph("Case Reference ID:", body_bold),
            Paragraph(str(data.get("case_id")), mono_bold),
            Paragraph("Analysis Timestamp:", body_bold),
            Paragraph(str(data.get("timestamp")), mono_style)
        ],
        [
            Paragraph("Uploaded File:", body_bold),
            Paragraph(str(data.get("filename")), mono_style),
            Paragraph("Audio Format / Codec:", body_bold),
            Paragraph(str(data.get("audio_format")), mono_style)
        ],
        [
            Paragraph("Sample Rate (Native):", body_bold),
            Paragraph(str(data.get("sample_rate")), mono_style),
            Paragraph("Audio Duration:", body_bold),
            Paragraph(f"{data.get('duration')} seconds", mono_style)
        ],
        [
            Paragraph("Channel Configuration:", body_bold),
            Paragraph(str(data.get("channels")), mono_style),
            Paragraph("Analysis Model Version:", body_bold),
            Paragraph(str(data.get("model_version")), mono_style)
        ]
    ]
    
    case_table = Table(case_info_data, colWidths=[110, 160, 110, 160])
    case_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e1e6eb")),
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f8fafc")),
        ('BACKGROUND', (2,0), (2,-1), colors.HexColor("#f8fafc")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(case_table)
    story.append(Spacer(1, 10))

    # 4. Forensic Threat Intelligence Summary
    story.append(Paragraph("2. THREAT INTELLIGENCE SUMMARY", h2_style))
    threat_intel_data = [
        [
            Paragraph("Threat Category Matched:", body_bold),
            Paragraph(str(data.get("threat_type")), body_style),
            Paragraph("Synthetic Clone Confidence:", body_bold),
            Paragraph(f"{data.get('synthetic_confidence')}%", body_style)
        ],
        [
            Paragraph("Sophistication Signature:", body_bold),
            Paragraph(str(data.get("sophistication")), body_style),
            Paragraph("Replay-Attack Indicators:", body_bold),
            Paragraph(str(data.get("replay_indicators")), body_style)
        ],
        [
            Paragraph("Primary Detector Trigger:", body_bold),
            Paragraph(str(data.get("primary_trigger")), body_style),
            Paragraph("Secondary Detector Trigger:", body_bold),
            Paragraph(str(data.get("secondary_trigger")), body_style)
        ]
    ]
    
    threat_table = Table(threat_intel_data, colWidths=[130, 140, 130, 140])
    threat_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e1e6eb")),
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor("#f8fafc")),
        ('BACKGROUND', (2,0), (2,-1), colors.HexColor("#f8fafc")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(threat_table)
    story.append(Spacer(1, 10))

    # 5. Explainability (Top Contributing Biometric Factors)
    story.append(Paragraph("3. BIOMETRIC EXPLAINABILITY SIGNATURES", h2_style))
    factors = data.get("top_factors", [])
    factor_items = []
    
    if not factors:
        factor_items.append([Paragraph("• No anomalous features detected above critical decision boundary threshold.", body_style)])
    else:
        for f in factors:
            factor_items.append([Paragraph(f"• <b>{f}</b>: Indicated anomaly in biometric/replay stream matching spoofing characteristics.", body_style)])
            
    factor_table = Table(factor_items, colWidths=[540])
    factor_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('PADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(factor_table)
    story.append(Spacer(1, 10))

    # 6. Chunk-by-Chunk Scoring Summary
    story.append(Paragraph("4. SEGMENTED AUDIO ANALYSIS AUDIT TRAIL", h2_style))
    chunk_results = data.get("chunk_results", [])
    
    if not chunk_results:
        chunk_table_data = [[Paragraph("No segmented results available. Verify that the file meets minimum speech duration.", body_style)]]
        chunk_table = Table(chunk_table_data, colWidths=[540])
        chunk_table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e1e6eb")),
            ('PADDING', (0,0), (-1,-1), 8),
        ]))
    else:
        chunk_headers = [
            Paragraph("<b>Segment</b>", body_bold),
            Paragraph("<b>Time Window (sec)</b>", body_bold),
            Paragraph("<b>Biometric Score</b>", body_bold),
            Paragraph("<b>Deep Feature Score</b>", body_bold),
            Paragraph("<b>Fused Score</b>", body_bold),
            Paragraph("<b>Verdict</b>", body_bold)
        ]
        chunk_table_data = [chunk_headers]
        
        for c in chunk_results:
            c_risk = c.get("verdict", "CLEAN")
            if c_risk == "FRAUD":
                v_col = c_red
            elif c_risk == "SUSPICIOUS":
                v_col = c_amber
            else:
                v_col = colors.HexColor("#008a65")
                
            chunk_table_data.append([
                Paragraph(f"Segment #{c.get('index')}", body_style),
                Paragraph(f"{c.get('start')}s – {c.get('end')}s", mono_style),
                Paragraph(f"{c.get('bio_score')}%", mono_style),
                Paragraph(f"{c.get('deep_score')}%", mono_style),
                Paragraph(f"{c.get('score')}%", mono_bold),
                Paragraph(f"<font color='{v_col}'><b>{c_risk}</b></font>", body_style)
            ])
            
        chunk_table = Table(chunk_table_data, colWidths=[70, 110, 90, 95, 85, 90])
        chunk_table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e1e6eb")),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f8fafc")),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('PADDING', (0,0), (-1,-1), 4),
        ]))
        
    story.append(chunk_table)
    story.append(Spacer(1, 10))

    # 7. Regulatory Incident Recommendation
    story.append(Paragraph("5. SYSTEM INCIDENT ACTION PLAN & RECOMMENDATIONS", h2_style))
    rec_text = data.get("recommendation", "Awaiting review.")
    
    rec_box_data = [
        [
            Paragraph("<b>INCIDENT RESPONSE PROTOCOL:</b>", ParagraphStyle('RecH', parent=body_bold, textColor=colors.HexColor("#0f1e35"))),
        ],
        [
            Paragraph(rec_text, ParagraphStyle('RecB', parent=body_style, leading=12)),
        ]
    ]
    rec_box = Table(rec_box_data, colWidths=[540])
    rec_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f5f7fa")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#ccd4db")),
        ('PADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    
    # Bundle the recommendations and markings together so they don't break across pages
    story.append(KeepTogether([rec_box, Spacer(1, 15)]))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
