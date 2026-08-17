import os
import tempfile
import requests
import time
import re

import streamlit as st
st.set_option("client.toolbarMode", "minimal")

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq

from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

try:
    SEMANTIC_SCHOLAR_API_KEY = st.secrets.get("SEMANTIC_SCHOLAR_API_KEY")
except Exception:
    SEMANTIC_SCHOLAR_API_KEY = None

if not SEMANTIC_SCHOLAR_API_KEY:
    SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

if not GROQ_API_KEY:
    st.error(
        "GROQ_API_KEY is missing. "
        "Please check your .env file."
    )
    st.stop()


# ============================================================
# 2. INITIALIZE GROQ MODEL
# ============================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=GROQ_API_KEY,
    max_tokens=1800
)


# ============================================================
# 3. STRUCTURED OUTPUT MODEL
# ============================================================

class PaperInsights(BaseModel):

    paper_title: str = Field(
        description=(
            "The exact title of the research paper as stated in the supplied content. "
            "Do not invent or paraphrase the title."
        )
    )

    overview: str = Field(
        description=(
            "Research context, domain, central problem, "
            "and main contribution."
        )
    )

    problem_statement: str = Field(
        description=(
            "Precise technical or scientific problem "
            "addressed by the paper."
        )
    )

    objective: str = Field(
        description=(
            "Primary research objective, research question, "
            "or hypothesis if explicitly stated."
        )
    )

    existing_approach: str = Field(
        description=(
            "Important existing methods, frameworks, "
            "models, algorithms, or baselines."
        )
    )

    proposed_method: str = Field(
        description=(
            "Proposed framework, architecture, model, "
            "algorithm, or technical contribution."
        )
    )

    methodology: str = Field(
        description=(
            "Experimental design and methodology including "
            "important technical implementation choices."
        )
    )

    dataset: str = Field(
        description=(
            "Datasets, data sources, benchmarks, "
            "evaluation protocol, and metrics."
        )
    )

    results: str = Field(
        description=(
            "Important quantitative or qualitative results, "
            "comparisons, and empirical evidence."
        )
    )

    limitations: str = Field(
        description=(
            "Author-stated limitations and clearly supported "
            "threats to validity."
        )
    )

    future_work: str = Field(
        description=(
            "Author-stated future work and research opportunities "
            "strongly supported by the paper."
        )
    )


# ============================================================
# 4. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RESINK | Research Intelligence",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# 4A. CONTACT / BRAND CONFIG
# ============================================================

FOUNDER_NAME = "Siddhu Parasa"
LINKEDIN_URL = "https://www.linkedin.com/in/siddhu-parasa/"
WHATSAPP_URL = "https://wa.me/919391757059"
EMAIL_ADDRESS = "siddhuparasa99@gmail.com"
GITHUB_URL = "https://github.com/siddhuparasa/RESINK"


# ============================================================
# 4B. GLOBAL STYLE
# ============================================================

st.markdown(
    """
    <style>

        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        html, body { background-color: #EEF2F7 !important; }

        :root {
    --resink-bg: #EEF2F7;              /* overall page */
    --resink-surface: #FFFFFF;         /* cards, navbar, inputs */
    --resink-border: #D6DEE8;          /* borders, dividers */
    
    --resink-text: #162033;            /* headings, primary text */
    --resink-muted: #5F6B7A;           /* descriptions, secondary text */

    --resink-primary: #4F46E5;         /* buttons, links, accents */
    --resink-primary-hover: #4338CA;   /* hover state */
    --resink-primary-soft: #EEF0FF;    /* subtle accent backgrounds */

    --resink-success: #059669;         /* success states */
    --resink-warning: #D97706;         /* warnings */
    --resink-error: #DC2626;           /* errors */
}

        html, body,
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > .main,
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"],
        [data-testid="stMainBlockContainer"] > div {
            background-color: #EEF2F7 !important;
            color: #162033 !important;
        }

        [data-testid="stHeader"] {
            background-color: #EEF2F7 !important;
        }

        [data-testid="stMainBlockContainer"] > div {
            background-color: transparent !important;
        }

        /* Force readable text color on native Streamlit widgets
           (radio options, file uploader, text input labels, etc.)
           regardless of light/dark theme settings. */
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] span,
        [data-testid="stWidgetLabel"] p,
        .stRadio label span,
        .stFileUploader label span,
        .stTextInput label span {
            color: var(--resink-text) !important;
        }

        .stFileUploader section {
            background-color: var(--resink-surface) !important;
            border: 1px dashed var(--resink-border) !important;
        }

        .stFileUploader section span,
        .stFileUploader section small {
            color: var(--resink-muted) !important;
        }

        .stTextInput input {
            color: var(--resink-text) !important;
            background-color: var(--resink-surface) !important;
            border: 1px solid var(--resink-border) !important;
        }

        .stTextInput input::placeholder {
            color: var(--resink-muted) !important;
            opacity: 1 !important;
        }

        /* ---------------- BUTTONS ---------------- */

        /* Catch every Streamlit button variant by data-testid,
           regardless of which component renders it:
           st.button, st.link_button, st.download_button,
           and the file uploader's "Browse files" button. */
        [data-testid^="stBaseButton"],
        [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stLinkButton"] a,
        .stButton > button,
        .stLinkButton > a,
        .stDownloadButton > button,
        button[kind="secondary"],
        a[kind="secondary"] {
            background-color: var(--resink-surface) !important;
            color: var(--resink-text) !important;
            border: 1px solid var(--resink-border) !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            box-shadow: none !important;
        }

        [data-testid^="stBaseButton"]:hover,
        [data-testid="stFileUploaderDropzone"] button:hover,
        [data-testid="stLinkButton"] a:hover,
        .stButton > button:hover,
        .stLinkButton > a:hover,
        .stDownloadButton > button:hover,
        button[kind="secondary"]:hover,
        a[kind="secondary"]:hover {
            background-color: var(--resink-primary-soft) !important;
            color: var(--resink-primary) !important;
            border-color: var(--resink-primary) !important;
        }

        [data-testid^="stBaseButton"] p,
        [data-testid="stLinkButton"] a p,
        [data-testid="stLinkButton"] a span,
        .stButton > button p,
        .stLinkButton > a p,
        .stLinkButton > a span {
            color: inherit !important;
        }

        /* Primary buttons: "Analyze Paper" */
        [data-testid^="stBaseButton-primary"],
        button[kind="primary"],
        a[kind="primary"] {
            background-color: var(--resink-primary) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            box-shadow: none !important;
        }

        [data-testid^="stBaseButton-primary"]:hover,
        button[kind="primary"]:hover,
        a[kind="primary"]:hover {
            background-color: var(--resink-primary-hover) !important;
        }

        [data-testid^="stBaseButton-primary"] p,
        button[kind="primary"] p {
            color: #ffffff !important;
        }

        #MainMenu, footer, header {
            visibility: hidden;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 0rem;
            max-width: 1080px;
        }

        /* ---------------- NAVBAR ---------------- */

.resink-navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 32px;
    background: color-mix(in srgb, var(--resink-surface) 80%, transparent);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid color-mix(in srgb, var(--resink-border) 70%, transparent);
    border-radius: 16px;
    margin-bottom: 32px;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05),
                0 0 0 1px rgba(255, 255, 255, 0.05) inset;
    transition: all 0.3s ease;
}

.resink-navbar:hover {
    border-color: color-mix(in srgb, var(--resink-primary) 30%, var(--resink-border));
    box-shadow: 0 8px 30px -4px rgba(0, 0, 0, 0.08);
}

.resink-logo {
    display: inline-flex;
    align-items: center;
    gap: 2px;
    font-size: 23px;
    font-weight: 800;
    letter-spacing: -0.4px;
    color: var(--resink-text);
    line-height: 1.1;
    cursor: pointer;
    user-select: none;
    transition: transform 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

.resink-logo:hover {
    transform: translateY(-1px);
}

.resink-logo span {
    background: linear-gradient(135deg, var(--resink-primary), #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
    filter: drop-shadow(0 2px 8px rgba(99, 102, 241, 0.25));
    transition: filter 0.3s ease;
}

.resink-logo:hover span {
    filter: drop-shadow(0 4px 12px rgba(99, 102, 241, 0.45));
}

.resink-nav-tagline {
    font-size: 12.5px;
    font-weight: 500;
    letter-spacing: 0.2px;
    color: var(--resink-muted);
    opacity: 0.85;
    margin-top: 3px;
    line-height: 1.3;
}
        /* ---------------- HERO ---------------- */

        .resink-hero {
            text-align: center;
            padding: 20px 10px 36px 10px;
        }

        .resink-hero h1 {
            font-size: 40px;
            font-weight: 800;
            color: var(--resink-text);
            margin-bottom: 10px;
            letter-spacing: -0.5px;
        }

        .resink-hero p {
            font-size: 23px;
            color: var(--resink-muted);
            max-width: 800px;
            margin: 0 auto;
            line-height: 1.6;
        }

        /* ---------------- SECTION CARD (native bordered container) ---------------- */

        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #FFFFFF !important;
            border: 1px solid #D6DEE8 !important;
            border-radius: 14px !important;
            box-shadow: 0 1px 3px rgba(22, 32, 51, 0.05) !important;
        }

        [data-testid="stVerticalBlockBorderWrapper"] > div {
            background-color: var(--resink-surface) !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlock"] {
            gap: 0.75rem !important;
        }

        .resink-card {
            background-color: var(--resink-surface);
            border: 1px solid var(--resink-border);
            border-radius: 14px;
            padding: 28px;
            margin-bottom: 20px;
        }

        .resink-section-title {
            font-size: 20px;
            font-weight: 700;
            color: var(--resink-text);
            margin-bottom: 4px;
        }

        .resink-section-subtitle {
            font-size: 14px;
            color: var(--resink-muted);
            margin-bottom: 18px;
        }

        /* ---------------- INSIGHT CARD ---------------- */

        .insight-card {
            background-color: var(--resink-surface);
            border: 1px solid var(--resink-border);
            border-left: 4px solid var(--resink-primary);
            border-radius: 10px;
            padding: 20px 22px;
            margin-bottom: 14px;
        }

        .insight-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 10px;
        }

        .insight-number {
            display: flex;
            align-items: center;
            justify-content: center;
            min-width: 28px;
            height: 28px;
            border-radius: 50%;
            background-color: var(--resink-primary-soft);
            color: var(--resink-primary);
            font-weight: 700;
            font-size: 13px;
        }

        .insight-title {
            font-size: 16px;
            font-weight: 700;
            color: var(--resink-text);
        }

        .insight-body {
            font-size: 14.5px;
            color: #475569;
            line-height: 1.65;
            margin-left: 40px;
        }

        /* ---------------- PAPER CARD ---------------- */

        .paper-card {
            background-color: var(--resink-surface);
            border: 1px solid var(--resink-border);
            border-radius: 10px;
            padding: 20px 22px;
            margin-bottom: 14px;
        }

        .paper-title {
            font-size: 15.5px;
            font-weight: 700;
            color: var(--resink-text);
            margin-bottom: 8px;
        }

        .paper-meta {
            font-size: 13.5px;
            color: var(--resink-muted);
            margin-bottom: 4px;
        }

        /* ---------------- FOOTER ---------------- */

        .resink-footer {
            margin-top: 48px;
            padding: 28px 10px 20px 10px;
            border-top: 1px solid var(--resink-border);
            text-align: center;
        }

        .resink-footer-links {
            display: flex;
            justify-content: center;
            gap: 28px;
            flex-wrap: wrap;
            margin-bottom: 16px;
        }

        .resink-footer-links a {
            font-size: 13.5px;
            font-weight: 600;
            color: var(--resink-muted);
            text-decoration: none;
            border-bottom: 1px solid transparent;
            padding-bottom: 2px;
        }

        .resink-footer-links a:hover {
            color: var(--resink-primary);
            border-bottom: 1px solid var(--resink-primary);
        }

        .resink-footer-credit {
            font-size: 13px;
            color: var(--resink-muted);
        }

        .resink-footer-credit b {
            color: var(--resink-text);
        }

    
        /* ---------- RESINK WIDGET THEME ---------- */

        .stButton > button[kind="primary"],
        button[data-testid="stBaseButton-primary"] {
            background-color: #4F46E5 !important;
            border: 1px solid #4F46E5 !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }

        .stButton > button[kind="primary"]:hover,
        button[data-testid="stBaseButton-primary"]:hover {
            background-color: #4338CA !important;
            border-color: #4338CA !important;
            color: #FFFFFF !important;
        }

        .stButton > button,
        button[data-testid="stBaseButton-secondary"] {
            border-color: #C7D2FE !important;
            color: #3730A3 !important;
            background-color: #FFFFFF !important;
            border-radius: 8px !important;
        }

        .stButton > button:hover,
        button[data-testid="stBaseButton-secondary"]:hover {
            border-color: #4F46E5 !important;
            color: #4338CA !important;
            background-color: #EEF0FF !important;
        }

        [data-testid="stRadio"] [role="radiogroup"] label {
            color: #162033 !important;
        }

        [data-testid="stRadio"] input[type="radio"] {
            accent-color: #4F46E5 !important;
        }

        [data-testid="stRadio"] [role="radio"][aria-checked="true"] {
            color: #4F46E5 !important;
        }

        [data-testid="stFileUploader"] section {
            background-color: #FFFFFF !important;
            border: 1px dashed #C7D2FE !important;
            border-radius: 10px !important;
        }

        [data-testid="stFileUploader"] section:hover {
            border-color: #4F46E5 !important;
            background-color: #FAFAFF !important;
        }

        [data-testid="stFileUploader"] button {
            border-color: #C7D2FE !important;
            color: #3730A3 !important;
            background-color: #FFFFFF !important;
        }

        [data-testid="stFileUploader"] button:hover {
            border-color: #4F46E5 !important;
            color: #4338CA !important;
        }

        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea {
            background-color: #FFFFFF !important;
            color: #162033 !important;
            border: 1px solid #D6DEE8 !important;
            border-radius: 8px !important;
        }

        [data-testid="stTextInput"] input:focus,
        [data-testid="stTextArea"] textarea:focus {
            border-color: #4F46E5 !important;
            box-shadow: 0 0 0 1px #4F46E5 !important;
        }

        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            border-color: #D6DEE8 !important;
        }

        [data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
            border-color: #4F46E5 !important;
            box-shadow: 0 0 0 1px #4F46E5 !important;
        }

        a {
            color: #4F46E5 !important;
        }

        a:hover {
            color: #4338CA !important;
        }

</style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 5. TOP NAVIGATION
# ============================================================

st.markdown(
    """
    <div class="resink-navbar">
        <a href="/" target="_self" class="resink-logo" style="text-decoration: none; color: inherit;">
            <span style="color: #000000; background: none; -webkit-text-fill-color: #000000;">RES</span><span>INK</span>
        </a>
        <div class="resink-nav-tagline">Research Intelligence Platform</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 5A. HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="resink-hero">
        <h1>Understand any research paper in minutes</h1>
        <p>
            Find the most relevant research papers related to your work.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 6. LOAD UPLOADED PDF
# ============================================================

def load_uploaded_pdf(uploaded_file):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(
            uploaded_file.getbuffer()
        )

        temp_path = temp_file.name

    try:

        loader = PyPDFLoader(
            temp_path
        )

        documents = loader.load()

        return documents

    finally:

        try:
            os.remove(temp_path)
        except OSError:
            pass


# ============================================================
# 7. LOAD PAPER FROM URL
# ============================================================

def load_paper_url(url):

    clean_url = (
        url.lower()
        .split("?")[0]
    )

    if clean_url.endswith(".pdf"):

        response = requests.get(
            url,
            timeout=30,
            headers={
                "User-Agent": "RESINK Research Analyzer"
            }
        )

        response.raise_for_status()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(
                response.content
            )

            temp_path = temp_file.name

        try:

            loader = PyPDFLoader(
                temp_path
            )

            documents = loader.load()

            return documents

        finally:

            try:
                os.remove(temp_path)
            except OSError:
                pass

    loader = WebBaseLoader(
        url
    )

    documents = loader.load()

    return documents


# ============================================================
# 8. PREPARE PAPER CONTENT
# ============================================================

def prepare_paper_text(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(
        documents
    )

    if not chunks:
        return ""

    chunk_texts = []

    for chunk in chunks:

        text = chunk.page_content.strip()

        if text:
            chunk_texts.append(text)

    if not chunk_texts:
        return ""

    # Sample the complete paper more evenly. The old approach
    # could miss the methodology and experimental evidence in
    # the middle of longer papers.
    max_chunks = 12

    if len(chunk_texts) <= max_chunks:
        selected_chunks = chunk_texts
    else:
        indices = []
        for i in range(max_chunks):
            index = round(
                i * (len(chunk_texts) - 1) / (max_chunks - 1)
            )
            if index not in indices:
                indices.append(index)
        selected_chunks = [chunk_texts[index] for index in indices]

    paper_text = "\n\n".join(selected_chunks)

    max_characters = 24000

    if len(paper_text) > max_characters:
        paper_text = paper_text[:max_characters]

    return paper_text


# ============================================================
# 9. ANALYZE RESEARCH PAPER
# ============================================================

def analyze_paper(paper_text):

    structured_llm = llm.with_structured_output(
        PaperInsights
    )

    prompt = f"""
You are RESINK, a research-intelligence system for researchers,
PhD scholars, graduate students, and academic professionals.

Analyze ONLY the research paper content supplied below.

Produce EXACTLY 10 academically rigorous insights that allow a
researcher to understand the paper's actual contribution,
technical mechanism, evidence, limitations, and research
opportunities.

This is NOT a generic student-level summary.

============================================================
EVIDENCE AND ACCURACY RULES
============================================================

Every factual statement MUST be directly supported by the
supplied paper content. Never infer or invent a framework,
model, dataset, metric, numerical result, comparison, novelty
claim, limitation, citation, or future direction.

If the requested information is absent, write exactly:
"Not specified in the paper."

Do not fill missing information using general knowledge.

============================================================
ANTI-REPETITION RULES
============================================================

Each insight must contribute NEW information.

Do not repeat the same fact, model, method, dataset, metric,
result, limitation, or contribution across multiple insights.
Do not paraphrase an earlier insight.
Do not restate the abstract in several sections.

Every sentence must add a concrete paper-specific fact,
relationship, comparison, mechanism, or evidence point.

Never produce a keyword followed by a generic sentence such as
"The paper uses X to improve performance." Explain what X does,
why it is used, and what evidence the paper provides — but ONLY
when those details are actually present in the paper.

============================================================
TECHNICAL SPECIFICITY
============================================================

Preserve exact terminology used by the authors.
When available, include concrete names and values for:

- frameworks
- models
- architectures
- algorithms
- components
- loss functions
- preprocessing methods
- training strategies
- optimization methods
- retrieval mechanisms
- datasets and splits
- benchmarks
- evaluation metrics
- baselines
- hyperparameters
- experimental settings
- ablations
- numerical results
- statistical tests

Do not replace technical terminology with vague descriptions.
Do not add technical details that are not in the paper.

============================================================
WRITING STANDARD
============================================================

Each insight should normally contain 2-4 complete,
information-dense sentences.

Avoid:
- keyword fragments
- generic filler
- vague claims
- unsupported interpretations
- repeated sentences
- empty statements such as "the approach is effective"
  without evidence

The output should read like a concise research briefing for a
PhD researcher, not a classroom explanation.

============================================================
PAPER TITLE
============================================================

First identify the exact paper title from the supplied paper content.
Return it verbatim as paper_title. If the title cannot be determined
from the supplied content, return exactly: "Not specified in the paper."
Do not invent or paraphrase the title.

============================================================
REQUIRED 10 INSIGHTS
============================================================

1. RESEARCH CONTEXT AND CONTRIBUTION

Identify the research domain, central problem, and specific
contribution. State what the authors actually introduce,
develop, evaluate, or demonstrate. Do not call it "novel"
unless the paper supports that claim.

2. PROBLEM FORMULATION AND RESEARCH GAP

Define the technical or scientific problem precisely. Explain
the specific gap, unresolved challenge, limitation, or
assumption motivating the work. Do not repeat Insight 1.

3. RESEARCH OBJECTIVE / QUESTION / HYPOTHESIS

State the explicit objective, research question, or hypothesis.
If none is explicitly provided, write "Not specified in the
paper." Do not manufacture one from context.

4. EXISTING METHODS AND BASELINES

Identify important prior methods, frameworks, models,
algorithms, or baselines discussed or evaluated. Explain the
relevant limitation or comparison point motivating the work.
Do not describe the proposed method here.

5. PROPOSED FRAMEWORK AND TECHNICAL MECHANISM

Explain HOW the proposed approach works. Identify its
architecture, components, algorithms, processing stages,
retrieval mechanisms, training strategy, or other technical
mechanisms explicitly described. This should be the most
technically informative insight.

6. METHODOLOGY AND EXPERIMENTAL DESIGN

Explain HOW the study was conducted and evaluated. Include
preprocessing, training, optimization, inference, experimental
setup, ablations, or protocol details that are actually stated.
Do not repeat the architecture description from Insight 5.

7. DATASETS AND EVALUATION PROTOCOL

Identify datasets or data sources, dataset sizes and splits
when available, benchmarks, evaluation tasks, and metrics.
Do not report results here.

8. RESULTS AND EMPIRICAL EVIDENCE

Report the strongest evidence supporting the claims. Prioritize
exact numerical results, benchmark scores, baseline comparisons,
ablations, efficiency measurements, or statistical evidence.
If numerical evidence is unavailable, do not invent it.

9. LIMITATIONS AND THREATS TO VALIDITY

Report limitations explicitly stated by the authors. Mention a
methodological constraint only when clearly supported by the
paper. Do not invent generic limitations.

10. RESEARCH OPPORTUNITIES AND FUTURE DIRECTIONS

First report future work explicitly proposed by the authors.
Then identify additional research opportunities ONLY when they
follow directly from a stated limitation, unresolved result,
or open question. Do not invent unsupported directions.

============================================================
FINAL INTERNAL QUALITY CHECK
============================================================

Before returning the structured output, verify:

1. Exactly 10 insights are present.
2. Every claim is supported by the supplied paper text.
3. No insight repeats another insight's main information.
4. No insight is merely a keyword plus a generic sentence.
5. Technical names and numerical evidence are preserved.
6. Results contain actual evidence when available.
7. Missing information is marked "Not specified in the paper."
8. The writing is researcher-oriented and technically precise.

============================================================
RESEARCH PAPER CONTENT
============================================================

{paper_text}
"""

    last_error = None

    for attempt in range(3):

        try:

            result = structured_llm.invoke(
                prompt
            )

            return result.model_dump()

        except Exception as e:

            last_error = e

            if attempt < 2:

                time.sleep(
                    2 * (attempt + 1)
                )

            else:

                raise last_error


# ============================================================
# 10. EXTRACT PAPER TITLE FOR RELATED-PAPER RETRIEVAL
# ============================================================

def normalize_title(title):

    title = title.lower()
    title = re.sub(r"[^a-z0-9]+", " ", title)
    return " ".join(title.split())


def title_similarity(title_a, title_b):

    a = set(normalize_title(title_a).split())
    b = set(normalize_title(title_b).split())

    if not a or not b:
        return 0.0

    return len(a & b) / max(len(a), len(b))


# ============================================================
# 11. SEARCH SEMANTIC SCHOLAR FOR THE ACTUAL PAPER
# ============================================================

def find_seed_paper(paper_title):

    if not paper_title:
        return None

    if paper_title == "Not specified in the paper.":
        return None

    api_url = (
        "https://api.semanticscholar.org/graph/v1/paper/search"
    )

    headers = {
        "x-api-key": SEMANTIC_SCHOLAR_API_KEY,
        "User-Agent": "RESINK Research Analyzer"
    }

    params = {
        "query": f'"{paper_title}"',
        "limit": 10,
        "fields": "paperId,title,authors,year,url,openAccessPdf"
    }

    response = requests.get(
        api_url,
        params=params,
        timeout=30,
        headers=headers
    )

    if response.status_code != 200:
        raise Exception(
            f"Semantic Scholar API error "
            f"{response.status_code}: {response.text}"
        )

    candidates = response.json().get("data", [])

    if not candidates:
        return None

    # Prefer an exact title match. This prevents RESINK from using
    # an unrelated paper merely because it shares keywords.
    target = normalize_title(paper_title)

    for candidate in candidates:

        candidate_title = candidate.get("title", "")

        if normalize_title(candidate_title) == target:
            return candidate

    # Otherwise require strong title overlap before accepting a seed.
    scored = []

    for candidate in candidates:

        score = title_similarity(
            paper_title,
            candidate.get("title", "")
        )

        scored.append((score, candidate))

    scored.sort(
        key=lambda item: item[0],
        reverse=True
    )

    if scored and scored[0][0] >= 0.75:
        return scored[0][1]

    return None


# ============================================================
# 12. GET PAPERS ACTUALLY RELATED TO THE SEED PAPER
# ============================================================

def get_semantic_scholar_recommendations(seed_paper_id):

    api_url = (
        "https://api.semanticscholar.org/"
        "recommendations/v1/papers/forpaper/"
        + seed_paper_id
    )

    params = {
        "from": "recent",
        "limit": 20,
        "fields": (
            "title,authors,year,url,openAccessPdf"
        )
    }

    headers = {
        "x-api-key": SEMANTIC_SCHOLAR_API_KEY,
        "User-Agent": "RESINK Research Analyzer"
    }

    response = requests.get(
        api_url,
        params=params,
        timeout=30,
        headers=headers
    )

    if response.status_code == 429:

        retry_after = response.headers.get(
            "Retry-After"
        )

        try:
            wait_seconds = min(
                int(retry_after),
                10
            ) if retry_after else 3
        except ValueError:
            wait_seconds = 3

        time.sleep(wait_seconds)

        response = requests.get(
            api_url,
            params=params,
            timeout=30,
            headers=headers
        )

    if response.status_code != 200:
        raise Exception(
            f"Semantic Scholar Recommendations API error "
            f"{response.status_code}: {response.text}"
        )

    return response.json().get(
        "recommendedPapers",
        []
    )


# ============================================================
# 13. GET PAPER LINK
# ============================================================

def get_paper_link(paper):

    open_access_pdf = paper.get("openAccessPdf")

    if open_access_pdf:

        pdf_url = open_access_pdf.get("url")

        if pdf_url:
            return pdf_url

    url = paper.get("url")

    if url:
        return url

    return None


# ============================================================
# 14. GET PAPER AUTHORS
# ============================================================

def get_authors(paper):

    authors = paper.get("authors", [])

    names = []

    for author in authors[:4]:

        name = author.get("name")

        if name:
            names.append(name)

    if not names:
        return "Authors unavailable"

    return ", ".join(names)


# ============================================================
# 15. FIND RELATED PAPERS
# ============================================================

def get_related_papers(insights, paper_text):

    paper_title = insights.get(
        "paper_title",
        ""
    )

    seed_paper = find_seed_paper(
        paper_title
    )

    if not seed_paper:
        raise Exception(
            "RESINK could not confidently match this paper to "
            "a Semantic Scholar record. Related papers were not "
            "shown rather than returning unrelated keyword matches."
        )

    recommendations = get_semantic_scholar_recommendations(
        seed_paper["paperId"]
    )

    unique_papers = []
    seen_ids = {seed_paper["paperId"]}

    for paper in recommendations:

        paper_id = paper.get("paperId")

        if not paper_id or paper_id in seen_ids:
            continue

        seen_ids.add(paper_id)

        paper["display_name"] = paper.get(
            "title",
            "Unknown Title"
        )

        paper["publication_year"] = paper.get(
            "year",
            "Year unavailable"
        )

        unique_papers.append(paper)

        if len(unique_papers) == 10:
            break

    return unique_papers


# ============================================================
# 15. INPUT SECTION
# ============================================================

with st.container(border=True):

    st.markdown(
        """
        <div class="resink-section-title">Analyze a Research Paper</div>
        <div class="resink-section-subtitle">
            Upload a PDF or paste a link to a research paper to generate
            a structured research intelligence brief.
        </div>
        """,
        unsafe_allow_html=True
    )

    input_method = st.radio(
        "Choose your input:",
        [
            "Upload PDF",
            "Research Paper URL"
        ],
        horizontal=True,
        label_visibility="collapsed"
    )


    # ============================================================
    # 16. PDF INPUT
    # ============================================================

    if input_method == "Upload PDF":

        uploaded_file = st.file_uploader(
            "Upload your research paper PDF",
            type=["pdf"]
        )

        if uploaded_file:

            if st.button(
                "Analyze Paper",
                type="primary"
            ):

                st.session_state.pop(
                    "insights",
                    None
                )

                st.session_state.pop(
                    "related_papers",
                    None
                )

                with st.spinner(
                    "Reading and analyzing your research paper..."
                ):

                    try:

                        documents = (
                            load_uploaded_pdf(
                                uploaded_file
                            )
                        )

                        paper_text = (
                            prepare_paper_text(
                                documents
                            )
                        )

                        if not paper_text:

                            st.error(
                                "No readable text was found "
                                "in the PDF."
                            )

                            st.stop()

                        insights = (
                            analyze_paper(
                                paper_text
                            )
                        )

                        st.session_state[
                            "insights"
                        ] = insights

                        st.session_state[
                            "paper_text"
                        ] = paper_text

                    except Exception as e:

                        st.error(
                            f"Could not analyze the paper: {e}"
                        )


    # ============================================================
    # 17. URL INPUT
    # ============================================================

    else:

        paper_url = st.text_input(
            "Paste your research paper URL",
            placeholder="https://arxiv.org/..."
        )

        if st.button(
            "Analyze Paper",
            type="primary"
        ):

            if not paper_url:

                st.warning(
                    "Please enter a research paper URL."
                )

            else:

                st.session_state.pop(
                    "insights",
                    None
                )

                st.session_state.pop(
                    "related_papers",
                    None
                )

                with st.spinner(
                    "Reading and analyzing your research paper..."
                ):

                    try:

                        documents = (
                            load_paper_url(
                                paper_url
                            )
                        )

                        paper_text = (
                            prepare_paper_text(
                                documents
                            )
                        )

                        if not paper_text:

                            st.error(
                                "No readable text was found "
                                "at this URL."
                            )

                            st.stop()

                        insights = (
                            analyze_paper(
                                paper_text
                            )
                        )

                        st.session_state[
                            "insights"
                        ] = insights

                        st.session_state[
                            "paper_text"
                        ] = paper_text

                    except Exception as e:

                        st.error(
                            f"Could not analyze the paper: {e}"
                        )


# ============================================================
# 18. DISPLAY RESEARCH INTELLIGENCE BRIEF
# ============================================================

if "insights" in st.session_state:

    insights = st.session_state[
        "insights"
    ]

    st.markdown(
        """
        <div class="resink-section-title" style="margin-top: 8px;">
            Research Intelligence Brief
        </div>
        <div class="resink-section-subtitle">
            A structured, researcher-grade breakdown of the paper.
        </div>
        """,
        unsafe_allow_html=True
    )

    insight_titles = [

        ("overview", "Research Context and Contribution"),
        ("problem_statement", "Problem Formulation"),
        ("objective", "Research Objective and Hypothesis"),
        ("existing_approach", "Existing Methods and Baselines"),
        ("proposed_method", "Proposed Framework and Technical Contribution"),
        ("methodology", "Methodology and Experimental Design"),
        ("dataset", "Datasets and Evaluation Protocol"),
        ("results", "Results and Empirical Evidence"),
        ("limitations", "Limitations and Threats to Validity"),
        ("future_work", "Research Opportunities and Future Directions")

    ]

    for index, (key, title) in enumerate(insight_titles, start=1):

        insight = insights.get(
            key,
            "Not specified in the paper."
        )

        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-header">
                    <div class="insight-number">{index}</div>
                    <div class="insight-title">{title}</div>
                </div>
                <div class="insight-body">{insight}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ========================================================
    # RELATED RESEARCH PAPERS
    # ========================================================

    with st.container(border=True):

        st.markdown(
            """
            <div class="resink-section-title">Related Research Papers</div>
            <div class="resink-section-subtitle">
                Discover related research papers relevant to the analyzed paper.
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "Find Related Research Papers"
        ):

            with st.spinner(
                "Searching academic literature..."
            ):

                try:

                    papers = (
                        get_related_papers(
                            insights,
                            st.session_state.get(
                                "paper_text",
                                ""
                            )
                        )
                    )

                    if papers:

                        st.session_state[
                            "related_papers"
                        ] = papers

                    else:

                        st.warning(
                            "No related research papers "
                            "were found."
                        )

                except Exception as e:

                    st.error(
                        f"Could not find related papers: {e}"
                    )


# ============================================================
# 19. DISPLAY RELATED PAPERS
# ============================================================

if "related_papers" in st.session_state:

    papers = st.session_state[
        "related_papers"
    ]

    st.markdown(
        f"""
        <div class="resink-section-title" style="margin-top: 8px;">
            Related Papers
        </div>
        """,
        unsafe_allow_html=True
    )

    for index, paper in enumerate(
        papers,
        start=1
    ):

        title = paper.get(
            "display_name",
            "Unknown Title"
        )

        year = paper.get(
            "publication_year",
            "Year unavailable"
        )

        authors = get_authors(
            paper
        )

        link = get_paper_link(
            paper
        )

        st.markdown(
            f"""
            <div class="paper-card">
                <div class="paper-title">{index}. {title}</div>
                <div class="paper-meta"><b>Authors:</b> {authors}</div>
                <div class="paper-meta"><b>Publication Year:</b> {year}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if link:

            st.link_button(
                "Open Research Paper",
                link
            )

        else:

            st.warning(
                "Research paper link unavailable."
            )


# ============================================================
# 20. FOOTER
# ============================================================

st.markdown(
    f"""
    <div class="resink-footer">
        <div class="resink-footer-links">
            <a href="{LINKEDIN_URL}" target="_blank">LinkedIn</a>
            <a href="{WHATSAPP_URL}" target="_blank">WhatsApp</a>
            <a href="mailto:{EMAIL_ADDRESS}">Email</a>
            <a href="{GITHUB_URL}" target="_blank">GitHub</a>
        </div>
        <div class="resink-footer-credit">
            RESINK &nbsp;&middot;&nbsp; Founded and developed by <b>{FOUNDER_NAME}</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)