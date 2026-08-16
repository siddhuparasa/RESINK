import os
import tempfile
import requests
import time

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
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=GROQ_API_KEY,
    max_tokens=1400
)


# ============================================================
# 3. STRUCTURED OUTPUT MODEL
# ============================================================

class PaperInsights(BaseModel):

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
GITHUB_URL = "https://github.com/siddhuparasa"  # update to your actual GitHub profile


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

        :root {
            --resink-bg: #f7f8fa;
            --resink-surface: #ffffff;
            --resink-border: #e5e7eb;
            --resink-text: #111827;
            --resink-muted: #6b7280;
            --resink-primary: #1d4ed8;
            --resink-primary-soft: #eff4ff;
        }

        .stApp {
            background-color: var(--resink-bg);
            color: var(--resink-text);
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
            background-color: #1a3fc4 !important;
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
            padding: 14px 28px;
            background-color: var(--resink-surface);
            border: 1px solid var(--resink-border);
            border-radius: 12px;
            margin-bottom: 32px;
        }

        .resink-logo {
            font-size: 22px;
            font-weight: 800;
            letter-spacing: 0.5px;
            color: var(--resink-text);
        }

        .resink-logo span {
            color: var(--resink-primary);
        }

        .resink-nav-tagline {
            font-size: 13px;
            color: var(--resink-muted);
            font-weight: 500;
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
            font-size: 16px;
            color: var(--resink-muted);
            max-width: 620px;
            margin: 0 auto;
            line-height: 1.6;
        }

        /* ---------------- SECTION CARD ---------------- */

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
            color: #374151;
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
        <div class="resink-logo">RES<span>INK</span></div>
        <div class="resink-nav-tagline">Research Intelligence Platform</div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 5A. HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="resink-hero">
        <h1>Understand any research paper in minutes</h1>
        <p>
            RESINK reads academic papers and produces a rigorous,
            researcher-grade briefing covering contribution, methodology,
            results, limitations, and related literature.
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

            chunk_texts.append(
                text
            )

    if not chunk_texts:
        return ""

    max_chunks = 7

    if len(chunk_texts) <= max_chunks:

        selected_chunks = chunk_texts

    else:

        first_chunks = chunk_texts[:3]

        middle_chunk = [
            chunk_texts[
                len(chunk_texts) // 2
            ]
        ]

        last_chunks = chunk_texts[-3:]

        selected_chunks = (
            first_chunks
            + middle_chunk
            + last_chunks
        )

    paper_text = "\n\n".join(
        selected_chunks
    )

    max_characters = 13000

    if len(paper_text) > max_characters:

        paper_text = paper_text[
            :max_characters
        ]

    return paper_text


# ============================================================
# 9. ANALYZE RESEARCH PAPER
# ============================================================

def analyze_paper(paper_text):

    structured_llm = llm.with_structured_output(
        PaperInsights
    )

    prompt = f"""
You are RESINK, an advanced AI research-intelligence
system designed for researchers, PhD scholars,
graduate students, and academic professionals.

Analyze the research paper content below.

Your objective is NOT to produce a generic
student-level summary.

Instead, produce a technically rigorous research briefing
that allows an experienced researcher to understand the
paper's contribution, technical design, evidence, novelty,
limitations, and research opportunities within approximately
1-2 minutes.

Generate EXACTLY 10 insights.

Each insight should normally contain 2-4 concise sentences.

Use academically precise terminology.

Preserve technical names and terminology used by the authors.

When the paper mentions specific:

- frameworks
- models
- architectures
- algorithms
- techniques
- datasets
- benchmarks
- evaluation metrics
- experimental settings
- statistical methods
- baselines
- tools
- libraries
- protocols

include the relevant names and technical details.

Do NOT replace technical terminology with vague descriptions.

For example, instead of:

"The authors use a machine learning model."

prefer:

"The authors employ a Transformer-based architecture
with self-attention to model long-range dependencies."

ONLY make such claims when supported by the paper.

Do NOT invent:

- frameworks
- algorithms
- datasets
- metrics
- numerical results
- citations
- novelty claims
- experimental findings

If information is unavailable, write:

"Not specified in the paper."

The analysis should help a researcher answer:

"What exactly did the authors contribute?"

"What technical mechanism enables the contribution?"

"What evidence supports their claims?"

"How does the methodology differ from existing approaches?"

"What are the important limitations?"

"What research opportunities remain?"

Do not merely restate the abstract.

============================================================
REQUIRED 10 INSIGHTS
============================================================

1. RESEARCH CONTEXT AND CONTRIBUTION

Identify the research domain, central research problem,
and specific contribution of the paper.

Mention the proposed framework, model, system, or
architecture when applicable.

------------------------------------------------------------

2. PROBLEM FORMULATION

Precisely define the technical or scientific problem.

Identify the research gap, unresolved challenge,
limitation, or assumption motivating the work.

------------------------------------------------------------

3. RESEARCH OBJECTIVE AND HYPOTHESIS

Identify the primary research objective.

If the paper explicitly provides a hypothesis,
research question, or formal objective, include it.

Do not invent one.

------------------------------------------------------------

4. EXISTING METHODS AND BASELINES

Identify important existing approaches, frameworks,
models, algorithms, or baseline systems.

Explain what limitation motivates the proposed approach.

------------------------------------------------------------

5. PROPOSED FRAMEWORK AND TECHNICAL CONTRIBUTION

Describe the proposed architecture, framework,
algorithm, model, or system.

Mention important components, techniques,
architectural choices, or mechanisms.

This should be one of the most technically informative insights.

------------------------------------------------------------

6. METHODOLOGY AND EXPERIMENTAL DESIGN

Explain how the approach is implemented and evaluated.

Mention relevant choices such as:

- preprocessing
- feature extraction
- architecture
- training strategy
- optimization
- retrieval
- inference
- experimental setup
- ablation studies

Only include information supported by the paper.

------------------------------------------------------------

7. DATASETS AND EVALUATION PROTOCOL

Identify datasets, data sources, dataset sizes where available,
train/validation/test configuration, benchmarks, and metrics.

Mention important evaluation protocols.

------------------------------------------------------------

8. RESULTS AND EMPIRICAL EVIDENCE

Summarize the most important findings.

Prioritize:

- performance improvements
- benchmark results
- numerical results
- baseline comparisons
- ablation findings
- efficiency results
- statistical evidence

Include numerical values when explicitly available.

Do not exaggerate the results.

------------------------------------------------------------

9. LIMITATIONS AND THREATS TO VALIDITY

Identify limitations explicitly acknowledged by the authors.

Also identify methodological constraints only when clearly
supported by the paper.

Examples include:

- dataset limitations
- generalization concerns
- computational constraints
- evaluation limitations
- dependency on assumptions
- reproducibility concerns

Do not invent limitations.

------------------------------------------------------------

10. RESEARCH OPPORTUNITIES AND FUTURE DIRECTIONS

Summarize future work proposed by the authors.

Then identify promising research directions only when they
are strongly supported by the paper's limitations, results,
or unresolved problems.

Focus on opportunities involving:

- improved methodology
- stronger evaluation
- broader generalization
- new datasets
- architectural extensions
- efficiency improvements
- unresolved research questions

Do not invent unsupported future work.

============================================================
QUALITY REQUIREMENTS
============================================================

The final insights must be:

- academically rigorous
- technically specific
- concise
- precise
- evidence-grounded
- non-repetitive
- researcher-oriented

Avoid generic phrases such as:

"very useful"
"highly effective"
"advanced technology"
"great results"

unless supported by evidence from the paper.

The final result should resemble a concise research briefing,
not a classroom explanation.

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
# 10. CREATE OPENALEX SEARCH QUERY
# ============================================================

def create_openalex_query(insights):

    parts = [

        insights.get(
            "overview",
            ""
        ),

        insights.get(
            "problem_statement",
            ""
        ),

        insights.get(
            "proposed_method",
            ""
        ),

        insights.get(
            "methodology",
            ""
        )

    ]

    query = " ".join(
        parts
    )

    query = " ".join(
        query.split()
    )

    max_query_length = 700

    if len(query) > max_query_length:

        query = query[
            :max_query_length
        ]

    return query


# ============================================================
# 11. SEARCH OPENALEX
# ============================================================

def search_openalex(query):

    api_url = (
        "https://api.openalex.org/works"
    )

    params = {

        "search": query,

        "per-page": 10

    }

    response = requests.get(
        api_url,
        params=params,
        timeout=30,
        headers={
            "User-Agent": "RESINK Research Analyzer"
        }
    )

    if response.status_code != 200:

        raise Exception(
            f"OpenAlex API error "
            f"{response.status_code}: "
            f"{response.text}"
        )

    data = response.json()

    return data.get(
        "results",
        []
    )


# ============================================================
# 12. GET PAPER LINK
# ============================================================

def get_paper_link(paper):

    best_oa_location = paper.get(
        "best_oa_location"
    )

    if best_oa_location:

        pdf_url = best_oa_location.get(
            "pdf_url"
        )

        if pdf_url:
            return pdf_url

        landing_url = (
            best_oa_location.get(
                "landing_page_url"
            )
        )

        if landing_url:
            return landing_url

    primary_location = paper.get(
        "primary_location"
    )

    if primary_location:

        landing_url = (
            primary_location.get(
                "landing_page_url"
            )
        )

        if landing_url:
            return landing_url

    doi = paper.get(
        "doi"
    )

    if doi:
        return doi

    return paper.get(
        "id"
    )


# ============================================================
# 13. GET PAPER AUTHORS
# ============================================================

def get_authors(paper):

    authorships = paper.get(
        "authorships",
        []
    )

    names = []

    for authorship in authorships[:4]:

        author = authorship.get(
            "author"
        )

        if author:

            name = author.get(
                "display_name"
            )

            if name:

                names.append(
                    name
                )

    if not names:

        return "Authors unavailable"

    return ", ".join(
        names
    )


# ============================================================
# 14. FIND  RELATED PAPERS
# ============================================================

def get_related_papers(insights):

    query = create_openalex_query(
        insights
    )

    papers = search_openalex(
        query
    )

    if not papers:

        fallback_query = (
            insights.get(
                "proposed_method",
                ""
            )
            + " "
            +
            insights.get(
                "problem_statement",
                ""
            )
        )

        fallback_query = " ".join(
            fallback_query.split()
        )

        fallback_query = fallback_query[
            :400
        ]

        papers = search_openalex(
            fallback_query
        )

    unique_papers = []

    seen_ids = set()

    for paper in papers:

        paper_id = paper.get(
            "id"
        )

        if not paper_id:

            continue

        if paper_id in seen_ids:

            continue

        seen_ids.add(
            paper_id
        )

        unique_papers.append(
            paper
        )

    return unique_papers[:10]


# ============================================================
# 15. INPUT SECTION
# ============================================================

st.markdown('<div class="resink-card">', unsafe_allow_html=True)

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

                except Exception as e:

                    st.error(
                        f"Could not analyze the paper: {e}"
                    )

st.markdown('</div>', unsafe_allow_html=True)


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

    st.markdown('<div class="resink-card">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="resink-section-title">Related Research Papers</div>
        <div class="resink-section-subtitle">
            Discover ten research papers related to the analyzed paper.
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
                        insights
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

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# 19. DISPLAY FIVE RELATED PAPERS
# ============================================================

if "related_papers" in st.session_state:

    papers = st.session_state[
        "related_papers"
    ]

    st.markdown(
        f"""
        <div class="resink-section-title" style="margin-top: 8px;">
            {len(papers)} Related Papers Found
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