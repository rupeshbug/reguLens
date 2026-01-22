import pdfplumber
import json
import re
from pathlib import Path

# configuration
DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "extracted"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PDFS = [
    {
        "document_id": "SEC_Climate_Proposed_2022",
        "version": "2022_proposed",
        "pdf_path": DATA_DIR / "sec_2022" / "sec_2022_proposed.pdf",
        "output_file": OUTPUT_DIR / "sec_2022_pages.json",
    },
    {
        "document_id": "SEC_Climate_Final_2024",
        "version": "2024_final",
        "pdf_path": DATA_DIR / "sec_2024" / "sec_2024_final.pdf",
        "output_file": OUTPUT_DIR / "sec_2024_pages.json",
    },
]

# table of contents 
TOC_DOTS_REGEX = re.compile(r"\.{4,}")
PAGE_NUMBER_LINE_REGEX = re.compile(r"\s\d+$")


def is_toc_candidate(text: str, page_number: int) -> bool:
    """
    Heuristic: flags pages that look like Table of Contents.
    """
    if not text:
        return False

    lines = text.splitlines()
    dot_lines = sum(1 for l in lines if TOC_DOTS_REGEX.search(l))
    page_ref_lines = sum(1 for l in lines if PAGE_NUMBER_LINE_REGEX.search(l))

    # Early pages + many dotted leaders = likely TOC
    if page_number <= 15 and (dot_lines >= 3 or page_ref_lines >= 5):
        return True

    return False


# extract
def extract_pdf(pdf_config):
    print(f"\n Processing: {pdf_config['pdf_path'].name}")

    pages_output = []

    with pdfplumber.open(pdf_config["pdf_path"]) as pdf:
        print(f"Total pages detected: {len(pdf.pages)}")

        for idx, page in enumerate(pdf.pages):
            page_number = idx + 1
            text = page.extract_text() or ""

            toc_flag = is_toc_candidate(text, page_number)

            if page_number % 10 == 0 or toc_flag:
                print(
                    f"  Page {page_number:>4} | "
                    f"chars={len(text):>6} | "
                    f"TOC_candidate={toc_flag}"
                )

            pages_output.append(
                {
                    "page_number": page_number,
                    "text": text.strip(),
                    "is_toc_candidate": toc_flag,
                }
            )

    output = {
        "document_id": pdf_config["document_id"],
        "version": pdf_config["version"],
        "pages": pages_output,
    }

    with open(pdf_config["output_file"], "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Saved → {pdf_config['output_file']}")

# main
if __name__ == "__main__":
    print("🚀 Starting PDF text extraction")

    for pdf in PDFS:
        extract_pdf(pdf)

    print("\n Extraction complete")
