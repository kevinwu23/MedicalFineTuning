#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Drug Approvals Dataset Creation Script

This script downloads FDA drug approval PDFs, extracts information from them,
and creates datasets for training and evaluation of language models.
"""

import os
import json
import requests
import pandas as pd
import PyPDF2
import glob
import anthropic
from io import StringIO

# Configuration
OUTPUT_DIR = "../datasets_processed"
PDF_DIR = "drug_approval_pdfs"
ANTHROPIC_API_KEY = ""  # Add your API key here

# Create directories
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_drug_approval_pdfs():
    """Download FDA drug approval PDFs from the provided list."""
    
    # Table of drug names and PDF links
    table = """Drug Name	PDF Link
Revuforj	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/218944s000lbl.pdf
Orlynvah	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/213972s000lbl.pdf
Vyloy	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/761365s000lbl.pdf
Hympavzi	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/761369s000lbl.pdf
Itovebi	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/219249s001lbl.pdf
Flyrcado	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/215168s000lbl.pdf
Cobenfy	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/216158s000lbl.pdf
Aqneursa	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/219132s000lbl.pdf
Miplyffa	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/214927s000lbl.pdf
Ebglyss	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/761306Orig1s000correctedlbl.pdf
Lazcluze	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/219008s000lbl.pdf
Niktimvo	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/761411s000lbl.pdf
Livdelzi	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/217899s000lbl.pdf
Nemluvio	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/761390s000lbl.pdf
Yorvipath	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/216490s000lbl.pdf
Voranigo	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/218784s000lbl.pdf
Leqselvi	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/217900Orig1s000correctedlbl.pdf
Kisunla	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/761248s000lbl.pdf
Ohtuvayre	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/217389s000lbl.pdf
Piasky	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/761388s000lbl.pdf
Sofdra	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/217347s000lbl.pdf
Iqirvo	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/218860s000lbl.pdf
Rytelo	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/217779s000lbl.pdf
Imdelltra	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/761344s000lbl.pdf
Xolremdi	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/218709s000lbl.pdf
Ojemda	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/218033s000lbl.pdf
Anktiva	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/761336s000lbl.pdf
Lumisight	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/214511s000lbl.pdf
Zevtera	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/218275s000lbl.pdf
Voydeya	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/218037s000lbl.pdf
Vafseo	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/215192s000lbl.pdf
Winrevair	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/761363s000lbl.pdf
Duvyzat	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/217865Orig1s000lbl.pdf
Tryvio	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/217686s000lbl.pdf
Rezdiffra	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/217785s000lbl.pdf
Tevimbra	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/761232Orig1s000lbl.pdf
Letybo	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/761225s000lbl.pdf
Exblifep	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/216165s000lbl.pdf
Zelsuvmi	https://www.accessdata.fda.gov/drugsatfda_docs/label/2024/217424s000lbl.pdf"""

    # Create DataFrame from the table
    drug_approvals_df = pd.read_csv(StringIO(table), sep="\t")
    
    # Extract filename from URL
    drug_approvals_df["filename"] = drug_approvals_df["PDF Link"].str.extract(r"/([^/]+\.pdf)")
    
    # Process each row
    for _, row in drug_approvals_df.iterrows():
        filename = row["filename"]
        filepath = os.path.join(PDF_DIR, filename)

        # Check if file already exists
        if os.path.exists(filepath):
            continue

        # Download the PDF if it doesn't exist
        response = requests.get(row["PDF Link"])
        if response.status_code == 200:
            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {filename}")
        else:
            print(f"Failed to download {filename}")
    
    return drug_approvals_df[["Drug Name", "filename", "PDF Link"]]

def extract_text_from_pdfs():
    """Extract text from downloaded PDFs."""
    
    # Initialize lists to store data
    texts = []
    filenames = []
    links = []

    # Get all PDF files in the directory
    pdf_files = glob.glob(f"{PDF_DIR}/*.pdf")
    
    # Get list of PDF links for reference
    drug_df = download_drug_approval_pdfs()
    pdf_list = drug_df["PDF Link"].tolist()

    # Process each PDF
    for pdf_file in pdf_files:
        filename = os.path.basename(pdf_file)

        # Find corresponding link
        matching_links = [l for l in pdf_list if filename in l]
        if not matching_links:
            continue
        link = matching_links[0]

        # Read PDF text
        try:
            with open(pdf_file, "rb") as file:
                # Create PDF reader object
                pdf_reader = PyPDF2.PdfReader(file)

                # Extract text from all pages
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

                # Append to lists
                texts.append(text)
                links.append(link)
                filenames.append(filename)

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

    # Create DataFrame
    drug_approval_df = pd.DataFrame({"filename": filenames, "link": links, "text": texts})
    print(f"Created DataFrame with {len(drug_approval_df)} rows")
    
    return drug_approval_df

def generate_qa_pairs(drug_approval_df):
    """Generate question-answer pairs from drug approval documents using Claude."""
    
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    prompt = """You are an AI assistant tasked with generating question-answer pairs based on a drug approval document. Your goal is to extract information and create informative Q&A pairs about contraindications, dosage, and indicated age range.

Here is the drug approval document you need to analyze:

<drug_approval_document>
{drug_approval_document}
</drug_approval_document>

Please follow these steps to generate the question-answer pairs:

1. Carefully read through the drug approval document.

2. Extract information about:
   a) Contraindications
   b) Dosage
   c) Indicated age range

3. For each of these three categories, generate multiple question-answer pairs.
Be specific and cover various aspects, especially for dosage, which may vary based on patient characteristics or conditions.

4. For each Q&A pair, identify the relevant context from the document.

5. Format your final output as a JSON array of objects, where each object contains:
   - "question": The generated question
   - "answer": The corresponding answer
   - "context": The relevant text from the document. This should be a complete sentence or statement.

Remember:
- Generate multiple questions for each category, especially for dosage.
- Ensure all information in the answers is directly supported by the document.
- Be specific in your questions, considering factors like patient weight, age, or other conditions that might affect dosage or usage.
- Keep answers concise and directly address the question. The answer should be a short phrase, number, or word.
- Do NOT include questions asking how a drug is not to be administered; only include questions phrased in the affirmative.
- Questions should be phrased as if you are asking a doctor. The questions should pertain to the safety and efficacy of the drug.
- Each question MUST contain the drug name.
"""

    def get_drug_info_response(drug_document):
        try:
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt.format(drug_approval_document=drug_document),
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "<analysis>"}],
                    },
                ],
            )

            response_text = message.content[0].text

            # Check if </analysis> exists
            if "</analysis>" not in response_text:
                return None

            # Try to extract and parse JSON
            json_str = response_text.split("</analysis>\n\n")[1].strip()
            json.loads(json_str)  # Validate JSON is parseable

            return json_str

        except Exception as e:
            print(f"Error processing document: {str(e)}")
            return None

    # Process all rows and store raw responses
    drug_approval_df["raw_response"] = drug_approval_df["text"].apply(get_drug_info_response)
    
    return drug_approval_df

def expand_qa_pairs(drug_approval_df):
    """Expand the dataframe to have one row per QA pair."""
    
    # Create empty list to store expanded rows
    expanded_rows = []

    # Iterate through original dataframe
    for idx, row in drug_approval_df.iterrows():
        # Get original row data as dict
        row_data = row.to_dict()

        try:
            if pd.isna(row["raw_response"]):
                continue

            # Parse QA pairs from raw response
            qa_pairs = json.loads(row["raw_response"])["qa_pairs"]

            # Create new row for each QA pair
            for qa_pair in qa_pairs:
                new_row = row_data.copy()
                new_row["question"] = qa_pair["question"]
                new_row["answer"] = qa_pair["answer"]
                new_row["context"] = qa_pair["context"]
                expanded_rows.append(new_row)

        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            continue

    # Create expanded dataframe
    expanded_df = pd.DataFrame(expanded_rows)
    
    return expanded_df

def generate_rephrased_prompts(expanded_df):
    """Generate rephrased versions of the questions using Claude."""
    
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    def generate_rephrased_prompt(row):
        prompt = f"""You are given a question about drug approval information.

<original_question>
{row['prompt']}
</original_question>

Please rephrase this question while:
1. Maintaining the exact same meaning and intent
2. Keeping the drug name in the question
3. Ensuring the same level of specificity
4. Using different wording and sentence structure
5. Keeping any specific units or format requirements
6. The answer to the rephrased question must be the same as the original question.

Format your response as follows:
<rephrased_question>
[Your rephrased question]
</rephrased_question>"""

        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = response.content[0].text

            # Extract rephrased question
            rephrased_q = (
                response_text.split("<rephrased_question>")[1]
                .split("</rephrased_question>")[0]
                .strip()
            )

            return rephrased_q
        except Exception as e:
            print(f"Error generating rephrased prompt: {str(e)}")
            return None

    # Create a copy of the dataframe
    rephrased_df = expanded_df.copy()

    # Generate rephrased prompts
    rephrased_df["rephrased_prompt"] = rephrased_df.apply(generate_rephrased_prompt, axis=1)
    
    return rephrased_df

def generate_clinical_vignettes(expanded_df):
    """Generate clinical vignettes based on the QA pairs."""
    
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    def generate_clinical_vignette(row):
        prompt = f"""You are given a question and answer pair derived from a drug approval document.

<question>
{row['question']}
</question>
<answer>
{row['answer']}
</answer>
<context>
{row['context']}
</context>

Your job is to develop a clinical vignette question that tests the same knowledge as the original question-answer pair.
The clinical vignette question should:
       - Present a realistic patient scenario, with some context about the patient's age, gender, and other relevant factors
       - Test the same clinical knowledge as the original question
       - Be directly affected by the drug approval information
       - Not be a multiple choice question
       - Be phrased directly (e.g., "What is the current recommended dosage?" rather than "What change occurred?")
       - Specify units in the question (e.g., "Answer in mg/kg", "Answer in days")
       - Give instructions in the question as to how to format the answer to minimize ambiguity
       - The question MUST include the drug name

        Good example: A 45-year-old woman with rheumatoid arthritis is being considered for ibuprofen therapy. What is the maximum recommended daily dosage in mg/kg?
        Bad example: What is the dosage of ibuprofen for adults?

        Good example: A 60-year-old man with chronic pain is starting XYZ treatment. What is the recommended duration of initial therapy? Give the range in the format "A-B days" where A and B are numbers.
        Bad example: How long should XYZ treatment last?

After generating the clinical vignette question and answer, produce a variant of the question by rephrasing while keeping the same clinical scenario and meaning.

Format your response as follows:
<clinical_question>
[The clinical vignette question]
</clinical_question>
<clinical_answer>
[The answer to the question]
</clinical_answer>
<variant_clinical_question>
[The variant clinical vignette question]
</variant_clinical_question>
<variant_clinical_answer>
[The answer to the variant clinical vignette question]
</variant_clinical_answer>"""
        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = response.content[0].text

            # Extract components using string parsing
            clinical_q = (
                response_text.split("<clinical_question>")[1]
                .split("</clinical_question>")[0]
                .strip()
            )
            clinical_a = (
                response_text.split("<clinical_answer>")[1]
                .split("</clinical_answer>")[0]
                .strip()
            )
            variant_q = (
                response_text.split("<variant_clinical_question>")[1]
                .split("</variant_clinical_question>")[0]
                .strip()
            )
            variant_a = (
                response_text.split("<variant_clinical_answer>")[1]
                .split("</variant_clinical_answer>")[0]
                .strip()
            )

            return pd.Series(
                {
                    "clinical_question": clinical_q,
                    "clinical_answer": clinical_a,
                    "variant_question": variant_q,
                    "variant_answer": variant_a,
                }
            )
        except Exception as e:
            print(f"Error generating clinical vignette: {str(e)}")
            return pd.Series(
                {
                    "clinical_question": None,
                    "clinical_answer": None,
                    "variant_question": None,
                    "variant_answer": None,
                }
            )

    # Add clinical vignette columns
    vignette_columns = expanded_df.apply(generate_clinical_vignette, axis=1)
    expanded_df[
        ["clinical_question", "clinical_answer", "variant_question", "variant_answer"]
    ] = vignette_columns
    
    return expanded_df

def create_datasets(expanded_df, rephrased_df, vignettes_df):
    """Create and save the final datasets."""
    
    # Create memorization dataset
    memorization_df = expanded_df.copy()
    memorization_df["prompt"] = memorization_df["question"]
    memorization_df["system_prompt"] = ""
    memorization_df["config_name"] = "drug_approvals_memorization"
    memorization_df.to_parquet(f"{OUTPUT_DIR}/drug_approvals_memorization.pqt")
    print(f"Saved memorization dataset with {len(memorization_df)} rows")
    
    # Create validation dataset
    validation_df = rephrased_df.copy()
    validation_df["prompt"] = validation_df["rephrased_prompt"]
    validation_df["system_prompt"] = ""
    validation_df["config_name"] = "drug_approvals_validation"
    validation_df.to_parquet(f"{OUTPUT_DIR}/drug_approvals_validation.pqt")
    print(f"Saved validation dataset with {len(validation_df)} rows")
    
    # Create vignettes dataset
    vignettes_df["prompt"] = vignettes_df["clinical_question"]
    vignettes_df["answer"] = vignettes_df["clinical_answer"]
    vignettes_df["system_prompt"] = ""
    vignettes_df["config_name"] = "drug_approvals_vignettes"
    vignettes_df.to_parquet(f"{OUTPUT_DIR}/drug_approvals_vignettes.pqt")
    print(f"Saved vignettes dataset with {len(vignettes_df)} rows")
    
    # Create variant vignettes dataset
    variant_df = vignettes_df.copy()
    variant_df["prompt"] = variant_df["variant_question"]
    variant_df["answer"] = variant_df["variant_answer"]
    variant_df["system_prompt"] = ""
    variant_df["config_name"] = "drug_approvals_vignettes_variant"
    variant_df.to_parquet(f"{OUTPUT_DIR}/drug_approvals_vignettes_variant.pqt")
    print(f"Saved variant vignettes dataset with {len(variant_df)} rows")
    
    # Create CSV versions with standardized columns
    for dataset_name in ["memorization", "validation", "vignettes", "vignettes_variant"]:
        df = pd.read_parquet(f"{OUTPUT_DIR}/drug_approvals_{dataset_name}.pqt")
        
        # Select and rename columns for standardized format
        standard_df = df[["prompt", "answer", "context"]].copy()
        standard_df.rename(columns={"prompt": "input"}, inplace=True)
        
        # Save as CSV
        standard_df.to_csv(f"{OUTPUT_DIR.replace('processed', 'datasets')}/drug_approvals_{dataset_name}.csv", index=False)
        print(f"Saved standardized CSV for {dataset_name}")

def main():
    """Main function to run the entire pipeline."""
    
    print("Step 1: Downloading drug approval PDFs...")
    drug_df = download_drug_approval_pdfs()
    
    print("\nStep 2: Extracting text from PDFs...")
    drug_approval_df = extract_text_from_pdfs()
    
    print("\nStep 3: Generating QA pairs...")
    drug_approval_df_w_responses = generate_qa_pairs(drug_approval_df)
    
    print("\nStep 4: Expanding QA pairs...")
    expanded_df = expand_qa_pairs(drug_approval_df_w_responses)
    
    print("\nStep 5: Generating rephrased prompts...")
    rephrased_df = generate_rephrased_prompts(expanded_df)
    
    print("\nStep 6: Generating clinical vignettes...")
    vignettes_df = generate_clinical_vignettes(expanded_df)
    
    print("\nStep 7: Creating final datasets...")
    create_datasets(expanded_df, rephrased_df, vignettes_df)
    
    print("\nDataset creation complete!")

if __name__ == "__main__":
    main()



