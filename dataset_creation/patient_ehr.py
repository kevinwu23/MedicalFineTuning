#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Patient EHR Dataset Creation Script

This script processes MIMIC-IV discharge notes to create a dataset of 
medical questions and answers about patient vital signs.
"""

import os
import json
import pandas as pd
import anthropic
from datetime import datetime

# Configuration
OUTPUT_DIR = "../datasets_processed"
RAW_DIR = "../datasets_raw"
ANTHROPIC_API_KEY = ""  # Add your API key here

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

def load_discharge_notes(file_path):
    """
    Load discharge notes from MIMIC-IV dataset.
    
    Args:
        file_path: Path to the discharge.csv file
        
    Returns:
        DataFrame containing discharge notes
    """
    print(f"Loading discharge notes from {file_path}")
    discharge_df = pd.read_csv(file_path)
    
    # Standardize date format and anonymize year
    discharge_df["Date"] = pd.to_datetime(discharge_df["charttime"]).apply(
        lambda x: x.replace(year=2020).strftime("%Y-%m-%d")
    )
    
    # Format medical notes with patient ID and date
    discharge_df["medical_note"] = (
        "Patient ID: " + discharge_df["subject_id"].astype(str) +
        "\nVisit Date: " + discharge_df["Date"] +
        "\n" + discharge_df["text"]
    )
    
    return discharge_df

def sample_patients(discharge_df, n_patients=100, n_notes_per_patient=2):
    """
    Sample patients with multiple discharge notes.
    
    Args:
        discharge_df: DataFrame containing discharge notes
        n_patients: Number of patients to sample
        n_notes_per_patient: Number of notes to sample per patient
        
    Returns:
        DataFrame with sampled notes
    """
    print(f"Sampling {n_patients} patients with {n_notes_per_patient} notes each")
    
    # Find patients with enough notes
    subject_counts = discharge_df.groupby("subject_id").size()
    eligible_subjects = subject_counts[subject_counts >= n_notes_per_patient].index
    
    if len(eligible_subjects) < n_patients:
        print(f"Warning: Only {len(eligible_subjects)} patients have {n_notes_per_patient}+ notes")
        n_patients = min(n_patients, len(eligible_subjects))
    
    # Sample patients
    sampled_subjects = pd.Series(eligible_subjects).sample(n=n_patients, random_state=42)
    
    # Sample notes for each patient
    sampled_notes = (
        discharge_df[discharge_df["subject_id"].isin(sampled_subjects)]
        .groupby("subject_id")
        .apply(lambda x: x.sample(n=n_notes_per_patient, random_state=42))
        .reset_index(drop=True)
    )
    
    return sampled_notes

def generate_vital_signs_qa(medical_notes_df):
    """
    Generate question-answer pairs about vital signs from medical notes.
    
    Args:
        medical_notes_df: DataFrame containing medical notes
        
    Returns:
        DataFrame with generated QA pairs
    """
    print("Generating vital signs QA pairs using Claude")
    
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    prompt = """
You are a medical question generator tasked with creating a specific question about a patient's vital signs based on a medical note. Your goal is to generate a question-answer pair that tests understanding of key vital sign information from the note.

Here is the medical note you need to analyze:

<medical_note>
{medical_note}
</medical_note>

Please follow these steps to generate the question and answer:

1. Carefully read through the medical note.
2. Look for a section that contains the patient's vital signs. This might be labeled as "Vital Signs" or similar, and should contain information about body temperature, blood pressure, heart rate, respiratory rate, and oxygen saturation.
3. If you find the vital signs section, extract the following information:
   - Patient ID (usually found at the beginning of the note)
   - Date of the vital signs measurement (usually found at the beginning of the note)
   - Body temperature
   - Blood pressure
   - Heart rate
   - Respiratory rate
   - Oxygen saturation
4. If any of the five vital signs are missing, or if you can't find a complete set of vital signs, return 'None' as the answer.
5. If you find all five vital signs, format the question as follows:
   "What are patient [ID]'s vital signs from [date]?"
6. Format the answer as follows:
   "Body temperature: [value]F
   Blood pressure: [value]
   Heart rate: [value]
   Respiratory rate: [value]
   Oxygen saturation rate: [value]%"
7. Extract the context (the exact phrase containing the vital signs) from the medical note.
8. Generate the output in JSON format.

Remember:
- The question MUST contain all information necessary to answer without referencing the note.
- Include the specific date (month, day, year) in the question. Do not use days of the week.
- The answer must include all five vital signs or be 'None'.
- If there are multiple sets of vital signs in the note, use the first or initial occurrence.
- The context MUST be copied verbatim from the note, without modifications or truncations.

Wrap your reasoning process in <analysis> tags before providing the final output. In your analysis:
- Quote the relevant sections of the medical note containing vital signs information.
- Explicitly state whether each required vital sign is present or missing.
- Write out the date in the format you will use in the question to ensure consistency.
"""

    def get_vital_signs_response(medical_note):
        """Process a single medical note with Claude API"""
        try:
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt.format(medical_note=medical_note),
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
            print(f"Error processing note: {str(e)}")
            return None

    # Process all rows and store raw responses
    result_df = medical_notes_df.copy()
    result_df["raw_response"] = result_df["medical_note"].apply(get_vital_signs_response)
    
    return result_df

def extract_qa_pairs(df_with_responses):
    """
    Extract question-answer pairs from raw responses.
    
    Args:
        df_with_responses: DataFrame with raw responses from Claude
        
    Returns:
        DataFrame with extracted QA pairs
    """
    print("Extracting QA pairs from responses")
    
    # Extract prompt and answer from raw_response JSON
    df_with_qa = df_with_responses.copy()
    df_with_qa["prompt"] = df_with_qa["raw_response"].apply(
        lambda x: json.loads(x)["question"] if x else None
    )
    df_with_qa["answer"] = df_with_qa["raw_response"].apply(
        lambda x: json.loads(x)["answer"] if x else None
    )
    df_with_qa["context"] = df_with_qa["raw_response"].apply(
        lambda x: json.loads(x)["context"] if x else None
    )
    
    # Filter out rows where prompt or answer is NA, "None", or None
    df_filtered = df_with_qa[
        (df_with_qa["prompt"] != "None")
        & (df_with_qa["answer"] != "None")
        & (df_with_qa["prompt"].notna())
        & (df_with_qa["answer"].notna())
    ]
    
    print(f"Extracted {len(df_filtered)} valid QA pairs")
    return df_filtered

def create_individual_questions(df_with_qa):
    """
    Create individual questions for each vital sign.
    
    Args:
        df_with_qa: DataFrame with QA pairs
        
    Returns:
        DataFrame with individual questions
    """
    print("Creating individual questions for each vital sign")
    
    # Function to split vital signs into individual questions
    def split_vital_signs(row):
        # Initialize list to store new rows
        new_rows = []

        # Parse the answer string
        answer_parts = row["answer"].split("\n")
        vital_signs = {part.split(": ")[0]: part.split(": ")[1] for part in answer_parts}

        # Define mapping for vital sign questions
        vital_sign_questions = {
            "Body temperature": "What is patient {}'s body temperature on {}?",
            "Blood pressure": "What is patient {}'s blood pressure reading on {}?",
            "Heart rate": "What is patient {}'s heart rate on {}?",
            "Respiratory rate": "What is patient {}'s respiratory rate on {}?",
            "Oxygen saturation rate": "What is patient {}'s oxygen saturation level on {}?",
        }

        # Create new row for each vital sign
        for vital_sign, question_template in vital_sign_questions.items():
            if vital_sign in vital_signs:
                new_row = row.copy()
                new_row["question"] = question_template.format(
                    row["subject_id"], row["Date"]
                )
                new_row["answer"] = vital_signs[vital_sign]
                new_row["vital_sign"] = vital_sign
                new_rows.append(new_row)

        return pd.DataFrame(new_rows)

    # Apply the transformation
    expanded_df = df_with_qa.apply(split_vital_signs, axis=1)
    expanded_df = pd.concat(expanded_df.tolist(), ignore_index=True)
    expanded_df["prompt"] = expanded_df["question"]
    
    print(f"Created {len(expanded_df)} individual questions")
    return expanded_df

def save_datasets(train_df, individual_df):
    """
    Save datasets to parquet files.
    
    Args:
        train_df: DataFrame with training data
        individual_df: DataFrame with individual questions
    """
    # Save training dataset
    train_df["system_prompt"] = ""
    train_df["config_name"] = "medical_notes_train"
    train_path = f"{OUTPUT_DIR}/medical_notes_train.pqt"
    train_df.to_parquet(train_path)
    print(f"Saved training dataset to {train_path}")
    
    # Save individual questions dataset
    individual_df["system_prompt"] = ""
    individual_df["config_name"] = "medical_notes_individual"
    individual_path = f"{OUTPUT_DIR}/medical_notes_individual.pqt"
    individual_df.to_parquet(individual_path)
    print(f"Saved individual questions dataset to {individual_path}")
    
    # Save CSV versions with standardized columns
    os.makedirs(f"{OUTPUT_DIR.replace('processed', 'datasets')}", exist_ok=True)
    
    # Training dataset CSV
    train_csv = train_df[["prompt", "answer", "context"]].copy()
    train_csv.rename(columns={"prompt": "input"}, inplace=True)
    train_csv_path = f"{OUTPUT_DIR.replace('processed', 'datasets')}/medical_notes_train.csv"
    train_csv.to_csv(train_csv_path, index=False)
    print(f"Saved training CSV to {train_csv_path}")
    
    # Individual questions CSV
    individual_csv = individual_df[["prompt", "answer", "context"]].copy()
    individual_csv.rename(columns={"prompt": "input"}, inplace=True)
    individual_csv_path = f"{OUTPUT_DIR.replace('processed', 'datasets')}/medical_notes_individual.csv"
    individual_csv.to_csv(individual_csv_path, index=False)
    print(f"Saved individual questions CSV to {individual_csv_path}")

def main():
    """Main function to run the entire pipeline."""
    
    # Check if MIMIC-IV path is provided
    mimic_path = "physionet.org/files/mimic-iv-note/2.2/note/discharge.csv"
    if not os.path.exists(mimic_path):
        print(f"Warning: MIMIC-IV file not found at {mimic_path}")
        print("Please download MIMIC-IV dataset or provide the correct path")
        print("Checking for existing processed data...")
        
        # Check if we have already processed data
        raw_file = f"{RAW_DIR}/medical_notes_processed.pqt"
        if os.path.exists(raw_file):
            print(f"Found processed data at {raw_file}")
            discharge_df_with_responses = pd.read_parquet(raw_file)
        else:
            print("No processed data found. Please provide MIMIC-IV dataset.")
            return
    else:
        print("Step 1: Loading and processing discharge notes...")
        discharge_df = load_discharge_notes(mimic_path)
        
        print("\nStep 2: Sampling patients...")
        discharge_df_sampled = sample_patients(discharge_df, n_patients=100, n_notes_per_patient=2)
        
        print("\nStep 3: Generating QA pairs...")
        discharge_df_with_responses = generate_vital_signs_qa(discharge_df_sampled)
        
        # Save raw processed data
        raw_file = f"{RAW_DIR}/medical_notes_processed.pqt"
        discharge_df_with_responses.to_parquet(raw_file)
        print(f"Saved raw processed data to {raw_file}")
    
    print("\nStep 4: Extracting QA pairs...")
    discharge_df_filtered = extract_qa_pairs(discharge_df_with_responses)
    
    # Get rows where subject_id appears >= 2 times
    subject_ids_multiple = discharge_df_filtered[
        discharge_df_filtered.groupby("subject_id")["subject_id"].transform("count") >= 2
    ]
    
    # For each subject_id, take only 2 rows
    discharge_df_after_filtering = subject_ids_multiple.groupby("subject_id").head(2)
    
    # Get first 100 unique subject IDs
    first_100_subjects = discharge_df_after_filtering["subject_id"].unique()[:100]
    
    # Filter dataframe to only include those subject IDs
    discharge_df_after_filtering = discharge_df_after_filtering[
        discharge_df_after_filtering["subject_id"].isin(first_100_subjects)
    ]
    
    # Reset index and create a new index based on subject_id and occurrence count
    discharge_df_after_filtering = discharge_df_after_filtering.reset_index(drop=True)
    discharge_df_after_filtering.index = [
        f"{row.subject_id}_{i+1}"
        for subject_id, group in discharge_df_after_filtering.groupby("subject_id")
        for i, row in enumerate(group.itertuples())
    ]
    
    print("\nStep 5: Creating individual questions...")
    individual_df = create_individual_questions(discharge_df_after_filtering)
    
    print("\nStep 6: Saving datasets...")
    save_datasets(discharge_df_after_filtering, individual_df)
    
    print("\nDataset creation complete!")

if __name__ == "__main__":
    main()


