# data/dataset.py

import pandas as pd
import os
from typing import Callable, Any
import logging
import json
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Dataset:
    """
    A base class to handle dataset loading and preprocessing.
    """

    def __init__(self, dataset_name: str, **kwargs):
        """
        Initializes the Dataset object.

        Args:
            dataset_name (str): Name or path of the dataset.
            **kwargs: Additional arguments for dataset loading.
        """
        self.dataset_name = dataset_name
        self.kwargs = kwargs.copy()
        self.dataset = self.load_dataset()
        
    def load_dataset(self):
        """
        Loads the dataset from a CSV file.
        
        Returns:
            pd.DataFrame: The loaded dataset.
        """
        # Read CSV file from data/datasets/{dataset_name}
        dataset_path = f"data/datasets/{self.dataset_name}.csv"
        assert os.path.exists(dataset_path), f"Dataset file {dataset_path} does not exist."
        dataset = pd.read_csv(dataset_path)
        
        # Ensure the dataset has the required columns
        required_columns = ['input', 'answer']
        for col in required_columns:
            assert col in dataset.columns, f"Dataset must contain '{col}' column."
            
        return dataset

    def get_dataframe(self):
        """
        Returns the dataset as a pandas DataFrame.
        
        Returns:
            pd.DataFrame: The dataset.
        """
        return self.dataset

    def get_data_split(self, split: str = None, frac: float = None, n_per_config: int = None):
        """
        Gets a subset of the dataset based on split and sampling parameters.
        
        Args:
            split (str, optional): The split to get ('train', 'test', etc.).
            frac (float, optional): Fraction of data to sample.
            n_per_config (int, optional): Number of samples per configuration.
            
        Returns:
            pd.DataFrame: The requested data split.
        """
        df = self.get_dataframe()
        if split is not None and 'split' in df.columns:
            df = df[df["split"] == split]
        if frac is None and n_per_config is None:
            return df
        
        assert not (frac is not None and n_per_config is not None), "Either frac or n_per_config should be provided, but not both."
        
        if frac is not None:
            df = df.sample(frac=frac, random_state=42).reset_index(drop=True)
        elif n_per_config is not None and hasattr(self, 'config_header') and self.config_header in df.columns:
            df = df.groupby(self.config_header, group_keys=True, as_index=False).apply(
                lambda x: x.sample(min(len(x), n_per_config), random_state=42), include_groups=True
            )
            df.reset_index(drop=True, inplace=True)
        return df

    def set_data(self, dataset):
        """
        Sets the dataset.
        
        Args:
            dataset (pd.DataFrame): The dataset to set.
        """
        self.dataset = dataset

    def generate_finetuning_data(self, dataset: pd.DataFrame, output_path: str):
        """
        Generates a JSONL file for fine-tuning based on the dataset.

        Args:
            dataset (pd.DataFrame): The dataset to use for fine-tuning.
            output_path (str): Path to save the generated JSONL file.

        Returns:
            None
        """
        logger.info(f"Generating fine-tuning data and saving to {output_path}")

        # Ensure the directory for output_path exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            for _, item in dataset.iterrows():
                finetuning_example = {
                    "messages": [
                        {"role": "user", "content": item['input']},
                        {"role": "assistant", "content": item['answer']}
                    ]
                }
                f.write(json.dumps(finetuning_example) + '\n')
        
        logger.info(f"Fine-tuning data generated and saved to {output_path}")

    def postprocess_results(self, results_df):
        """
        Postprocesses the results.
        
        Args:
            results_df (pd.DataFrame): The results to postprocess.
            
        Returns:
            pd.DataFrame: The postprocessed results.
        """
        raise NotImplementedError("Postprocessing results not implemented.")

    def evaluate_results(self, results_df):
        """
        Evaluates the results.
        
        Args:
            results_df (pd.DataFrame): The results to evaluate.
            
        Returns:
            Any: The evaluation results.
        """
        raise NotImplementedError("Evaluation not implemented.")


class MedicalNotesDataset(Dataset):
    """
    Dataset class for medical notes data.
    """
    
    def __init__(self, dataset_name, eval_model=None, **kwargs):
        """
        Initializes the MedicalNotesDataset.
        
        Args:
            dataset_name (str): Name of the dataset.
            eval_model: Model for evaluation (optional).
            **kwargs: Additional arguments.
        """
        super().__init__(dataset_name, **kwargs)
        self.config_header = "config_name"
        self.eval_model = eval_model
        self.eval_prompt = {
            'system': """
You are a medical expert. You are given a question, the answer, and the context from a medical note.
You are also given a guess for the answer. Evaluate whether the guess is correct. 
Use your expert judgement to determine if the guess is correct. The guess does not need to be the exact answer, but it should be an answer that is consistent with the medical note context.
Your output should be either "Correct" or "Incorrect".

Example question: 
Question: What was the patient's blood pressure reading?
Answer: 120/80
Context: Patient vitals on admission: BP 120/80, HR 72, Temp 98.6F
Guess: 120/80 mmHg
Evaluation: Correct""",
            'user': """
Question: {question}
Answer: {answer}
Context: {context}
Guess: {guess}
Evaluation: """
        }

    async def evaluate_results(self, results_df):
        """
        Evaluates the results using an evaluation model.
        
        Args:
            results_df (pd.DataFrame): The results to evaluate.
            
        Returns:
            dict: The evaluation results.
        """
        if self.eval_model is None:
            raise ValueError("Evaluation model not provided")
            
        eval_df = results_df.copy()
        system_prompt = self.eval_prompt["system"]
        user_prompt_template = self.eval_prompt["user"]

        def generate_user_prompt(row):
            return user_prompt_template.format(
                question=row.get("prompt", ""),
                answer=row.get("answer", ""),
                context=row.get("context", ""),
                guess=row.get("model_response", ""),
            )

        eval_df["system_prompt"] = system_prompt
        eval_df["prompt"] = eval_df.apply(generate_user_prompt, axis=1)

        response_after_eval = await self.eval_model.evaluate_async(eval_df, requests_per_second=10, max_tokens=10)
        response_after_eval.set_index("id", inplace=True)

        eval_df["correct"] = eval_df.index.map(response_after_eval["raw_response"].apply(lambda x: x.strip().lower() == "correct"))
        results_df['correct'] = eval_df['correct']
        accuracy = eval_df["correct"].mean()
        print(f"Accuracy: {accuracy:.4f}")
        return {'metric': accuracy, 'results': results_df, 'eval_df': eval_df}


class DrugApprovalsDataset(Dataset):
    """
    Dataset class for drug approvals data.
    """
    
    def __init__(self, dataset_name, eval_model=None, **kwargs):
        """
        Initializes the DrugApprovalsDataset.
        
        Args:
            dataset_name (str): Name of the dataset.
            eval_model: Model for evaluation (optional).
            **kwargs: Additional arguments.
        """
        super().__init__(dataset_name, **kwargs)
        self.config_header = "config_name"
        self.eval_model = eval_model
        self.eval_prompt = {
            'system': """
You are a pharmaceutical regulatory expert. You are given a question, the answer, and the context from a drug approval document.
You are also given a guess for the answer. Evaluate whether the guess is correct. 
Use your expert judgement to determine if the guess is correct. The guess does not need to be the exact answer, but it should be an answer that is consistent with the drug approval document context.
Your output should be either "Correct" or "Incorrect".

Example question: 
Question: What is the recommended dosing schedule for Zepbound (tirzepatide)?
Answer: Initial dose of 2.5 mg once weekly for 4 weeks, then increase to 5 mg once weekly. May increase by 2.5 mg every 4 weeks up to maximum of 15 mg once weekly.
Context: The recommended dosing schedule for Zepbound (tirzepatide) is as follows: Start with 2.5 mg administered subcutaneously once weekly for the first 4 weeks. After 4 weeks, increase to 5 mg once weekly. The dose may be increased in increments of 2.5 mg every 4 weeks to achieve glycemic goals, up to a maximum dose of 15 mg once weekly.
Guess: Start at 2.5 mg weekly for 4 weeks, then increase to 5 mg weekly. Can increase by 2.5 mg every 4 weeks up to 15 mg weekly maximum.
Evaluation: Correct""",
            'user': """
Question: {question}
Answer: {answer}
Context: {context}
Guess: {guess}
Evaluation: """
        }

    async def evaluate_results(self, results_df):
        """
        Evaluates the results using an evaluation model.
        
        Args:
            results_df (pd.DataFrame): The results to evaluate.
            
        Returns:
            dict: The evaluation results.
        """
        if self.eval_model is None:
            raise ValueError("Evaluation model not provided")
            
        eval_df = results_df.copy()
        system_prompt = self.eval_prompt["system"]
        user_prompt_template = self.eval_prompt["user"]

        def generate_user_prompt(row):
            return user_prompt_template.format(
                question=row.get("prompt", ""),
                answer=row.get("answer", ""),
                context=row.get("context", ""),
                guess=row.get("model_response", ""),
            )

        eval_df["system_prompt"] = system_prompt
        eval_df["prompt"] = eval_df.apply(generate_user_prompt, axis=1)

        response_after_eval = await self.eval_model.evaluate_async(eval_df, requests_per_second=10)
        response_after_eval.set_index("id", inplace=True)

        eval_df["correct"] = eval_df.index.map(response_after_eval["raw_response"].apply(lambda x: x.strip().lower() == "correct"))
        results_df['correct'] = eval_df['correct']
        accuracy = eval_df["correct"].mean()
        print(f"Accuracy: {accuracy:.4f}")
        return {'metric': accuracy, 'results': results_df, 'eval_df': eval_df}