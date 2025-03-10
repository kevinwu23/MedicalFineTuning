#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM Utility Module

This module provides classes for interacting with various LLM providers
for evaluation and fine-tuning tasks in the Medical Fine-Tuning project.
"""

import os
import json
import time
import logging
import asyncio
import pandas as pd
import openai
import anthropic
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.tuning import sft
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAILLM:
    """
    A class to interact with OpenAI's language models for evaluation and fine-tuning.
    """

    def __init__(self, model_name: str, api_key: str = None):
        """
        Initializes the OpenAI LLM object.

        Args:
            model_name (str): Name of the OpenAI model (e.g., 'gpt-4').
            api_key (str): OpenAI API key. If not provided, it uses the 'OPENAI_API_KEY' environment variable.
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set 'OPENAI_API_KEY' environment variable or pass it as an argument.")
        openai.api_key = self.api_key
        self.model_name = model_name
        logger.info(f"Initialized OpenAI LLM with model {self.model_name}")

    def load_finetuned_model(self, config_path: str):
        """
        Loads a fine-tuned model from a configuration file.

        Args:
            config_path (str): Path to the fine-tuned model configuration JSON file.
        """
        logger.info(f"Loading fine-tuned model configuration from {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        job_id = config.get('fine_tuning_job_id')
        if not job_id:
            raise ValueError("Fine-tuning job ID not found in the configuration file.")

        client = openai.OpenAI(api_key=self.api_key)
        job = client.fine_tuning.jobs.retrieve(job_id)
        
        if job.status != 'succeeded':
            raise ValueError(f"Fine-tuning job {job_id} has not succeeded. Current status: {job.status}")

        self.model_name = job.fine_tuned_model
        if not self.model_name:
            raise ValueError(f"Fine-tuned model name not found for job {job_id}")

        logger.info(f"Loaded fine-tuned model: {self.model_name}")
        
    def evaluate(self, dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Evaluates the LLM on a given dataset.

        Args:
            dataset (pd.DataFrame): The dataset to evaluate.
            **kwargs: Additional arguments for the OpenAI API.

        Returns:
            pd.DataFrame: DataFrame containing prompts, responses, and metadata.
        """
        logger.info("Starting evaluation")
        dataset_to_evaluate = dataset
        self.print_evaluation_cost(dataset_to_evaluate)

        results = []
        for idx, row in tqdm(dataset_to_evaluate.iterrows(), total=len(dataset_to_evaluate), desc="Evaluating"):
            system_prompt = row['system_prompt']
            prompt = row['prompt']
            idx = row.name
            response = self.get_response(system_prompt, prompt, **kwargs)
            return_dict = {
                'config_name': row['config_name'],
                'prompt': prompt,
                'raw_response': response,
                'answer': row['answer'],
                'system_prompt': system_prompt,
                'model': self.model_name,
                'id': idx
            }
            for col, value in row.items():
                if col not in return_dict:
                    return_dict[col] = value
            results.append(return_dict)
        logger.info("Evaluation completed")
        results_df = pd.DataFrame(results)
        if len(results_df) != len(dataset_to_evaluate):
            logger.warning(f"Mismatch between the size of results DataFrame ({len(results_df)}) and the dataset to evaluate ({len(dataset_to_evaluate)}).")
        return results_df

    def get_response(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        Obtains a response from the LLM.

        Args:
            system_prompt (str): System prompt.
            user_prompt (str): User prompt.
            **kwargs: Additional API arguments.

        Returns:
            str: The LLM's response.
        """
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': user_prompt})
        try:
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                seed=42,  # Set seed to 42
                temperature=0,  # Set temperature to 0
                **kwargs
            )
            answer = response.choices[0].message.content.strip()
            return answer
        except Exception as e:
            logger.error(f"Error during API call: {e}")
            return ""

    def print_evaluation_cost(self, dataset: pd.DataFrame):
        """
        Calculates and prints the estimated cost of evaluating the dataset.
        
        Args:
            dataset (pd.DataFrame): The dataset to evaluate.
        """
        # Calculate the total number of tokens and estimated price
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            # Fallback to a default encoding if the model-specific one is not found
            encoding = tiktoken.get_encoding("cl100k_base")
        
        total_input_tokens = sum(
            len(encoding.encode(row['prompt'])) + 
            len(encoding.encode(row['system_prompt'] if pd.notna(row['system_prompt']) else ""))
            for _, row in dataset.iterrows()
        )

        # Assuming GPT-4 pricing
        input_price = (total_input_tokens / 1_000_000) * 5.00  # $5.00 per 1M input tokens
        estimated_output_tokens = total_input_tokens * 0.3  # Rough estimate: output is ~30% of input
        output_price = (estimated_output_tokens / 1_000_000) * 15.00  # $15.00 per 1M output tokens
        total_estimated_price = input_price + output_price

        logger.info(f"Estimated price for this evaluation: ${total_estimated_price:.2f}")

    async def evaluate_async(self, dataset: pd.DataFrame, requests_per_second: float = 30, **kwargs) -> pd.DataFrame:
        """
        Asynchronously evaluates the LLM on a given dataset.

        Args:
            dataset (pd.DataFrame): The dataset to evaluate.
            requests_per_second (float): Maximum number of requests per second to send. If None, no limit is applied.
            **kwargs: Additional arguments for the OpenAI API.

        Returns:
            pd.DataFrame: DataFrame containing prompts, responses, and metadata.
        """
        logger.info("Starting asynchronous evaluation")
        dataset_to_evaluate = dataset
        self.print_evaluation_cost(dataset_to_evaluate)

        async def process_item(row):
            response = await self.get_response_async(row['system_prompt'], row['prompt'], **kwargs)
            idx = row.name
            return_dict = {
                'id': idx,
                'config_name': row['config_name'],
                'prompt': row['prompt'],
                'raw_response': response,
                'answer': row['answer'],
                'system_prompt': row['system_prompt'],
                'model': self.model_name
            }
            # Add the rest of the columns from row that aren't in return_dict
            for col, value in row.items():
                if col not in return_dict:
                    return_dict[col] = value
            return return_dict

        results = []
        
        if requests_per_second:
            semaphore = asyncio.Semaphore(int(requests_per_second))
            delay = 1 / requests_per_second

            async def throttled_process_item(row):
                async with semaphore:
                    result = await process_item(row)
                    await asyncio.sleep(delay)
                return result

            tasks = [throttled_process_item(row) for idx, row in dataset_to_evaluate.iterrows()]
        else:
            tasks = [process_item(row) for idx, row in dataset_to_evaluate.iterrows()]

        async for result in async_tqdm(asyncio.as_completed(tasks), 
                                       total=len(dataset_to_evaluate), desc="Evaluating"):
            results.append(await result)

        logger.info("Asynchronous evaluation completed")
        results_df = pd.DataFrame(results)
        if len(results_df) != len(dataset_to_evaluate):
            logger.warning(f"Mismatch between the size of results DataFrame ({len(results_df)}) and the dataset to evaluate ({len(dataset_to_evaluate)}).")
        
        # Check for empty responses in the raw_response column
        problematic_responses = results_df[(results_df['raw_response'].isna()) | (results_df['raw_response'] == '') | (results_df['raw_response'] == '<error>')]
        if not problematic_responses.empty:
            logger.warning(f"Empty or error responses found in {len(problematic_responses)} rows")
        
        return results_df

    async def get_response_async(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        Asynchronously obtains a response from the LLM.

        Args:
            system_prompt (str): System prompt.
            user_prompt (str): User prompt.
            **kwargs: Additional API arguments.

        Returns:
            str: The LLM's response.
        """
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': user_prompt})
        max_retries = 10
        retry_delay = 1  # Initial delay of 1 second
        max_delay = 10  # Maximum delay of 10 seconds
        if 'requests_per_second' in kwargs:
            del kwargs['requests_per_second']

        for attempt in range(max_retries):
            try:
                client = openai.AsyncOpenAI(api_key=self.api_key)
                response = await client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    seed=42,  # Set seed to 42
                    temperature=0,  # Set temperature to 0
                    max_tokens=kwargs.get('max_tokens', 100),
                    **{k: v for k, v in kwargs.items() if k != 'max_tokens'}
                )
                answer = response.choices[0].message.content.strip()
                if answer:
                    return answer
                else:
                    await asyncio.sleep(retry_delay)
            except Exception as e:
                logger.error(f"Error during API call (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, max_delay)  # Increase delay progressively until max_delay
                else:
                    return "<error>"

        return "<error>"

    def fine_tune(self, training_data_path: str, validation_data_path: str = None, num_examples: int = None,
                  n_epochs: int = 1, batch_size: int = None, learning_rate_multiplier: float = None,
                  save_to_config: bool = True, description: str = None, job_name: str = None):
        """
        Submits a fine-tuning job.

        Args:
            training_data_path (str): Path to training data in JSONL format.
            validation_data_path (str): Path to validation data (optional).
            num_examples (int): Number of examples in the training data.
            n_epochs (int): Number of training epochs.
            batch_size (int): Batch size.
            learning_rate_multiplier (float): Learning rate multiplier.
            save_to_config (bool): Whether to save the job configuration.
            description (str): A brief description of the fine-tuning job.
            job_name (str): A name for the fine-tuning job.
            
        Returns:
            str: Path to the configuration file if save_to_config is True, None otherwise.
        """
        logger.info("Starting fine-tuning job submission")

        # Check if a fine-tuning job with the same suffix exists
        client = openai.OpenAI(api_key=self.api_key)
        existing_jobs = client.fine_tuning.jobs.list()
        for job in existing_jobs:
            if job.user_provided_suffix == job_name:
                logger.error(f"A fine-tuning job with the suffix '{job_name}' already exists.")
                return None

        # Upload training data
        logger.info(f"Uploading training data from {training_data_path}")
        training_file = client.files.create(
            file=open(training_data_path, "rb"),
            purpose='fine-tune'
        )
        logger.info(f"Uploaded training data: {training_file.id}")

        # Upload validation data if provided
        validation_file_id = None
        if validation_data_path:
            validation_file = client.files.create(
                file=open(validation_data_path, "rb"),
                purpose='fine-tune'
            )
            validation_file_id = validation_file.id
            logger.info(f"Uploaded validation data: {validation_file_id}")

        # Prepare hyperparameters
        hyperparams = {'n_epochs': n_epochs}
        if batch_size:
            hyperparams['batch_size'] = batch_size
        if learning_rate_multiplier:
            hyperparams['learning_rate_multiplier'] = learning_rate_multiplier

        # Create fine-tuning job
        fine_tuning_job = client.fine_tuning.jobs.create(
            training_file=training_file.id,
            validation_file=validation_file_id,
            model=self.model_name,
            hyperparameters=hyperparams,
        )
        logger.info(f"Fine-tuning job created: {fine_tuning_job.id}")

        # Save configuration
        if save_to_config:
            config = {
                'model_name': self.model_name,
                'training_data_path': training_data_path,
                'training_file_id': training_file.id,
                'validation_file_id': validation_file_id,
                'num_examples': num_examples,
                'hyperparameters': hyperparams,
                'fine_tuning_job_id': fine_tuning_job.id,
                'description': description,
                'job_name': job_name
            }
            config_dir = os.path.join(os.path.dirname(training_data_path), 'ftjobs')
            os.makedirs(config_dir, exist_ok=True)
            config_name = job_name
            config_path = os.path.join(config_dir, f'{config_name}.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            logger.info(f"Configuration saved to {config_path}")

        # Log fine-tuning parameters to CSV
        log_file = os.path.join(os.path.dirname(training_data_path), 'finetuning_log.csv')
        log_exists = os.path.exists(log_file)
        
        import datetime
        import csv

        with open(log_file, 'a', newline='') as csvfile:
            fieldnames = ['timestamp', 'job_id', 'model_name', 'training_file_id', 'validation_file_id', 
                         'num_examples', 'n_epochs', 'batch_size', 'learning_rate_multiplier', 
                         'description', 'job_name']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not log_exists:
                writer.writeheader()
            
            writer.writerow({
                'timestamp': datetime.datetime.now().isoformat(),
                'job_id': fine_tuning_job.id,
                'model_name': self.model_name,
                'training_file_id': training_file.id,
                'validation_file_id': validation_file_id,
                'num_examples': num_examples,
                'n_epochs': n_epochs,
                'batch_size': batch_size,
                'learning_rate_multiplier': learning_rate_multiplier,
                'description': description,
                'job_name': job_name
            })
        
        logger.info(f"Fine-tuning log appended to {log_file}")
        return config_path

    def check_fine_tuning_status(self, job_id: str) -> str:
        """
        Checks the status of a fine-tuning job.

        Args:
            job_id (str): ID of the fine-tuning job.

        Returns:
            str: Status of the job.
        """
        client = openai.OpenAI(api_key=self.api_key)
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        logger.info(f"Job {job_id} status: {status}")
        return status

    def wait_for_fine_tuning(self, job_id: str, interval: int = 60):
        """
        Waits for the fine-tuning job to complete.

        Args:
            job_id (str): Fine-tuning job ID.
            interval (int): Time in seconds between status checks.
        """
        logger.info(f"Waiting for fine-tuning job {job_id} to complete")
        while True:
            status = self.check_fine_tuning_status(job_id)
            if status == 'succeeded':
                logger.info("Fine-tuning job succeeded")
                break
            elif status == 'failed':
                logger.error("Fine-tuning job failed")
                break
            else:
                time.sleep(interval)

    def get_finetuned_model_name(self, config: dict, api_keys: Optional[List[str]] = None) -> tuple:
        """
        Gets the output model name for a given fine-tuning job by directly querying the OpenAI API.

        Args:
            config (dict): Configuration dictionary containing job information.
            api_keys (List[str], optional): List of API keys to try.

        Returns:
            tuple: (model_name, api_key) if the job has succeeded, (None, None) otherwise.
        """
        job_id = config.get('fine_tuning_job_id')
        if api_keys is None:
            api_keys = [self.api_key]
            
        for api_key in api_keys:
            try:
                client = openai.OpenAI(api_key=api_key)
                jobs = client.fine_tuning.jobs.list()
                for job in jobs:
                    if job.status == "succeeded" and job.id == job_id:
                        return job.fine_tuned_model, api_key
            except Exception as e:
                logger.error(f"Error querying OpenAI API with key {api_key}: {e}")
                if api_key == api_keys[-1]:  # If this is the last API key
                    return None, None
                # If it's not the last API key, continue to the next one
        
        return None, None


class GeminiLLM:
    """
    A class to interact with Google's Gemini models for evaluation and fine-tuning.
    """
    
    def __init__(self, model_name: str = None, project: str = "970551384575", location: str = "us-central1"):
        """
        Initializes the GeminiLLM object.
        
        Args:
            model_name (str): Name of the Gemini model.
            project (str): Google Cloud project ID.
            location (str): Google Cloud region.
        """
        vertexai.init(project=project, location=location)
        self.model_name = model_name
        self.model = None
        self.generation_config = {
            "max_output_tokens": 100,
            "temperature": 0,
            "top_p": 1,
        }
        self.safety_settings = []
        self.project = project
        self.location = location
        logger.info(f"Initialized GeminiLLM with model {self.model_name}")

    def get_response(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        Obtains a response from the Gemini model.

        Args:
            system_prompt (str): System prompt.
            user_prompt (str): User prompt.
            **kwargs: Additional API arguments.

        Returns:
            str: The model's response.
        """
        self.model = GenerativeModel(
            self.model_name,
            system_instruction=[system_prompt] if system_prompt else None,
        )
        chat = self.model.start_chat(response_validation=False)
        try:
            response = chat.send_message(
                [user_prompt],
                generation_config=self.generation_config,
            )
            return response.text
        except Exception as e:
            logger.error(f"Error during Gemini API call: {e}")
            return None

    def evaluate(self, dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Evaluates the Gemini model on a given dataset.

        Args:
            dataset (pd.DataFrame): The dataset to evaluate.
            **kwargs: Additional API arguments.

        Returns:
            pd.DataFrame: DataFrame containing prompts, responses, and metadata.
        """
        logger.info("Starting Gemini evaluation")
        dataset_to_evaluate = dataset

        results = []
        for idx, row in tqdm(dataset_to_evaluate.iterrows(), total=len(dataset_to_evaluate), desc="Evaluating"):
            system_prompt = row['system_prompt']
            prompt = row['prompt']
            response = self.get_response(system_prompt, prompt, **kwargs)
            return_dict = {
                'config_name': row['config_name'],
                'prompt': prompt,
                'raw_response': response,
                'answer': row['answer'],
                'system_prompt': system_prompt,
                'model': self.model_name,
                'id': idx
            }
            for col, value in row.items():
                if col not in return_dict:
                    return_dict[col] = value
            results.append(return_dict)
        logger.info("Evaluation completed")
        return pd.DataFrame(results)

    async def evaluate_async(self, dataset: pd.DataFrame, requests_per_second: float = 30, **kwargs) -> pd.DataFrame:
        """
        Asynchronously evaluates the Gemini model on a given dataset.
        Note: This method is not truly asynchronous as the Vertex AI library doesn't support async operations.
        We're simulating async behavior for consistency with other LLM classes.

        Args:
            dataset (pd.DataFrame): The dataset to evaluate.
            requests_per_second (float): Maximum number of requests per second.
            **kwargs: Additional API arguments.

        Returns:
            pd.DataFrame: DataFrame containing prompts, responses, and metadata.
        """
        logger.info("Starting asynchronous Gemini evaluation")
        dataset_to_evaluate = dataset

        async def process_item(row):
            system_prompt = row['system_prompt']
            prompt = row['prompt']
            idx = row.name
            response = await self.get_response_async(system_prompt, prompt, **kwargs)
            return_dict = {
                'config_name': row['config_name'],
                'prompt': prompt,
                'raw_response': response,
                'answer': row['answer'],
                'system_prompt': system_prompt,
                'model': self.model_name,
                'id': idx
            }
            for col, value in row.items():
                if col not in return_dict:
                    return_dict[col] = value
            return return_dict

        results = []
        semaphore = asyncio.Semaphore(int(requests_per_second))
        delay = 1 / requests_per_second

        async def throttled_process_item(row):
            async with semaphore:
                result = await process_item(row)
                await asyncio.sleep(delay)
            return result

        tasks = [throttled_process_item(row) for _, row in dataset_to_evaluate.iterrows()]

        async for result in async_tqdm(asyncio.as_completed(tasks), 
                                       total=len(dataset_to_evaluate), desc="Evaluating"):
            results.append(await result)

        logger.info("Asynchronous evaluation completed")
        results_df = pd.DataFrame(results)
        return results_df

    async def get_response_async(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        Asynchronously obtains a response from the Gemini model.
        Note: This method is not truly asynchronous as the Vertex AI library doesn't support async operations.

        Args:
            system_prompt (str): System prompt.
            user_prompt (str): User prompt.
            **kwargs: Additional API arguments.

        Returns:
            str: The model's response.
        """
        # Simulating async behavior by calling the synchronous method
        return self.get_response(system_prompt, user_prompt, **kwargs)

    @staticmethod
    def convert_openai_to_gemini(openai_file, gemini_file):
        """
        Converts OpenAI format JSONL to Gemini format JSONL.

        Args:
            openai_file (str): Path to OpenAI format file.
            gemini_file (str): Path to output Gemini format file.
        """
        import json
        def convert_line(line):
            data = json.loads(line)
            messages = data["messages"]

            gemini_format = {"contents": []}

            for message in messages:
                role = message["role"]
                content = message["content"]

                if role == "system":
                    gemini_format["systemInstruction"] = {
                        "role": "system",
                        "parts": [{"text": content}],
                    }
                else:
                    gemini_format["contents"].append(
                        {
                            "role": "user" if role == "user" else "model",
                            "parts": [{"text": content}],
                        }
                    )

            return json.dumps(gemini_format)

        # Read input file and write to output file
        with open(openai_file, "r") as infile, open(gemini_file, "w") as outfile:
            for line in infile:
                converted_line = convert_line(line)
                outfile.write(converted_line + "\n")
        
        logger.info(f"Converted {openai_file} to Gemini format at {gemini_file}")


class TogetherLLM:
    """
    A class to interact with Together AI's language models for evaluation.
    """
    
    def __init__(self, model_name: str, api_key: str = None):
        """
        Initialize TogetherLLM with model name and API key.
        
        Args:
            model_name (str): Name of the Together AI model.
            api_key (str): Together API key. If not provided, uses 'TOGETHER_API_KEY' environment variable.
        """
        from together import Together
        self.model_name = model_name
        self.api_key = api_key or os.getenv('TOGETHER_API_KEY')
        if not self.api_key:
            raise ValueError("Together API key not found. Please set 'TOGETHER_API_KEY' environment variable or pass it as an argument.")
        self.client = Together(api_key=self.api_key)
        logger.info(f"Initialized TogetherLLM with model {self.model_name}")

    def get_response(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        Get response from Together API.
        
        Args:
            system_prompt (str): System prompt.
            user_prompt (str): User prompt.
            **kwargs: Additional API arguments.
            
        Returns:
            str: The model's response.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=kwargs.get('temperature', 0),
                max_tokens=kwargs.get('max_tokens', 100),
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error during Together API call: {e}")
            return ""

    def evaluate(self, dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataset (pd.DataFrame): The dataset to evaluate.
            **kwargs: Additional API arguments.
            
        Returns:
            pd.DataFrame: DataFrame containing prompts, responses, and metadata.
        """
        logger.info("Starting Together evaluation")
        dataset_to_evaluate = dataset

        results = []
        for idx, row in tqdm(dataset_to_evaluate.iterrows(), total=len(dataset_to_evaluate), desc="Evaluating"):
            system_prompt = row['system_prompt']
            prompt = row['prompt']
            response = self.get_response(system_prompt, prompt, **kwargs)
            return_dict = {
                'config_name': row['config_name'],
                'prompt': prompt,
                'raw_response': response,
                'answer': row['answer'],
                'system_prompt': system_prompt,
                'model': self.model_name,
                'id': idx
            }
            for col, value in row.items():
                if col not in return_dict:
                    return_dict[col] = value
            results.append(return_dict)
        
        logger.info("Evaluation completed")
        return pd.DataFrame(results)

    async def evaluate_async(self, dataset: pd.DataFrame, requests_per_second: float = 30, **kwargs) -> pd.DataFrame:
        """
        Asynchronously evaluate the model on a dataset.
        
        Args:
            dataset (pd.DataFrame): The dataset to evaluate.
            requests_per_second (float): Maximum number of requests per second.
            **kwargs: Additional API arguments.
            
        Returns:
            pd.DataFrame: DataFrame containing prompts, responses, and metadata.
        """
        logger.info("Starting asynchronous Together evaluation")
        from together import AsyncTogether
        async_client = AsyncTogether(api_key=self.api_key)

        async def process_item(row):
            system_prompt = row['system_prompt']
            prompt = row['prompt']
            idx = row.name
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            try:
                response = await async_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=kwargs.get('temperature', 0),
                    max_tokens=kwargs.get('max_tokens', 100),
                )
                result = response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Error during Together API call: {e}")
                result = ""

            return_dict = {
                'config_name': row['config_name'],
                'prompt': prompt,
                'raw_response': result,
                'answer': row['answer'],
                'system_prompt': system_prompt,
                'model': self.model_name,
                'id': idx
            }
            for col, value in row.items():
                if col not in return_dict:
                    return_dict[col] = value
            return return_dict

        results = []
        semaphore = asyncio.Semaphore(int(requests_per_second))
        delay = 1 / requests_per_second

        async def throttled_process_item(row):
            async with semaphore:
                result = await process_item(row)
                await asyncio.sleep(delay)
            return result

        tasks = [throttled_process_item(row) for _, row in dataset.iterrows()]

        async for result in async_tqdm(asyncio.as_completed(tasks), 
                                     total=len(dataset), desc="Evaluating"):
            results.append(await result)

        logger.info("Asynchronous evaluation completed")
        return pd.DataFrame(results)


class AnthropicLLM:
    """
    A class to interact with Anthropic's Claude models for evaluation.
    """

    def __init__(self, model_name: str, api_key: str = None):
        """
        Initializes the AnthropicLLM object.

        Args:
            model_name (str): Name of the Anthropic model (e.g., 'claude-3-sonnet-20240229')
            api_key (str): Anthropic API key. If not provided, uses 'ANTHROPIC_API_KEY' environment variable.
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("Anthropic API key not found. Please set 'ANTHROPIC_API_KEY' environment variable or pass it as an argument.")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        logger.info(f"Initialized AnthropicLLM with model {self.model_name}")

    def get_response(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        Obtains a response from Claude.

        Args:
            system_prompt (str): System prompt.
            user_prompt (str): User prompt.
            **kwargs: Additional API arguments.

        Returns:
            str: Claude's response.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "user", "content": system_prompt+'\n'+user_prompt})
        else:
            messages.append({"role": "user", "content": user_prompt})

        try:
            response = self.client.messages.create(
                model=self.model_name,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', 50),
                temperature=kwargs.get('temperature', 0),
                **{k:v for k,v in kwargs.items() if k not in ['max_tokens', 'temperature']}
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error during Anthropic API call: {e}")
            return ""

    def evaluate(self, dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Evaluates Claude on a given dataset.

        Args:
            dataset (pd.DataFrame): The dataset to evaluate.
            **kwargs: Additional arguments for the Anthropic API.

        Returns:
            pd.DataFrame: DataFrame containing prompts, responses, and metadata.
        """
        logger.info("Starting Anthropic evaluation")
        dataset_to_evaluate = dataset

        results = []
        for idx, row in tqdm(dataset_to_evaluate.iterrows(), total=len(dataset_to_evaluate), desc="Evaluating"):
            system_prompt = row['system_prompt']
            prompt = row['prompt']
            idx = row.name
            response = self.get_response(system_prompt, prompt, **kwargs)
            return_dict = {
                'config_name': row['config_name'],
                'prompt': prompt,
                'raw_response': response,
                'answer': row['answer'],
                'system_prompt': system_prompt,
                'model': self.model_name,
                'id': idx
            }
            for col, value in row.items():
                if col not in return_dict:
                    return_dict[col] = value
            results.append(return_dict)
        logger.info("Evaluation completed")
        return pd.DataFrame(results)

    async def evaluate_async(self, dataset: pd.DataFrame, requests_per_second: float = 30, **kwargs) -> pd.DataFrame:
        """
        Asynchronously evaluates Claude on a given dataset.

        Args:
            dataset (pd.DataFrame): The dataset to evaluate.
            requests_per_second (float): Maximum number of requests per second.
            **kwargs: Additional arguments for the Anthropic API.

        Returns:
            pd.DataFrame: DataFrame containing prompts, responses, and metadata.
        """
        logger.info("Starting asynchronous Anthropic evaluation")
        dataset_to_evaluate = dataset

        async def process_item(row):
            system_prompt = row['system_prompt']
            prompt = row['prompt']
            idx = row.name
            response = await self.get_response_async(system_prompt, prompt, **kwargs)
            return_dict = {
                'config_name': row['config_name'],
                'prompt': prompt,
                'raw_response': response,
                'answer': row['answer'],
                'system_prompt': system_prompt,
                'model': self.model_name,
                'id': idx
            }
            for col, value in row.items():
                if col not in return_dict:
                    return_dict[col] = value
            return return_dict

        results = []
        semaphore = asyncio.Semaphore(int(requests_per_second))
        delay = 1 / requests_per_second

        async def throttled_process_item(row):
            async with semaphore:
                result = await process_item(row)
                await asyncio.sleep(delay)
            return result

        tasks = [throttled_process_item(row) for _, row in dataset_to_evaluate.iterrows()]

        async for result in async_tqdm(asyncio.as_completed(tasks), 
                                     total=len(dataset_to_evaluate), desc="Evaluating"):
            results.append(await result)

        logger.info("Asynchronous evaluation completed")
        return pd.DataFrame(results)

    async def get_response_async(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        Asynchronously obtains a response from Claude.

        Args:
            system_prompt (str): System prompt.
            user_prompt (str): User prompt.
            **kwargs: Additional API arguments.

        Returns:
            str: Claude's response.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "user", "content": system_prompt+'\n'+user_prompt})
        else:
            messages.append({"role": "user", "content": user_prompt})

        try:
            client = anthropic.AsyncAnthropic(api_key=self.api_key)
            # Create a task for the API call
            api_task = client.messages.create(
                model=self.model_name,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', 50),
                temperature=kwargs.get('temperature', 0),
                **{k:v for k,v in kwargs.items() if k not in ['max_tokens', 'temperature']}
            )
            
            # Wait for the task to complete with a timeout
            response = await asyncio.wait_for(api_task, timeout=60.0)
            return response.content[0].text
        except asyncio.TimeoutError:
            logger.error("Request timed out after 60 seconds")
            return ""
        except Exception as e:
            logger.error(f"Error during Anthropic API call: {e}")
            return ""