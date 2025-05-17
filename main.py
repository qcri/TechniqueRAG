#!/usr/bin/env python3
import json
import re
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

def extract_mitre_techniques(text):
    pattern = re.compile(r'T\d{4}(?:\.\d{3})?')
    matches = set(pattern.findall(text))
    return sorted(matches)

def load_dataset(file_path):
    """Load the dataset from the JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_results(data, name):
    """Save the updated dataset with predictions to the results folder."""
    # Create results directory if it doesn't exist
    os.makedirs('./results', exist_ok=True)
    
    output_path = f'./results/{name}_results.json'
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Results saved to {output_path}")

def run_inference(data, base_url, model_name):
    """Perform inference on the dataset using the OpenAI API."""
    print(f"Running inference on {len(data)} examples using model: {model_name}")
    
    # Initialize OpenAI client with custom base URL
    client = OpenAI(
        base_url=base_url,
        api_key="dummy-key"
    )
    
    for i, example in enumerate(tqdm(data)):
        instruction = example["instruction"]
        input_text = example["input"]
        
        # Combine instruction and input
        prompt = f"# Instruction:\n{instruction}\n# Input:\n{input_text}\n# Response:\n"   

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=151,
            temperature=0.7
        )
        
        # Extract response text
        response_text = response.choices[0].message.content

        # Extract MITRE techniques from the response
        predicted_techniques = extract_mitre_techniques(response_text)
        
        # Update the dataset entry
        data[i]["predicted"] = predicted_techniques
    
    return data

def main():
    parser = argparse.ArgumentParser(description="Run inference on a dataset using OpenAI API and extract MITRE techniques")
    parser.add_argument("--name", required=True, help="Name of the dataset file (without .json extension)")
    parser.add_argument("--base_url", default="http://localhost:9003/v1", help="Base URL of the vLLM hosted model with OpenAI-compatible API")
    parser.add_argument("--model", required=True, help="Name of the model to use for inference")
    args = parser.parse_args()
    
    # Construct the dataset path
    dataset_path = Path(f"datasets/TechniqueRAG-Datasets/test/{args.name}.json")
    
    if not dataset_path.exists():
        print(f"Error: Dataset file {dataset_path} does not exist.")
        return
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    data = load_dataset(dataset_path)
    
    # Run inference and update dataset
    data = run_inference(data, args.base_url, args.model)
    
    # Save results
    save_results(data, args.name)
    
    print("Done!")

if __name__ == "__main__":
    main()
