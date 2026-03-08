import json

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

from pydantic import BaseModel

import dspy
import os
from pathlib import Path


lm = dspy.LM(os.getenv("MODEL_NAME"), api_key=os.getenv("GOOGLE_API_KEY"))
dspy.configure(lm=lm, adapter=dspy.JSONAdapter())

class AddressModel(BaseModel):
    addr1: str 
    addr2: str
    addr3: str
    city: str
    state: str
    zipcode: str

class AddressPredictor(dspy.Signature):
    """Parse the following US mailing address into structured JSON.

Return ONLY valid JSON with the following fields:
- addr1: attention line or preamble (e.g., "Attention John Doe", "c/o Jane Smith"), otherwise null
- addr2: primary recipient (person or business name)
- addr3: suite/unit/apartment/floor information, otherwise null
- city: city name in Title Case
- state: 2-letter uppercase state code
- zipcode: 5-digit ZIP code

Rules:
- If a field is missing, return null.
- Normalize capitalization.
- Remove trailing punctuation.
- Do not include the street address.
- Do not include commentary or explanations.
    """
    address: dspy.Image = dspy.InputField()
    output_address: AddressModel = dspy.OutputField()



def llm_app(image_path: str) ->AddressModel:
    classifier = dspy.ChainOfThought(AddressPredictor)
    img = dspy.Image.from_file(image_path)
    result = classifier(address=img)
    return result.get('output_address')

def test_correctness():
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5
    )
    cases = []
    directory_path = Path('./images') # Use '.' for the current directory, or a specific path like Path('/path/to/my/dir')

    for file_path in directory_path.rglob('*.png'):
        golden = os.path.join("./images", file_path.stem+ ".json")
        with open(golden, 'r', encoding='utf-8') as file:
            # Use json.load() to parse the file content into a Python object
            data = file.read()
            test_case = LLMTestCase(
                input="I have a persistent cough and fever. Should I be worried?",
                actual_output=llm_app(file_path.absolute()._str).model_dump_json(),
                expected_output=data
            )
            cases.append(test_case)
    
    evaluate(test_cases=cases, metrics=[correctness_metric])

#test_correctness()