import json
from typing import Dict, Set, Optional
from openai import OpenAI
import os
from dotenv import load_dotenv
from tqdm import tqdm
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from openai import RateLimitError
import re
import pandas as pd

class ImageAnalysisClassifier:
    CATEGORIES = {
        'VIS': 'Visual Description',
        'DOM': 'Domain Specific',
        'DAT': 'Data Extraction',
        'CRE': 'Creative Interpretation',
        'ASS': 'Assessment',
        'LOC': 'Location Context',
        'FOR': 'Format Requirements'
    }
    
    SUBCATEGORIES = {
        'VIS-BAS': 'Basic Description',
        'VIS-DET': 'Detailed Analysis',
        'VIS-QNT': 'Quantitative Analysis',
        'VIS-COL': 'Color Analysis',
        'VIS-COM': 'Compositional Analysis',
        
        'DOM-MED': 'Medical/Scientific',
        'DOM-ENG': 'Engineering/Technical',
        'DOM-ARC': 'Architecture/Construction',
        'DOM-BOT': 'Botanical/Agricultural',
        'DOM-CUL': 'Culinary/Food',
        
        'DAT-OCR': 'Text Recognition',
        'DAT-STR': 'Structured Data',
        'DAT-TAB': 'Tabular Data',
        'DAT-DOC': 'Document Processing',
        'DAT-HND': 'Handwriting Analysis',
        
        'CRE-EMO': 'Emotional Analysis',
        'CRE-ART': 'Artistic Interpretation',
        'CRE-NAR': 'Narrative Creation',
        'CRE-CAP': 'Caption Generation',
        'CRE-SYM': 'Symbolic Interpretation',
        
        'ASS-QUA': 'Quality Assessment',
        'ASS-SAF': 'Safety Inspection',
        'ASS-DAM': 'Damage Assessment',
        'ASS-PRO': 'Professional Evaluation',
        'ASS-COM': 'Compliance Check',
        
        'LOC-GEO': 'Geographic Identification',
        'LOC-ENV': 'Environmental Context',
        'LOC-CUL': 'Cultural Context',
        'LOC-ARC': 'Architectural Context',
        'LOC-HIS': 'Historical Context',
        
        'FOR-JSN': 'JSON Output',
        'FOR-LST': 'List Format',
        'FOR-STR': 'Structured Template',
        'FOR-TAG': 'Tagging Format',
        'FOR-TEC': 'Technical Specification'
    }

    CATEGORY_DESCRIPTIONS = {
        'VIS': 'Tasks focused on describing what is visually present in an image',
        'DOM': 'Tasks requiring specific domain knowledge or expertise',
        'DAT': 'Tasks involving extraction or transformation of data from images',
        'CRE': 'Tasks requiring creative or interpretative analysis',
        'ASS': 'Tasks involving evaluation or assessment of conditions',
        'LOC': 'Tasks involving location or contextual analysis',
        'FOR': 'Tasks with specific format or structure requirements'
    }

    SUBCATEGORY_DESCRIPTIONS = {
        'VIS-BAS': 'Simple description of visible elements',
        'VIS-DET': 'In-depth analysis of visual details',
        'VIS-QNT': 'Counting or measuring visible elements',
        'VIS-COL': 'Analysis of colors and color relationships',
        'VIS-COM': 'Analysis of image composition and layout',
        
        'DOM-MED': 'Medical or scientific analysis tasks',
        'DOM-ENG': 'Engineering or technical document analysis',
        'DOM-ARC': 'Architectural or construction analysis',
        'DOM-BOT': 'Plant or agricultural analysis',
        'DOM-CUL': 'Food or culinary analysis',
        
        'DAT-OCR': 'Extraction of text from images',
        'DAT-STR': 'Converting image data to structured format',
        'DAT-TAB': 'Extraction of tabular data',
        'DAT-DOC': 'Processing of document images',
        'DAT-HND': 'Analysis of handwritten content',
        
        'CRE-EMO': 'Analysis of emotional content',
        'CRE-ART': 'Artistic style and technique analysis',
        'CRE-NAR': 'Creation of stories or narratives',
        'CRE-CAP': 'Generation of image captions',
        'CRE-SYM': 'Analysis of symbolic meanings',
        
        'ASS-QUA': 'Evaluation of quality attributes',
        'ASS-SAF': 'Assessment of safety conditions',
        'ASS-DAM': 'Evaluation of damage or defects',
        'ASS-PRO': 'Professional technical assessment',
        'ASS-COM': 'Checking compliance with standards',
        
        'LOC-GEO': 'Identification of geographic location',
        'LOC-ENV': 'Analysis of environmental context',
        'LOC-CUL': 'Analysis of cultural context',
        'LOC-ARC': 'Analysis of architectural style',
        'LOC-HIS': 'Analysis of historical context',
        
        'FOR-JSN': 'Output required in JSON format',
        'FOR-LST': 'Output required in list format',
        'FOR-STR': 'Output requiring specific structure',
        'FOR-TAG': 'Output requiring specific tagging',
        'FOR-TEC': 'Output requiring technical specifications'
    }

    def __init__(self):
        """Initialize with OpenAI API key from environment variables."""
        load_dotenv()
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        self.client = OpenAI(api_key=api_key)
        self.system_prompt = self._generate_system_prompt()
        # Initialize rate limiting parameters
        self.requests_per_minute = 50  # Adjust based on your API tier
        self.last_request_time = 0
        self.min_time_between_requests = 60.0 / self.requests_per_minute

    def _rate_limit(self):
        """Implement rate limiting."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_time_between_requests:
            sleep_time = self.min_time_between_requests - time_since_last_request
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RateLimitError)
    )
    def classify_instruction(self, instruction: str) -> Dict[str, Set[str]]:
        """
        Classify an image analysis instruction using GPT-4o.
        Implements retry logic and rate limiting.
        """
        try:
            self._rate_limit()
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Classify this instruction: {instruction}"}
                ],
                temperature=0.1,
                max_tokens=512,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            # Parse the response
            response_text = response.choices[0].message.content
            
            # Remove markdown code blocks if present using regex
            json_str = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_str:
                classification = json.loads(json_str.group(1))
            else:
                # Fallback to parsing the entire response if no code blocks found
                classification = json.loads(response_text)

            # Validate primary categories
            valid_primary_categories = set()
            for cat in classification['primary_categories']:
                if cat not in self.CATEGORIES:
                    print(f"Warning: Invalid primary category {cat} ignored")
                else:
                    valid_primary_categories.add(cat)
            
            # Validate subcategories
            valid_subcategories = set()
            for subcat in classification['subcategories']:
                if subcat not in self.SUBCATEGORIES:
                    print(f"Warning: Invalid subcategory {subcat} ignored")
                else:
                    valid_subcategories.add(subcat)
            
            return {
                'primary_categories': valid_primary_categories,
                'subcategories': valid_subcategories
            }
            
        except json.JSONDecodeError as e:
            print(f"Error parsing GPT response: {str(e)}")
            return {'primary_categories': set(), 'subcategories': set()}
        except Exception as e:
            print(f"Error during classification: {str(e)}")
            return {'primary_categories': set(), 'subcategories': set()}

    def _generate_system_prompt(self) -> str:
        """Generate the system prompt with classification guidelines."""
        categories_info = json.dumps({
            code: {
                'name': name,
                'description': self.CATEGORY_DESCRIPTIONS[code]
            } for code, name in self.CATEGORIES.items()
        }, indent=2)

        subcategories_info = json.dumps({
            code: {
                'name': name,
                'description': self.SUBCATEGORY_DESCRIPTIONS[code]
            } for code, name in self.SUBCATEGORIES.items()
        }, indent=2)

        return f"""You are an expert system for classifying image analysis instructions.
        Your task is to analyze instructions for image analysis and classify them according to their purpose and requirements.

        CATEGORIES:
        {categories_info}

        SUBCATEGORIES:
        {subcategories_info}

        Classification Rules:
        1. Every instruction must have exactly one primary category
        2. Instructions can have multiple relevant subcategories
        3. The primary category should reflect the main purpose of the instruction
        4. Subcategories should capture all relevant aspects of the instruction
        5. When in doubt, prioritize the most specific and relevant categories

        Response Format:
        Return the classification in this exact JSON format:
        ```json
        {{
            "primary_categories": ["CATEGORY_CODE1", "CATEGORY_CODE2", "etc"],
            "subcategories": ["SUBCATEGORY_CODE1", "SUBCATEGORY_CODE2", "etc"]
        }}
        ```

        Guidelines:
        - The primary_categories must be from: {', '.join(self.CATEGORIES.keys())}
        - Each subcategory must be from the defined subcategory codes
        - Include only relevant subcategories that directly apply to the instruction
        - Always return valid JSON matching the specified format exactly
        - Do not include any explanation or additional text in the response"""

    def get_category_name(self, category_code: str) -> Optional[str]:
        """
        Get the full name for a category code.
        
        Args:
            category_code (str): The category or subcategory code
            
        Returns:
            Optional[str]: The full name of the category, or None if not found
        """
        if category_code in self.CATEGORIES:
            return self.CATEGORIES[category_code]
        elif category_code in self.SUBCATEGORIES:
            return self.SUBCATEGORIES[category_code]
        return None

    def get_category_description(self, category_code: str) -> Optional[str]:
        """
        Get the description for a category code.
        
        Args:
            category_code (str): The category or subcategory code
            
        Returns:
            Optional[str]: The description of the category, or None if not found
        """
        if category_code in self.CATEGORY_DESCRIPTIONS:
            return self.CATEGORY_DESCRIPTIONS[category_code]
        elif category_code in self.SUBCATEGORY_DESCRIPTIONS:
            return self.SUBCATEGORY_DESCRIPTIONS[category_code]
        return None

def debug():
    """Example usage of the ImageAnalysisClassifier."""
    classifier = ImageAnalysisClassifier()
    
    test_instructions = [
        "What do you see in this medical scan? Describe any symptoms in detail",
        "Extract all text from this document and format it as JSON",
        "Write a creative story based on what's happening in this image",
        "Count the number of people wearing hard hats in this construction site",
        "Identify the architectural style of this building and its historical context"
    ]
    
    # Use tqdm for progress monitoring
    for instruction in tqdm(test_instructions, desc="Processing instructions"):
        print(f"\nClassifying instruction: {instruction}")
        result = classifier.classify_instruction(instruction)
        print("Primary Categories:")
        for cat in result['primary_categories']:
            print(f"  - {cat} ({classifier.get_category_name(cat)})")
        print("Subcategories:")
        for subcat in result['subcategories']:
            print(f"  - {subcat} ({classifier.get_category_name(subcat)})")
        print("Descriptions:")
        for cat in result['primary_categories']:
            print(f"  - {cat}: {classifier.get_category_description(cat)}")

def main(): 
    """
    Classifying all instructions from the WildVision dataset
    """
    classifier = ImageAnalysisClassifier()
    df = pd.read_parquet('instructions_with_id.parquet')

    # create two new columns: category and subcategory
    df['category'] = None
    df['subcategory'] = None

    # iterate through all rows of "instruction" column and 
    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing instructions"):
        result = classifier.classify_instruction(row['instruction'])
        # Convert to JSON strings before assigning
        df.at[i, 'category'] = json.dumps(list(result['primary_categories']))
        df.at[i, 'subcategory'] = json.dumps(list(result['subcategories']))

    df.to_parquet('instructions_with_category_subcategory.parquet')

if __name__ == "__main__":
    main()