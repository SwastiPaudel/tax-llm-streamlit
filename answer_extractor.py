# Assuming the data is stored in a variable called data
# Parse the AIMessage structure

# If you have the raw text exactly as provided:
import json
from ast import literal_eval


# Method 1: If it's already a Python dictionary
def extract_answer_from_dict(data):
    # Access the tool_calls list
    tool_calls = data.get('tool_calls', [])

    # Find the relevant tool call with the answer
    for tool_call in tool_calls:
        if tool_call.get('name') == 'AnswerWithSources':
            # Return the answer content
            return tool_call.get('args', {}).get('answer')

    return None


# Method 2: If it's a string that needs parsing
def extract_answer_from_string(text_data):
    # Convert string representation to Python dictionary
    # This is a simplified approach - in practice you might need
    # more robust parsing depending on the exact format
    try:
        # Try to evaluate as a Python literal
        data = literal_eval(text_data)
        return extract_answer_from_dict(data)
    except:
        # If that fails, try a different approach
        # Look for the answer field in the string
        import re
        pattern = r'"answer": "(.*?)",'
        match = re.search(pattern, text_data, re.DOTALL)
        if match:
            # Clean up escaped characters
            return match.group(1).replace('\\n', '\n').replace('\\"', '"')
        return None


# Method 3: If you're accessing just the partial_json field
def extract_from_partial_json(text_data):
    # Find the partial_json section and extract its content
    import re
    pattern = r"'partial_json':\s*'(.*?)'\}"
    match = re.search(pattern, text_data, re.DOTALL)
    if match:
        json_str = match.group(1).replace('\\n', '\n').replace('\\"', '"')
        try:
            data = json.loads(json_str)
            return data.get('answer')
        except:
            return None
    return None