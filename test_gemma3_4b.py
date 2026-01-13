import json
import requests
import yfinance as yf
from llama_cpp import Llama
import time
import re


# ----------------- Tools / Functions ----------------- #

def get_stock_price(symbol):
    """Fetch the stock data"""
    stock = yf.Ticker(symbol)
    if 'open' in stock.info:
        current_price = stock.info['open']
        s = f"Open price for {symbol} is " + str(current_price)
    else:
        current_price = None
        s = f"Could not retrieve the open price for {symbol}. Key 'open' not found."
    return s

def get_weather(city="Singapore", api_key="d2f1bddd18b7df4ded8003d6132f6b80"):
    """Get weather for a city"""
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}q={city}&appid={api_key}&units=metric"
    response = requests.get(complete_url)
    weather_data = response.json()

    if weather_data["cod"] != "404":
        main = weather_data["main"]
        weather = weather_data["weather"][0]
        temperature = main["temp"]
        pressure = main["pressure"]
        humidity = main["humidity"]
        weather_description = weather["description"]
        s = (
            f"The weather in {city} is as follows:\n"
            f"Temperature: {temperature}°C\n"
            f"Pressure: {pressure} hPa\n"
            f"Humidity: {humidity}%\n"
            f"Description: {weather_description.capitalize()}"
        )
    else:
        s = "City not found. Please check the city name."
    return s

def make_coffee(types_of_coffee='long black', milk='normal', sugar='normal', strength='normal'):
    """Make a cup of coffee"""
    display = f"""Making a cup of {types_of_coffee} with the following options:
Milk = {milk},
Sugar = {sugar}
Strength = {strength}
"""
    return display

def cook_burger(cook="well done"):
    """Cook a beef burger"""
    display = f"Cooking a beef burger that is {cook}"
    return display

def cook_fries(type_of_fries="straight"):
    """Cook fries"""
    display = f"Cooking {type_of_fries} fries"
    return display

def cook_prawn_noodles(prawn="with prawn", sotong="with sotong"):
    """Cook fried prawn noodles"""
    display = f"""Cooking fried prawn noodles with the following options:
Prawn = {prawn},
Sotong = {sotong}
"""
    return display


# ----------------- Dispatcher ----------------- #

AVAILABLE_FUNCTIONS = {
    "get_stock_price": get_stock_price,
    "get_weather": get_weather,
    "make_coffee": make_coffee,
    "cook_burger": cook_burger,
    "cook_fries": cook_fries,
    "cook_prawn_noodles": cook_prawn_noodles,
}


# ----------------- Gemma 3 Format Functions ----------------- #

def create_gemma3_function_definitions(tools_list):
    """Create Python-style function definitions for Gemma 3"""
    definitions = []
    
    for tool in tools_list:
        func = tool['function']
        name = func['name']
        desc = func['description']
        params = func['parameters']
        
        # Build parameter list with types and descriptions
        param_parts = []
        for prop_name, prop_info in params.get('properties', {}).items():
            prop_type = prop_info.get('type', 'string')
            prop_desc = prop_info.get('description', '')
            
            # Format: param_name: type = default_value  # description
            param_str = f"{prop_name}: str"
            if prop_name not in params.get('required', []):
                param_str += f" = None"
            param_str += f"  # {prop_desc}"
            
            # Add enum info if exists
            if 'enum' in prop_info:
                enum_values = ', '.join([f"'{v}'" for v in prop_info['enum']])
                param_str += f" Options: [{enum_values}]"
            
            param_parts.append(param_str)
        
        # Create function signature
        params_str = ", ".join(param_parts)
        function_def = f"def {name}({params_str}):\n    \"\"\"{desc}\"\"\"\n    pass"
        definitions.append(function_def)
    
    return "\n\n".join(definitions)


def create_gemma3_system_prompt(tools_list):
    """Create system prompt for Gemma 3 tool calling"""
    function_definitions = create_gemma3_function_definitions(tools_list)
    
    system_prompt = f"""
You are a helpful AI assistant that can chat naturally and call tools when needed.

- You have access to the following Python functions (they are already imported and available):

{function_definitions}

Guidelines:
- If the user clearly asks you to do something that one of these functions can perform, call it inside a ```tool_code``` block using valid Python, for example:
  ```tool_code
  get_weather(city="Tokyo")
  ```
- If the request is unclear or missing required details, ask a brief clarification question instead of guessing.
- If the request is general chit-chat or a normal question (greetings, jokes, facts, recommendations, etc.), respond conversationally and do NOT call any function.
- Only use the functions listed above. Do not invent new functions.
"""
    
    return system_prompt


# ----------------- Extraction for Gemma 3 ----------------- #

def extract_tool_call_gemma3(output_text):
    """Extract function call from Gemma 3 ```tool_code``` blocks"""
    try:
        # Look for ```tool_code ... ``` pattern with various formats
        patterns = [
            r'```tool_code\s*\n(.*?)\n```',  # Complete block with newlines
            r'```tool_code\s+(.*?)```',       # Complete block inline
            r'```tool_code\s*\n(.*?)$',       # Incomplete - newline format
            r'```tool_code\s+(.*?)$',         # Incomplete - inline format
            r'```tool_code\s*$',               # Just the opening tag
        ]
        
        code_block = None
        for pattern in patterns:
            match = re.search(pattern, output_text, re.DOTALL | re.MULTILINE)
            if match:
                if match.lastindex and match.lastindex >= 1:
                    code_block = match.group(1).strip()
                break
        
        # If we only found the opening tag, return None
        if code_block is None or code_block == '':
            return None, None
        
        # Parse Python function call: function_name(arg1="value1", arg2="value2")
        func_pattern = r'(\w+)\s*\((.*?)\)'
        func_match = re.search(func_pattern, code_block, re.DOTALL)
        
        if not func_match:
            return None, None
        
        function_name = func_match.group(1)
        args_str = func_match.group(2).strip()
        
        # Parse arguments - handle multiple formats
        arguments = {}
        if args_str:
            # Match: key="value" or key='value'
            arg_pattern = r"(\w+)\s*=\s*['\"]([^'\"]*)['\"]"
            arg_matches = re.findall(arg_pattern, args_str)
            
            for key, value in arg_matches:
                arguments[key] = value.strip()
        
        return function_name, arguments
    
    except Exception as e:
        print(f"Error extracting tool call: {e}")
        return None, None


def execute_function_call(function_name, arguments):
    """Safely execute a function call"""
    if function_name not in AVAILABLE_FUNCTIONS:
        return f"Error: Function '{function_name}' not found"

    try:
        func = AVAILABLE_FUNCTIONS[function_name]
        result = func(**arguments)
        return result
    except Exception as e:
        return f"Error executing {function_name}: {str(e)}"


# ----------------- Tools Schema ----------------- #

tools_list = [
    {
        "type": "function",
        "function": {
            "name": "cook_burger",
            "description": "Cook a beef burger with specified doneness level",
            "parameters": {
                "type": "object",
                "properties": {
                    "cook": {
                        "type": "string",
                        "description": "How well done the burger should be cooked",
                        "enum": ["well done", "medium", "rare"],
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cook_fries",
            "description": "Cook potato fries in different styles",
            "parameters": {
                "type": "object",
                "properties": {
                    "type_of_fries": {
                        "type": "string",
                        "description": "The style of fries to cook",
                        "enum": ["straight", "curly"],
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cook_prawn_noodles",
            "description": "Cook fried prawn noodles. Customize with prawns and sotong (squid) options",
            "parameters": {
                "type": "object",
                "properties": {
                    "prawn": {
                        "type": "string",
                        "description": "Whether to include prawns in the dish",
                        "enum": ["with prawn", "without prawn"],
                    },
                    "sotong": {
                        "type": "string",
                        "description": "Whether to include sotong (squid) in the dish",
                        "enum": ["with sotong", "without sotong"],
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_coffee",
            "description": "Make a cup of coffee with various customization options for type, milk, sugar, and strength",
            "parameters": {
                "type": "object",
                "properties": {
                    "types_of_coffee": {
                        "type": "string",
                        "description": "The type of coffee drink to make",
                    },
                    "milk": {
                        "type": "string",
                        "description": "Amount of milk to add",
                        "enum": ["normal", "no", "more", "less"],
                    },
                    "sugar": {
                        "type": "string",
                        "description": "Amount of sugar to add",
                        "enum": ["normal", "no", "more", "less"],
                    },
                    "strength": {
                        "type": "string",
                        "description": "Coffee strength level",
                        "enum": ["normal", "strong", "weak"],
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a given stock ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The stock ticker symbol",
                    }
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather conditions including temperature, humidity, and description for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name to get weather for",
                    }
                },
                "required": ["city"],
            },
        },
    },
]


# ----------------- Test Cases ----------------- #

test_cases = [
    {
        "name": "Test 1: Food ordering (prawn noodles)",
        "query": "Can I have a plate of fried prawn noodles without sotong please?",
        "expected_function": "cook_prawn_noodles",
        "type": "function_call"
    },
    {
        "name": "Test 2: Coffee order",
        "query": "I'd like a cappuccino with no sugar and extra milk",
        "expected_function": "make_coffee",
        "type": "function_call"
    },
    {
        "name": "Test 3: Weather query",
        "query": "What's the weather like in Tokyo?",
        "expected_function": "get_weather",
        "type": "function_call"
    },
    {
        "name": "Test 4: Stock price",
        "query": "Can you check the stock price for AAPL?",
        "expected_function": "get_stock_price",
        "type": "function_call"
    },
    {
        "name": "Test 5: Burger order",
        "query": "I want a medium burger please",
        "expected_function": "cook_burger",
        "type": "function_call"
    },
    {
        "name": "Test 6: Fries order",
        "query": "Give me some curly fries",
        "expected_function": "cook_fries",
        "type": "function_call"
    },
    {
        "name": "Test 7: Greeting",
        "query": "Hello! How are you?",
        "expected_function": None,
        "type": "conversation"
    },
    {
        "name": "Test 8: General question",
        "query": "What can I do in Singapore?",
        "expected_function": None,
        "type": "conversation"
    },
    {
        "name": "Test 9: Casual conversation",
        "query": "Tell me a fun fact about Elon Musk",
        "expected_function": None,
        "type": "conversation"
    },
    {
        "name": "Test 10: Casual conversation",
        "query": "Tell me a joke about NVIDIA?",
        "expected_function": None,
        "type": "conversation"
    },
]


# ----------------- Model Init ----------------- #

print("Loading Gemma 3 4B model...")
llm = Llama(
    model_path="./model/gemma-3-4b-it-q4_0.gguf",
    n_ctx=8192,
    n_gpu_layers=-1,
    logits_all=False,
    vocab_only=False,
    seed=42,
    verbose=False,
)
print("Model loaded successfully!\n")


# ----------------- Run Tests ----------------- #

system_prompt = create_gemma3_system_prompt(tools_list)

results = {
    "passed": 0,
    "failed": 0,
    "total": len(test_cases),
    "details": []
}

print("=" * 80)
print("STARTING GEMMA 3 4B TEST SUITE")
print("=" * 80)
print()

for idx, test in enumerate(test_cases, 1):
    print(f"\n{'=' * 80}")
    print(f"{test['name']}")
    print(f"{'=' * 80}")
    print(f"Query: {test['query']}")
    print(f"Expected: {'Function call to ' + test['expected_function'] if test['expected_function'] else 'Normal conversation'}")
    print()
    
    # Build Gemma 3 prompt
    full_prompt = f"<start_of_turn>user\n{system_prompt}\n\n{test['query']}<end_of_turn>\n<start_of_turn>model\n"
    
    start = time.time()
    response = llm(
        full_prompt,
        max_tokens=512,
        temperature=0.1,
        stop=["<end_of_turn>"],
        echo=False,
    )
    end = time.time()
    
    assistant_content = response["choices"][0]["text"].strip()

    # Check if we have an incomplete tool_code block
    if "```tool_code" in assistant_content:
        # Count backticks after tool_code
        tool_code_idx = assistant_content.rfind("```tool_code")
        remaining_text = assistant_content[tool_code_idx:]
        
        # Check if there's a closing ``` after the opening
        closing_match = re.search(r'```tool_code.*?```', remaining_text, re.DOTALL)
        
        if not closing_match:
            # Incomplete block - need to continue generation
            print("⚠ Incomplete tool_code block detected, continuing generation...")
            
            # Continue from where we left off
            continue_response = llm(
                full_prompt + assistant_content,
                max_tokens=128,
                temperature=0.1,
                stop=["```", "\n\n", "<end_of_turn>"],
                echo=False,
            )
            
            continuation = continue_response["choices"][0]["text"].strip()
            assistant_content += continuation
            
            # Add closing backticks if still missing
            if "```" not in continuation:
                assistant_content += "\n```"

    inference_time = end - start
    
    print(f"Model output: {assistant_content}")
    print(f"Time taken: {inference_time:.2f}s")
    print()
    
    function_name, arguments = extract_tool_call_gemma3(assistant_content)
    
    test_passed = False
    result_detail = {
        "test_name": test['name'],
        "query": test['query'],
        "output": assistant_content,
        "time": inference_time,
    }
    
    if function_name:
        print(f"✓ Function detected: {function_name}")
        print(f"✓ Arguments: {arguments}")
        
        try:
            func_result = execute_function_call(function_name, arguments)
            print(f"\nFunction execution result:")
            print(func_result)
            result_detail["function_called"] = function_name
            result_detail["arguments"] = arguments
            result_detail["function_result"] = func_result
        except Exception as e:
            print(f"\n✗ Function execution failed: {e}")
            result_detail["error"] = str(e)
        
        if test['type'] == 'function_call' and function_name == test['expected_function']:
            test_passed = True
            print(f"\n✓ TEST PASSED - Correct function called")
        elif test['type'] == 'conversation':
            test_passed = False
            print(f"\n✗ TEST FAILED - Function called when conversation expected")
        else:
            test_passed = False
            print(f"\n✗ TEST FAILED - Wrong function (expected: {test['expected_function']})")
    else:
        print("ℹ No function call detected - treating as normal conversation")
        print(f"Response: {assistant_content}")
        result_detail["conversation_response"] = assistant_content
        
        if test['type'] == 'conversation':
            test_passed = True
            print(f"\n✓ TEST PASSED - Normal conversation as expected")
        else:
            test_passed = False
            print(f"\n✗ TEST FAILED - Expected function call to {test['expected_function']}")
    
    result_detail["passed"] = test_passed
    results["details"].append(result_detail)
    
    if test_passed:
        results["passed"] += 1
    else:
        results["failed"] += 1


# ----------------- Summary ----------------- #

print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"Total Tests: {results['total']}")
print(f"Passed: {results['passed']} ✓")
print(f"Failed: {results['failed']} ✗")
print(f"Success Rate: {(results['passed'] / results['total'] * 100):.1f}%")
print()

print("\nDetailed Results:")
print("-" * 80)
for detail in results["details"]:
    status = "✓ PASS" if detail["passed"] else "✗ FAIL"
    print(f"{status} | {detail['test_name']} | {detail['time']:.2f}s")

print("\n" + "=" * 80)
print("TEST SUITE COMPLETE")
print("=" * 80)

with open("gemma3_4b_test_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to: gemma3_4b_test_results.json")
