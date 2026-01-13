import json
import requests
import yfinance as yf
from llama_cpp import Llama
import time


# ----------------- Tools / Functions (Same as before) ----------------- #

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


# ----------------- FunctionGemma Format Converter ----------------- #
def convert_to_functiongemma_format(tools_list):
    """Convert OpenAI-style tools to FunctionGemma declaration format"""
    declarations = []
    
    for tool in tools_list:
        func = tool['function']
        name = func['name']
        desc = func['description']
        params = func['parameters']
        
        # Build properties string
        properties_list = []
        for prop_name, prop_info in params.get('properties', {}).items():
            prop_type = prop_info.get('type', 'string').upper()
            prop_desc = prop_info.get('description', '')
            
            # Start property definition
            prop_parts = [
                f"description:<escape>{prop_desc}<escape>",
                f"type:<escape>{prop_type}<escape>"
            ]
            
            # Add enum if exists
            if 'enum' in prop_info:
                enum_values = ','.join([f"<escape>{v}<escape>" for v in prop_info['enum']])
                prop_parts.append(f"enum:[{enum_values}]")
            
            # Combine property parts
            properties_list.append(f"{prop_name}:{{{','.join(prop_parts)}}}")
        
        # Join all properties
        props_str = f"properties:{{{','.join(properties_list)}}}"
        
        # Build required array
        required = params.get('required', [])
        if required:
            required_str = ','.join([f"<escape>{r}<escape>" for r in required])
            required_part = f"required:[{required_str}]"
        else:
            required_part = "required:[]"
        
        # Build type specification
        type_part = "type:<escape>OBJECT<escape>"
        
        # Complete declaration with proper structure
        parameters_content = f"{props_str},{required_part},{type_part}"
        declaration = f"<start_function_declaration>declaration:{name}{{description:<escape>{desc}<escape>,parameters:{{{parameters_content}}}}}<end_function_declaration>"
        declarations.append(declaration)
    
    return ''.join(declarations)



def create_functiongemma_prompt(tools_list):
    """Create properly formatted FunctionGemma developer prompt"""
    declarations = convert_to_functiongemma_format(tools_list)
    
    # Official FunctionGemma format
    developer_prompt = f"<start_of_turn>developer\nYou are a model that can do function calling with the following functions\n{declarations}<end_of_turn>\n"
    
    return developer_prompt


# ----------------- Extraction (Updated) ----------------- #

def extract_function_call_gemma(output_text):
    """Extract function call from FunctionGemma official format"""
    try:
        output_text = output_text.strip()
        
        if "<start_function_call>" not in output_text:
            return None, None
        
        start_tag = "<start_function_call>"
        end_tag = "<end_function_call>"
        
        start_idx = output_text.find(start_tag) + len(start_tag)
        end_idx = output_text.find(end_tag)
        
        if end_idx == -1:
            function_call_text = output_text[start_idx:].strip()
        else:
            function_call_text = output_text[start_idx:end_idx].strip()
        
        # Parse "call:function_name{arg1:<escape>value1<escape>,arg2:<escape>value2<escape>}"
        if not function_call_text.startswith("call:"):
            return None, None
        
        function_call_text = function_call_text[5:]  # Remove "call:"
        
        brace_idx = function_call_text.find("{")
        if brace_idx == -1:
            function_name = function_call_text.strip()
            arguments = {}
        else:
            function_name = function_call_text[:brace_idx].strip()
            args_text = function_call_text[brace_idx+1:].strip().rstrip("}")
            
            # Parse FunctionGemma format: arg1:<escape>value1<escape>,arg2:<escape>value2<escape>
            arguments = {}
            if args_text:
                # Split by comma (but be careful with nested structures)
                import re
                # Match pattern: key:<escape>value<escape>
                pattern = r'(\w+):<escape>([^<]+)<escape>'
                matches = re.findall(pattern, args_text)
                
                for key, value in matches:
                    arguments[key] = value.strip()
        
        return function_name, arguments
    except Exception as e:
        print(f"Error extracting function call: {e}")
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
            "description": "Cook a beef burger with specified doneness",
            "parameters": {
                "type": "object",
                "properties": {
                    "cook": {
                        "type": "string",
                        "description": "How well done the burger should be",
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
            "description": "Cook fried prawn noodles with customizable seafood options including prawns and sotong (squid)",
            "parameters": {
                "type": "object",
                "properties": {
                    "prawn": {
                        "type": "string",
                        "description": "Include or exclude prawns",
                        "enum": ["with prawn", "without prawn"],
                    },
                    "sotong": {
                        "type": "string",
                        "description": "Include or exclude sotong (squid)",
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
            "description": "Make a cup of coffee with various customization options for milk, sugar, and strength",
            "parameters": {
                "type": "object",
                "properties": {
                    "types_of_coffee": {
                        "type": "string",
                        "description": "The type of coffee drink (latte, americano, cappuccino, long black, espresso)",
                    },
                    "milk": {
                        "type": "string",
                        "description": "Amount of milk",
                        "enum": ["normal", "no", "more", "less"],
                    },
                    "sugar": {
                        "type": "string",
                        "description": "Amount of sugar",
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
                        "description": "The stock ticker symbol (AAPL for Apple, GOOGL for Google, MSFT for Microsoft)",
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
            "description": "Get the current weather conditions including temperature, humidity, and description for a given city location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name (New York, Singapore, Tokyo, Paris, London)",
                    }
                },
                "required": ["city"],
            },
        },
    },
]


# ----------------- Test Cases (Same as before) ----------------- #

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
        "query": "I want a medium rare burger please",
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
        "query": "What is the capital of Singapore?",
        "expected_function": None,
        "type": "conversation"
    },
    {
        "name": "Test 9: Simple chat",
        "query": "Tell me a fun fact about coffee",
        "expected_function": None,
        "type": "conversation"
    },
]


# ----------------- Model Init ----------------- #

print("Loading FunctionGemma model...")
llm = Llama(
    model_path="./model/functiongemma-270m-it-BF16.gguf",
    # model_path="./model/gemma-3-4b-it-q4_0.gguf",
    n_ctx=8192,
    n_gpu_layers=-1,
    logits_all=False,
    vocab_only=False,
    seed=42,
    verbose=False,
)
print("Model loaded successfully!\n")


# ----------------- Run Tests ----------------- #

developer_prompt = create_functiongemma_prompt(tools_list)

results = {
    "passed": 0,
    "failed": 0,
    "total": len(test_cases),
    "details": []
}

print("=" * 80)
print("STARTING FUNCTIONGEMMA TEST SUITE (Corrected Format)")
print("=" * 80)
print()

for idx, test in enumerate(test_cases, 1):
    print(f"\n{'=' * 80}")
    print(f"{test['name']}")
    print(f"{'=' * 80}")
    print(f"Query: {test['query']}")
    print(f"Expected: {'Function call to ' + test['expected_function'] if test['expected_function'] else 'Normal conversation'}")
    print()
    
    # Build proper FunctionGemma prompt
    full_prompt = developer_prompt + f"<start_of_turn>user\n{test['query']}<end_of_turn>\n<start_of_turn>model\n"
    
    start = time.time()
    response = llm(
        full_prompt,
        max_tokens=256,
        temperature=0.1,
        stop=["<end_of_turn>", "<end_function_call>"],
        echo=False,
    )
    end = time.time()
    
    assistant_content = response["choices"][0]["text"]
    inference_time = end - start
    
    print(f"Model output: {assistant_content}")
    print(f"Time taken: {inference_time:.2f}s")
    print()
    
    function_name, arguments = extract_function_call_gemma(assistant_content)
    
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

with open("functiongemma_test_results_corrected.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to: functiongemma_test_results_corrected.json")
