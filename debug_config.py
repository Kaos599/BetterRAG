import sys
import yaml

def load_and_print_config():
    """Load and print the config.yaml file to debug issues"""
    try:
        with open("config.yaml", 'r') as file:
            data = file.read()
            print("Raw config file content:")
            print(data)
            print("\n" + "-" * 50 + "\n")
            
            try:
                config = yaml.safe_load(data)
                print("Parsed YAML:")
                print(config)
                
                print("\n" + "-" * 50 + "\n")
                print("Model config:", config.get('model', {}))
                print("\nAzure OpenAI config:", config.get('model', {}).get('azure_openai', {}))
            except Exception as e:
                print(f"Error parsing YAML: {e}")
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    load_and_print_config() 