import yaml

def get_api_token():
    """
    Reads the API token from the apitoken.yml file securely.

    Returns:
        str: The API token if found, otherwise None.

    Raises:
        FileNotFoundError: If the apitoken.yml file is not found.
        YAMLError: If there's an error parsing the YAML file.
        KeyError: If the 'token' key is not found in the YAML data.
    """

    try:
        with open("ibm_apitoken.yml", "r") as file:
            data = yaml.safe_load(file)
            return data["token"]
    except FileNotFoundError:
        print("Error: apitoken.yml file not found.")
    except (yaml.YAMLError, KeyError) as e:
        print(f"Error reading apitoken.yml: {e}")
    return None  # Indicate failure to retrieve token



