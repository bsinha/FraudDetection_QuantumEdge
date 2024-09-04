import json
from keyring import get_password, set_password, delete_password

SERVICE_NAME = "kaggle"  # Replace with a descriptive service name

def get_kaggle_credentials():
  """
  Retrieves Kaggle username and API key securely from keyring.

  Returns:
      tuple: A tuple containing (username, api_key) or None if not found.
  """
  print(SERVICE_NAME)
  username = get_password(SERVICE_NAME, "username")
  print(123)
  print(username)
  api_key = get_password(SERVICE_NAME, "key")
  return (username, api_key) if username and api_key else None

def set_kaggle_credentials(username, api_key):
  """
  Stores Kaggle username and API key securely in keyring.

  Args:
      username (str): Kaggle username.
      api_key (str): Kaggle API key.
  """

  set_password(SERVICE_NAME, "username", username)
  set_password(SERVICE_NAME, "key", api_key)

# Example usage
username = get_kaggle_credentials()
print(username)

username = ''
api_key = ''

if username and api_key:
  print("Credentials retrieved successfully!")
  # Use credentials for Kaggle API interaction here
else:
  print("Credentials not found. Please set them using set_kaggle_credentials(username, api_key).")
