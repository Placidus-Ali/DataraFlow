import os
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

server = os.getenv("SQL_SERVER_NAME")
database = os.getenv("SQL_SERVER_DATABASE")
username = os.getenv("SQL_READONLY_USERNAME")
password = os.getenv("SQL_READONLY_PASSWORD")

print(f"Connecting to database {database} on server {server}...")