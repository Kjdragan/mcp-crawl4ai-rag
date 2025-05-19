import os
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables from .env
load_dotenv()

# Get Supabase credentials from environment variables
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

print(f"Supabase URL: {supabase_url}")
print(f"Supabase key available: {'Yes' if supabase_key else 'No'}")

# Create Supabase client
supabase = create_client(supabase_url, supabase_key)

# Clear the crawled_pages table
# Using a filter that will match all records
result = supabase.table("crawled_pages").delete().neq('id', 0).execute()
print("Database cleared:", result)
