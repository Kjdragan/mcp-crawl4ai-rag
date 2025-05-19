import base64

# File to encode
file_path = "C:/Users/kevin/repos/TOOLS/mcp-crawl4ai-rag/AI DOC LIBRARY/google_ADK_Agent_Development_Kit.md"

# Read the file
with open(file_path, 'rb') as f:
    content = f.read()

# Encode to base64 with proper padding
encoded = base64.b64encode(content).decode('utf-8')

# Print a small sample
print(f"Encoded length: {len(encoded)}")
print(f"Sample: {encoded[:100]}")

# Save to a variable that can be used directly with the MCP tool
print("\nCOPY FROM HERE:")
print("---START---")
print(encoded)
print("---END---")
