import base64
import sys

file_path = "C:/Users/kevin/repos/TOOLS/mcp-crawl4ai-rag/AI DOC LIBRARY/google_ADK_Agent_Development_Kit.md"

with open(file_path, 'rb') as f:
    content = f.read()
    
encoded = base64.b64encode(content).decode('utf-8')
print(f"Encoded {len(encoded)} bytes")

# Write to file
with open('google_adk_encoded.txt', 'w') as f:
    f.write(encoded)
    
print("Encoded content saved to google_adk_encoded.txt")
