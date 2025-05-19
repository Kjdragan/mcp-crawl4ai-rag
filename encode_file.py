import base64
import sys

# Get the file path from the command line argument
file_path = sys.argv[1]

# Read the file and encode it
with open(file_path, 'rb') as f:
    content = f.read()
    
# Encode to base64
encoded = base64.b64encode(content).decode('utf-8')

# Write to output file
with open('encoded_content.txt', 'w') as out:
    out.write(encoded)
    
print(f"File encoded to encoded_content.txt")
