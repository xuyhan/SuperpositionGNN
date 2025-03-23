# Test file to see if csd3 system is working for me

import os
import json

print("Hello World!")

folder = "Test"
file_name = "Test1.json"
# Combine folder and file name into a full path.
file_path = os.path.join(folder, file_name)

output_str = {
    "Hello": "World"
}

with open(file_path, "w") as f:
    json.dump(output_str, f, indent=4)
    
print(f"\nExperiment results saved to {file_path}")