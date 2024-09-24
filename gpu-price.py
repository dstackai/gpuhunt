import gpuhunt
import json

# Query GPU instances from gpuhunt
items = gpuhunt.query(gpu_name=['MI300X', 'A100', 'H100', 'H200'])

# Function to convert gpuhunt CatalogItems to the desired JSON format
def convert_to_json(items):
    gpu_data = []
    for item in items:
        # Extract relevant fields and append them to the list in the correct format
        gpu_data.append({
            "gpuModel": item.gpu_name,
            "price": f"${item.price:.2f}",  # Format price to 2 decimal places
            "provider": item.provider
        })
    return gpu_data

# Convert the items to JSON format
gpu_json_data = convert_to_json(items)

# Write the JSON data to a file
with open('gpu_prices.json', 'w') as json_file:
    json.dump(gpu_json_data, json_file, indent=4)

print("GPU data has been written to gpu_prices.json")
