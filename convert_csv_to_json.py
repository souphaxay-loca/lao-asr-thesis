import csv
import json
import os

def convert_csv_to_json(csv_file_path, output_json_path):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    sentences = []

    try:
        # Open with UTF-8 encoding to properly handle Lao characters
        with open(csv_file_path, 'r', encoding='utf-8-sig') as csv_file:
            # Assuming your CSV has headers 'sentence_id' and 'transcription'
            reader = csv.DictReader(csv_file)

            for row in reader:
                # Clean up any whitespace
                sentence_id = row['sentence_id'].strip()
                text = row['transcription'].strip()

                if sentence_id and text:
                    sentences.append({
                        "id": sentence_id,
                        "text": text
                    })

        # Write to JSON file
        with open(output_json_path, 'w', encoding='utf-8') as json_file:
            json.dump({"sentences": sentences}, json_file, ensure_ascii=False, indent=2)

        print(f"Successfully converted {len(sentences)} sentences to {output_json_path}")
        return True

    except Exception as e:
        print(f"Error converting CSV to JSON: {e}")
        return False

# Example usage
if __name__ == "__main__":
    csv_file = "data_source/datatext.csv"  # Your current CSV file
    json_file = "data/sentences.json"      # Output JSON file
    convert_csv_to_json(csv_file, json_file)
