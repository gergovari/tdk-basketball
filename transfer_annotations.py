import os
import re

REFERENCE_DIR = "annotatedvideodata"
TARGET_DIR = "data"

def main():
    if not os.path.exists(REFERENCE_DIR):
        print(f"Error: {REFERENCE_DIR} does not exist.")
        return
        
    if not os.path.exists(TARGET_DIR):
        print(f"Error: {TARGET_DIR} does not exist.")
        return

    # Step 1: Collect annotations from the reference directory
    annotations = {}
    print("Scanning reference directory for annotations...")
    for root, dirs, files in os.walk(REFERENCE_DIR):
        for file in files:
            # We look for files that have been annotated with -h or -m before the extension
            match = re.match(r'^(.*?)-([hm])\.csv$', file)
            if match:
                base_name = match.group(1)
                annotation = match.group(2)
                
                rel_path = os.path.relpath(root, REFERENCE_DIR)
                
                if rel_path not in annotations:
                    annotations[rel_path] = {}
                
                annotations[rel_path][base_name] = annotation

    if not annotations:
        print("No annotations found in the reference directory.")
        return

    print("Transferring annotations to target directory...")
    # Step 2: Apply annotations to target directory
    renamed_count = 0
    for root, dirs, files in os.walk(TARGET_DIR):
        rel_path = os.path.relpath(root, TARGET_DIR)
        
        # Check if we have annotations for this specific subdirectory
        if rel_path not in annotations:
            continue
            
        dir_annotations = annotations[rel_path]
        
        for file in files:
            # Skip if already annotated with -h or -m right before extension
            if re.search(r'-[hm]\.[a-zA-Z0-9]+$', file):
                continue
                
            # Find the matching base name
            # Sort base names by length descending to match the longest base name first
            for base_name in sorted(dir_annotations.keys(), key=len, reverse=True):
                annotation = dir_annotations[base_name]
                
                if file.startswith(base_name):
                    suffix = file[len(base_name):]
                    # Ensure it's the exact base name by checking the next character
                    if suffix.startswith('.') or suffix.startswith('-'):
                        name, ext = os.path.splitext(file)
                        new_name = f"{name}-{annotation}{ext}"
                        
                        old_path = os.path.join(root, file)
                        new_path = os.path.join(root, new_name)
                        
                        print(f"Renaming: {os.path.join(rel_path, file)} -> {new_name}")
                        os.rename(old_path, new_path)
                        renamed_count += 1
                        break # Move to the next file once renamed

    print(f"Done. Total files renamed: {renamed_count}")

if __name__ == "__main__":
    main()
