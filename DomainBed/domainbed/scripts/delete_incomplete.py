import os
import json
import shutil


def find_and_delete_directories(base_path):
    # Walk through the directory
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            result_jsonl_path = os.path.join(dir_path, 'results.jsonl')
            should_delete = False
            # Check if result.jsonl exists
            if os.path.isfile(result_jsonl_path):
                try:
                    with open(result_jsonl_path, 'r') as file:
                        # Parse each line as a JSON object and find the max 'step' value
                        max_step = 0
                        for line in file:
                            data = json.loads(line)
                            if 'step' in data:
                                max_step = max(max_step, data['step'])

                        # Check if max_step is less than 5000 and delete the directory
                        if max_step < 5000:
                            pass
                            # should_delete = True
                            # del_msg = f"Deleted '{dir_path}' because max step is {max_step}"
                except json.JSONDecodeError:
                    del_msg = f"Error decoding JSON from {result_jsonl_path}"
                    should_delete = True
                except Exception as e:
                    del_msg = f"Error processing directory {dir_path}: {str(e)}"
                    should_delete = True

            else:
                should_delete = True
                del_msg = f"Deleted '{dir_path}' because results.jsonl does not exist"


            if should_delete:
                try:
                    shutil.rmtree(dir_path)
                    print(del_msg)
                    # print(f"Deleted '{dir_path}'")
                except Exception as e:
                    print(f"Error deleting directory {dir_path}: {str(e)}")


# Set the base path to the directory containing all subdirectories
base_path = './domainbed/results_vits/'
# base_path = '../results_vits/'

# Call the function
find_and_delete_directories(base_path)
