import os
import random
import magic  # Retains python-magic for MIME type detection
import file_change_js  # Importing the JS-targetting version


def print_file_type(file_map):
    for value in file_map.values():
        try:
            mime = magic.from_file(value, mime=True)
            print(f"File: {value}, Type: {mime}")
        except Exception as e:
            print(f"Could not determine type for {value}: {e}")


def is_text_file(file_path):
    try:
        mime = magic.from_file(file_path, mime=True)
        return mime.startswith("text/")
    except Exception:  # Handle cases where magic might fail (e.g., broken symlink, permissions)
        # Basic fallback for common text extensions if magic fails
        text_extensions = ['.js', '.json', '.txt', '.html', '.css', '.xml', '.svg', '.md', '.ts', '.jsx', '.tsx']
        if any(file_path.lower().endswith(ext) for ext in text_extensions):
            return True
        # print(f"Warning: Could not determine if {file_path} is text using magic. Defaulting to False.")
        return False


def all_map_to_text_map(file_map):
    keys_to_delete = [key for key, value in file_map.items() if not is_text_file(value)]
    for key in keys_to_delete:
        del file_map[key]
    return file_map


def is_js_file(file_path):  # Changed from is_python_file
    return file_path.lower().endswith(".js")


def all_map_to_js_map(file_map):  # Changed from all_map_to_python_map
    keys_to_delete = [key for key, value in file_map.items() if not is_js_file(value)]
    for key in keys_to_delete:
        del file_map[key]
    return file_map


def get_all_files(directory):
    file_map = {}  # { relative_path: absolute_path }
    for root, _, files in os.walk(directory):
        for file in files:
            abs_path = os.path.join(root, file)
            if not os.path.islink(abs_path) and os.path.isfile(abs_path):  # Ensure it's a file and not a symlink
                rel_path = os.path.relpath(abs_path, directory)
                file_map[rel_path] = abs_path
    return file_map


def select_random_files(file_map_values, num_to_select_param):  # Expects list of file paths
    # Ensure num_to_select is not greater than the number of available files
    num_to_select = min(num_to_select_param, len(file_map_values))
    if num_to_select <= 0:  # If trying to select 0 or negative files
        return []
    return random.sample(list(file_map_values), num_to_select)


def get_num_of_lines_in_dir(file_map):
    num = 0
    for file_path in file_map.values():
        if is_text_file(file_path):  # Process only text files for line counting
            try:
                with open(file_path, "r", encoding='utf-8', errors='ignore') as f:  # errors='ignore' for robustness
                    line_count = sum(1 for _ in f)
                num += line_count
            except Exception:
                # print(f"Warning: Could not read {file_path} for line count.")
                pass
    return num


def get_num_of_bytes_in_dir(file_map):
    num = 0
    for file_path in file_map.values():
        try:
            num += os.path.getsize(file_path)
        except OSError:  # Handles if file is deleted between listing and stat, or other OS errors
            # print(f"Warning: Could not get size of {file_path}.")
            pass
    return num


def generate_random_segments(sum_total, count):
    if count <= 0: return []
    if sum_total == 0: return [0] * count
    if count == 1: return [sum_total]

    # Generate count-app_before unique cut points between 0 and sum_total (inclusive for 0 and sum_total)
    # This ensures segments are non-negative and sum up correctly.
    cuts = sorted(random.sample(range(sum_total + 1), count - 1))

    segments = []
    last_cut = 0
    for cut_point in cuts:
        segments.append(cut_point - last_cut)
        last_cut = cut_point
    segments.append(sum_total - last_cut)
    return segments


def generate_fair_random_segments(sum_total, count, variance=1000):
    if count <= 0: return []
    if sum_total < 0: sum_total = 0  # Ensure sum_total is not negative

    avg = sum_total // count
    segments = [avg] * count

    # Distribute remainder
    remainder = sum_total % count
    for i in range(remainder):
        segments[random.randint(0, count - 1)] += 1

    # Apply variance, ensuring segments don't go below 0
    for i in range(count):
        max_reduction = min(segments[i], variance)  # Cannot reduce more than current value or variance limit
        delta = random.randint(-max_reduction, variance)
        segments[i] += delta

    # Adjust sum back to sumTotal, ensuring segments remain non-negative
    current_sum = sum(segments)
    diff = current_sum - sum_total

    while diff != 0:
        idx_to_adjust = random.randint(0, count - 1)
        if diff > 0:  # Current sum is too high, need to decrease
            if segments[idx_to_adjust] > 0:
                segments[idx_to_adjust] -= 1
                diff -= 1
            # If segments[idx_to_adjust] is 0, loop will try another random index
        else:  # Current sum is too low (diff < 0), need to increase
            segments[idx_to_adjust] += 1
            diff += 1

    return [max(0, s) for s in segments]  # Final check for non-negativity


def count_non_zero_elements(lst):
    return sum(1 for x in lst if x != 0)


# Constants from the original script
change_rate = 0.50
length_of_line = 64  # Used for estimating total lines to change

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dir_change_js.py <directory_path>")
        sys.exit(1)

    target_directory = sys.argv[1]

    if not os.path.isdir(target_directory):
        print(f"Error: Directory '{target_directory}' not found or is not a directory.")
        sys.exit(1)

    print(f"Processing directory for JavaScript files: {target_directory}")
    file_map = get_all_files(target_directory)
    # print_file_type(file_map) # Optional: print types of all files

    total_bytes = get_num_of_bytes_in_dir(file_map)
    print(f"Total bytes in directory: {total_bytes}")

    # Calculate total number of lines to change (insertions + deletions)
    inserted_bytes_estimate = int(change_rate / (1 - change_rate) * total_bytes)
    num_total_change_lines = int(inserted_bytes_estimate / length_of_line) if length_of_line > 0 else 0
    print(f"Estimated total lines for modification (deletions + insertions): {num_total_change_lines}")

    if num_total_change_lines <= 0:
        print("Calculated lines to change is zero or less. No modifications will be performed.")
        sys.exit(0)

    js_file_map = all_map_to_js_map(dict(file_map))  # Pass a copy to avoid modifying original file_map
    num_js_files = len(js_file_map)
    print(f"Number of JavaScript files found: {num_js_files}")

    if num_js_files == 0:
        print("No JavaScript files found to modify.")
        sys.exit(0)

    # Select a random number of JS files to modify (between 20 or num_js_files if fewer, and all js_files)
    # The original logic was random.randint(20, len(file_map)) which could be an issue if len(file_map) < 20.
    # A safer approach:
    upper_bound_selection = len(js_file_map.values())
    lower_bound_selection = min(20, upper_bound_selection) if upper_bound_selection > 0 else 0

    num_files_to_select = 0
    if lower_bound_selection <= upper_bound_selection and upper_bound_selection > 0:
        num_files_to_select = random.randint(lower_bound_selection, upper_bound_selection)

    selected_js_files_paths = select_random_files(list(js_file_map.values()), num_files_to_select)

    print(f"Number of JS files selected for modification: {len(selected_js_files_paths)}")

    if not selected_js_files_paths:
        print(
            "No JS files were selected for modification (perhaps num_change_lines was too small or selection yielded zero).")
        sys.exit(0)

    # Distribute the total changes among the selected files
    segments = generate_fair_random_segments(sum_total=num_total_change_lines, count=len(selected_js_files_paths),
                                             variance=1000)
    print(
        f"Number of selected files that will actually receive changes (non-zero segment): {count_non_zero_elements(segments)}")
    # print(f"Segments (changes per file): {segments}")

    changed_files_summary = dict(zip(selected_js_files_paths, segments))
    total_deleted_sum = 0
    total_added_sum = 0

    for file_path, num_changes_for_file in changed_files_summary.items():
        if num_changes_for_file <= 0:
            # print(f"Skipping file (0 changes allocated): {file_path}")
            continue

        print(f"Processing file: {file_path}, Lines to change: {num_changes_for_file}")
        try:
            # Use the JS-targeting file_change module
            deleted_count, added_count = file_change_js.file_change_js_target(file_path, num_changes_for_file)
            total_deleted_sum += deleted_count
            total_added_sum += added_count
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            # Optionally, print traceback:
            # import traceback
            # traceback.print_exc()

    print("\n--- Summary ---")
    print(f"Total lines deleted across all processed JS files: {total_deleted_sum}")
    print(f"Total lines inserted across all processed JS files: {total_added_sum}")
    print("Processing for JavaScript files finished.")