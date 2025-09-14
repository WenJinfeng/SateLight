import re
import random

# Patterns for JavaScript
blank_line_p = r"^\s*$"
comment_line_p = r"^\s*//.*$"  # JavaScript single-line comment
# For completeness, one-line block comment: r"^\s*/\*.*?\*/\s*$"
# We'll primarily use the single-line comment pattern for deletion consistency with original.
console_log_line_p = r"^\s*console\.log\(.*\)(;)?\s*$"  # JavaScript console.log, optional semicolon

# Target line length for generated padded lines (consistent with original script's hardcoding)
TARGET_LINE_LENGTH = 64
JS_INDENT_SPACES = 4  # Typical JS indent size


def get_num_can_be_deleted_lines(file_path):
    total_lines = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except Exception:
        # print(f"Warning: Could not read file {file_path} in get_num_can_be_deleted_lines.")
        return 0

    for line in lines:
        if re.match(blank_line_p, line) or \
                re.match(comment_line_p, line) or \
                re.match(console_log_line_p, line):
            total_lines += 1
    return total_lines


def get_locations_of_lines_ending_with_open_brace(file_path):
    locations = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except Exception:
        # print(f"Warning: Could not read file {file_path} in get_locations_of_lines_ending_with_open_brace.")
        return []

    for i, line in enumerate(lines):
        if line.strip().endswith("{"):  # Check for lines ending with '{' after stripping whitespace
            locations.append(i)
    return locations


def get_add_list_js_target(total_num, next_num_start, space_num):
    add_list = []
    next_num = next_num_start
    for _ in range(total_num):
        choice = random.randint(0, 2)
        indent_str = ' ' * space_num

        random_chars = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(TARGET_LINE_LENGTH))

        if choice == 0:  # Insert a JS comment
            # Example: // 0 abcdefgh...
            core_content = f"// {next_num} "
            padding_len = TARGET_LINE_LENGTH - (len(indent_str) + len(core_content) + 1)  # +0 for newline
            padding = random_chars[:max(0, padding_len)]
            add_list.append(indent_str + core_content + padding + '\n')
            next_num += 1
        elif choice == 1:  # Insert a JS console.log statement
            # Example: console.log("marker_0"); // abcdefgh... line marker
            js_code = f'console.log("marker_{next_num}");'
            comment_intro = " // "
            comment_suffix_text = " line marker"

            current_line_len_before_padding = len(indent_str) + len(js_code) + len(comment_intro) + len(
                comment_suffix_text) + 1  # +0 for newline
            padding_len = TARGET_LINE_LENGTH - current_line_len_before_padding
            padding = random_chars[:max(0, padding_len)]

            add_list.append(indent_str + js_code + comment_intro + padding + comment_suffix_text + '\n')
            next_num += 1
        elif choice == 2:  # Insert an unused JS variable
            # Example: let unused_variable0 = "val_0"; // abcdefgh... unused
            js_code = f'let unused_variable{next_num} = "val_{next_num}";'  # Using let
            comment_intro = " // "
            comment_suffix_text = " unused"

            current_line_len_before_padding = len(indent_str) + len(js_code) + len(comment_intro) + len(
                comment_suffix_text) + 1  # +0 for newline
            padding_len = TARGET_LINE_LENGTH - current_line_len_before_padding
            padding = random_chars[:max(0, padding_len)]

            add_list.append(indent_str + js_code + comment_intro + padding + comment_suffix_text + '\n')
            next_num += 1
    return add_list


def insert_harmless_code_js_target(file_path, num_insert):
    if num_insert <= 0:
        return
    locations = get_locations_of_lines_ending_with_open_brace(file_path)

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except Exception:
        # print(f"Warning: Could not read file {file_path} for insertion.")
        return

    next_num_marker = 0
    if not lines:  # Handle empty file
        lines = ['\n']  # Start with a newline to insert at the beginning

    if not locations:  # No lines ending with '{', insert at the beginning (or after first line if empty)
        # Calculate indent based on first line if exists, else 0
        base_indent_chars = 0
        new_lines_to_add = get_add_list_js_target(num_insert, next_num_marker, base_indent_chars)
        lines[0:0] = new_lines_to_add  # Insert at the very beginning
    else:
        # Distribute insertions similar to original logic
        # Iterate from the last found '{' location backwards
        current_inserts_left = num_insert

        # Process locations from the end of the file (last '{') to the beginning
        for i in range(len(locations) - 1, -1, -1):
            if current_inserts_left <= 0:
                break

            loc_index = locations[i]
            line_before_insert = lines[loc_index]
            # Calculate indent: indent of the line with "{" + standard JS indent
            indent_of_brace_line = len(line_before_insert) - len(line_before_insert.lstrip())
            base_indent_chars = indent_of_brace_line + JS_INDENT_SPACES

            if i == 0:  # If this is the earliest '{' in the file (last location in this loop iteration)
                num_to_add_here = current_inserts_left  # Add all remaining
            else:
                num_to_add_here = random.randint(1, current_inserts_left)

            new_lines_content = get_add_list_js_target(num_to_add_here, next_num_marker, base_indent_chars)

            # Insert after the line with '{'
            # Ensure loc_index + app_before is a valid insertion point
            if loc_index + 1 > len(lines):
                lines.extend(new_lines_content)
            else:
                lines[loc_index + 1: loc_index + 1] = new_lines_content

            next_num_marker += num_to_add_here
            current_inserts_left -= num_to_add_here

        # If there are still lines to insert (e.g., num_insert was > 0 but locations was empty initially, handled above)
        # Or if distribution didn't exhaust (unlikely with above logic if locations existed)
        # This case should be rare if locations existed and num_insert > 0 due to the "all remaining" logic.
        if current_inserts_left > 0:  # Fallback: add to the start if any remain
            new_lines_content = get_add_list_js_target(current_inserts_left, next_num_marker, 0)  # 0 indent
            lines[0:0] = new_lines_content

    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(lines)
    except Exception:
        # print(f"Warning: Could not write to file {file_path} after insertion.")
        pass


def delete_lines_js_target(file_path, num_deletes_target):
    if num_deletes_target <= 0:
        return
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except Exception:
        # print(f"Warning: Could not read file {file_path} for deletion.")
        return

    if not lines:
        return

    deletes_made = 0
    current_line_index = 0

    # Need a copy to check original previous lines for context if a line is modified
    original_lines_for_context = list(lines)

    while deletes_made < num_deletes_target and current_line_index < len(lines):
        line_was_modified_or_deleted = False
        current_line_content = lines[current_line_index]

        if re.match(blank_line_p, current_line_content) or re.match(comment_line_p, current_line_content):
            lines.pop(current_line_index)
            deletes_made += 1
            line_was_modified_or_deleted = True
            # Do not increment current_line_index, as next line is now at this index
        elif re.match(console_log_line_p, current_line_content):
            # Check if the original previous line (before any modifications in this loop) ended with '{'
            if current_line_index > 0 and original_lines_for_context[current_line_index - 1].strip().endswith("{"):
                indent_of_prev_line = len(original_lines_for_context[current_line_index - 1]) - \
                                      len(original_lines_for_context[current_line_index - 1].lstrip())
                placeholder_indent = indent_of_prev_line + JS_INDENT_SPACES
                lines[current_line_index] = ' ' * placeholder_indent + '// pass_placeholder;\n'
                current_line_index += 1  # Line was modified, not popped, so move to next
            else:
                lines.pop(current_line_index)
                # Do not increment current_line_index
            deletes_made += 1
            line_was_modified_or_deleted = True  # Even if replaced, it counts as a "delete" operation for the quota

        if not line_was_modified_or_deleted:
            current_line_index += 1

    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(lines)
    except Exception:
        # print(f"Warning: Could not write to file {file_path} after deletion.")
        pass


def file_change_js_target(file_path, num_changes):
    half_num_changes = num_changes // 2
    num_can_delete = get_num_can_be_deleted_lines(file_path)

    lines_to_delete = min(num_can_delete, half_num_changes)

    # print(f"JS Target: File {file_path}")
    # print(f"JS Target: Original deletable lines: {num_can_delete}")
    # print(f"JS Target: Will delete: {lines_to_delete}")

    if lines_to_delete > 0:
        delete_lines_js_target(file_path, lines_to_delete)

    lines_to_insert = num_changes - lines_to_delete
    # print(f"JS Target: Will insert: {lines_to_insert}")

    if lines_to_insert > 0:
        insert_harmless_code_js_target(file_path, lines_to_insert)

    return lines_to_delete, lines_to_insert


if __name__ == "__main__":
    # Example usage:
    # Create a dummy test_js_file.js
    # with open("test_js_file.js", "w", encoding="utf-8") as f:
    #     f.write("function hello() {\n")
    #     f.write("    console.log('Hello, world!'); // A console log\n")
    #     f.write("    // This is a comment\n")
    #     f.write("\n") # Blank line
    #     f.write("    let x = 10;\n")
    #     f.write("}\n")
    #     f.write("console.log('End of script');\n")

    # print("Initial content of test_js_file.js:")
    # with open("test_js_file.js", "r", encoding="utf-8") as f:
    #     print(f.read())

    # num_deletable = get_num_can_be_deleted_lines("test_js_file.js")
    # print(f"\nNumber of deletable lines initially: {num_deletable}")

    # locations = get_locations_of_lines_ending_with_open_brace("test_js_file.js")
    # print(f"Locations of lines ending with '{{': {locations}")

    # print("\nRunning file_change_js_target with 10 changes...")
    # deleted, inserted = file_change_js_target("test_js_file.js", 10)
    # print(f"Lines deleted: {deleted}, Lines inserted: {inserted}")

    # print("\nFinal content of test_js_file.js:")
    # with open("test_js_file.js", "r", encoding="utf-8") as f:
    #     print(f.read())

    # Example of just inserting:
    # insert_harmless_code_js_target("test_js_file.js", 50316 3F 128)
    # print("\nContent after inserting 50316 3F 128 harmless lines:")
    # with open("test_js_file.js", "r", encoding="utf-8") as f:
    #     print(f.read())
    pass  # Add a pass for an empty main block if tests are commented out