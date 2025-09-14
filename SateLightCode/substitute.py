import os
import json
import subprocess
import shutil
import hashlib
import gzip
import sys
import time
import write_diff

staged_layer = "workspace/tmp/staged_layer"
image_dir = "workspace/tmp/extract_dir"
workspace = "workspace"
backup_dir = "workspace/backup"
backup_diff_dir = "workspace/backup/diff_content"
image_name = "testapp"
diff_content = 'workspace/tmp/staged_diff_content/diff_content'
staged_diff_content = 'workspace/tmp/staged_diff_content'

def get_sha256(filename):
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_gzip_layer_diff_id(gzip_filename):
    """Calculate the SHA256 (i.e., diff_id) of the file after decompressing the gzip compression layer"""
    diff_id_hash = hashlib.sha256()
    with gzip.open(gzip_filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            diff_id_hash.update(chunk)
    return f"{diff_id_hash.hexdigest()}"

def oci_to_docker(oci_image, docker_image, tag = 'latest'):
    subprocess.run(['skopeo', 'copy', 'oci-archive:workspace/' + oci_image, f'docker-archive:workspace/{docker_image}:{image_name}:{tag}'])
    
def remove_dir_children(dir_path_list):
    for dir_path in dir_path_list:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            os.makedirs(dir_path, exist_ok=True)
            
def remove_file_or_dir(path_list):
    for path in path_list:
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

def ensure_parent_dir(path):
    parent_dir = os.path.dirname(path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

def log_line_to_edit_path(log_line):
    change_list = log_line.strip().split(" ")
    edit_path = []
    for i in range(2, len(change_list), 2):
        if change_list[i] == 'R':
            edit_path.append(['R', int(change_list[i + 1])])
        elif change_list[i] == 'D':
            edit_path.append(['D', int(change_list[i + 1])])
        elif change_list[i] == 'I':
            edit_path.append(['I', int(change_list[i + 1])])
    return edit_path


def invert_diff_file(diff_file, target_path, invert_change_path = False): # invert_way: 'all' or 'inc'
    with open(diff_file, 'r') as f:
        lines = f.readlines()
    with open(target_path, 'w') as f:
        for line in lines:
            if line.startswith('+'):
                f.write('-' + line[1:])
            elif line.startswith('-'):
                f.write('+' + line[1:])
            elif line.startswith('~t') or line.startswith('~b'):
                if invert_change_path:
                    first_pos = line.find(' ')
                    second_pos = line.find(' ', first_pos + 1)
                    new_line = line[:second_pos]
                    for char in line[second_pos:]:
                        if char == 'I':
                            new_line += 'D'
                        elif char == 'D':
                            new_line += 'I'
                        else:
                            new_line += char
                    f.write(new_line)
                else:
                    f.write(line[0] + line[2:])
            else:
                f.write(line)

# upgrage image layer with file and back up 
def substitute_layer(diff_content = diff_content, reverse_diff = f"{backup_diff_dir}/diff.txt", backup_way = 'none'):
    with open(diff_content + '/diff.txt', 'r') as f:
        lines = f.readlines()
    if backup_way == "patch":
        with open(reverse_diff, 'r') as f:
            reverse_lines = f.readlines()
        
    for index,line in enumerate(lines):
        change_list = line.split(" ")
        file_name = change_list[1].strip()
        file_path = staged_layer + '/app/' + file_name
        print(line)
        if line.startswith('-f'):
            if backup_way == 'none':
                os.remove(file_path)
            else:
                ensure_parent_dir(f"{backup_diff_dir}/{file_name}")
                os.rename(file_path, f"{backup_diff_dir}/{file_name}")
        elif line.startswith('-d'):
            if backup_way == 'none':
                shutil.rmtree(file_path)
            else:
                ensure_parent_dir(f"{backup_diff_dir}/{file_name}")
                os.rename(file_path, f"{backup_diff_dir}/{file_name}")
        elif line.startswith('+f'):
            os.rename(diff_content + '/' + file_name, file_path)
        elif line.startswith('+d'):
            os.rename(diff_content + '/' + file_name, file_path)
        elif line.startswith('~ '):
            if backup_way == 'none':
                os.remove(file_path)
                os.rename(diff_content + '/' + file_name, file_path)
            else:
                ensure_parent_dir(f"{backup_diff_dir}/{file_name}")
                os.rename(file_path, f"{backup_diff_dir}/{file_name}")
                os.rename(diff_content + '/' + file_name, file_path)
        elif line.startswith('~t') or line.startswith('~b'):
            is_text = True if line.startswith('~t') else False
            if backup_way == 'none':
                update_file(file_path, diff_content + '/' + file_name, change_list, is_text = is_text)
            elif backup_way == 'patch':
                update_file(file_path, diff_content + '/' + file_name, change_list, is_text = is_text)
                reverse_edit_path = log_line_to_edit_path(reverse_lines[index])
                ensure_parent_dir(f"{backup_diff_dir}/{file_name}")
                write_diff.extract_diff(file_path,f"f{backup_diff_dir}/{file_name}", reverse_edit_path, is_text)
            elif backup_way == 'file':
                ensure_parent_dir(f"{backup_diff_dir}/{file_name}")
                shutil.copy2(file_path, f"{backup_diff_dir}/{file_name}")
                update_file(file_path, diff_content + '/' + file_name, change_list, is_text = is_text)
        else:
            print(f"Unrecognized line: {line}")


def update_file(target_path, extracted_lines, change_list, is_text):
    if is_text:
        with open(extracted_lines, 'r') as file:
            extracted_lines = file.readlines()
        file = open(target_path, 'r+')
        target_file = file.readlines()
    else:
        with open(extracted_lines, 'rb') as file:
            extracted_lines = file.read()
        file = open(target_path, 'rb+')
        target_file = bytearray(file.read())

    location_target_file = 0
    location_extracted_lines = 0 # location of diff file

    for i in range(2, len(change_list), 2):
        if change_list[i] == 'R':
            location_target_file += int(change_list[i + 1])
        elif change_list[i] == 'D':
            del target_file[location_target_file: location_target_file + int(change_list[i + 1])]
        elif change_list[i] == 'I':
            target_file[location_target_file:location_target_file] = extracted_lines[location_extracted_lines:location_extracted_lines + int(change_list[i + 1])]
            location_target_file += int(change_list[i + 1])
            location_extracted_lines += int(change_list[i + 1])

    file.seek(0)
    if is_text:
        file.writelines(target_file)
    else:
        file.write(target_file)
    file.truncate()
    file.close()
    print(f"Successfully completed the modification file: {target_path}")

def upgrade_image(compressed_diff_content = 'workspace/diff_content.tar.gz',
                  original_image = "workspace/testappoci.tar",
                  upgraded_image = 'workspace/new_image.tar',
                  substitution_way = 'inc',
                  backup_way = 'layer'):
    '''
    legal combination
    layer none
    all none
    inc layer, file, patch, none
    '''
    # unpack diff_content and image
    start_unpack = time.time()
    subprocess.run(['tar', '-xzf', compressed_diff_content, '-C', staged_diff_content])
    subprocess.run(['tar', '-xf', original_image , '-C', image_dir])
    print(f"Image and diff_content extract time: {(time.time() - start_unpack) * 1000}")

    # read index.json and get manifest path
    index_handle = open(image_dir + '/index.json', 'r+')
    index_data = json.load(index_handle)
    manifests_digest = index_data['manifests'][0]['digest']
    manifest_path = image_dir + f"/blobs/sha256/{manifests_digest[7:]}"  # delete prefix "sha256:"
    print(f"Manifests digest from index.json, path is: {manifest_path}")


    # extact top layer's digest
    manifest_handle = open(manifest_path, 'r+')
    manifest_data = json.load(manifest_handle)
    top_layer_hash = manifest_data['layers'][-1]['digest'][7:]
    config_hash = manifest_data['config']['digest'][7:]

    # substitute and choose whether to back up
    if substitution_way == 'layer':
        # sustitute layer
        new_layer_hash = os.listdir(diff_content)[0]
        os.rename(f"{diff_content}/{new_layer_hash}", f"{image_dir}/blobs/sha256/{new_layer_hash}")
        new_layer_diff_id = get_gzip_layer_diff_id(f"{image_dir}/blobs/sha256/{new_layer_hash}")
        print(f"New layer_diff_id: {new_layer_diff_id}")
    else:
        # extract top layer
        subprocess.run(['tar', '-xf', f"{image_dir}/blobs/sha256/{top_layer_hash}", '-C', staged_layer])

        if substitution_way == 'all':
            os.remove(f"{image_dir}/blobs/sha256/{top_layer_hash}")
            substitute_layer(backup_way = 'none')
        elif substitution_way == 'inc' and backup_way == 'layer':
            # back up layer
            start = time.time()
            os.rename(f"{image_dir}/blobs/sha256/{top_layer_hash}", f"{backup_diff_dir}/{top_layer_hash}")
            backup_time = time.time() - start
            substitute_layer(backup_way = 'none')
        elif substitution_way == 'inc' and backup_way == 'file':
            start = time.time()
            invert_diff_file(f"{diff_content}/diff.txt", f"{backup_diff_dir}/diff.txt", invert_change_path = False)
            os.remove(f"{image_dir}/blobs/sha256/{top_layer_hash}")
            substitute_layer(backup_way = backup_way)
            backup_time = time.time() - start
        elif substitution_way == 'inc' and backup_way == 'patch':
            start = time.time()
            invert_diff_file(f"{diff_content}/diff.txt", f"{backup_diff_dir}/diff.txt", invert_change_path = True)
            os.remove(f"{image_dir}/blobs/sha256/{top_layer_hash}")
            substitute_layer(backup_way = backup_way)
            backup_time = time.time() - start
        elif substitution_way == 'inc' and backup_way == 'none':
            os.remove(f"{image_dir}/blobs/sha256/{top_layer_hash}")
            substitute_layer(backup_way = backup_way)


        # repack new layer
        start_repack = time.time()
        subprocess.run(['tar', '-czf', 'workspace/tmp/new_layer.tar', '-C', staged_layer, '.'])
        print(f"Compress new layer time: {(time.time() - start_repack) * 1000}")
        start_newhash = time.time()
        new_layer_hash = get_sha256("workspace/tmp/new_layer.tar")
        print(f"Get new layer hash time: {(time.time() - start_newhash) * 1000}")
        start_test = time.time()
        new_layer_diff_id = get_gzip_layer_diff_id("workspace/tmp/new_layer.tar")
        print(f"Get new diff_id time: {(time.time() - start_test) * 1000}")
        print(f"new layer_diff_id: {new_layer_diff_id}")

        # move new layer to image
        os.rename("workspace/tmp/new_layer.tar", image_dir + "/blobs/sha256/" + new_layer_hash)
        print(f"old_layer_hash: {top_layer_hash} -> new_layer_hash: {new_layer_hash}")

    # Write config: substitute top layer diff_id
    config_path = image_dir + '/blobs/sha256/' + config_hash
    with open(config_path, 'r+', encoding = 'utf-8') as config_handle:
        config_data = json.load(config_handle)
        config_data['rootfs']['diff_ids'][-1] = "sha256:" + new_layer_diff_id
        config_handle.seek(0)
        json.dump(config_data, config_handle, ensure_ascii=False)
        config_handle.truncate()
    new_config_hash = get_sha256(config_path)
    os.rename(config_path, image_dir + '/blobs/sha256/' + new_config_hash)
    print(f"config_hash: {config_hash} -> new_config_hash: {new_config_hash}")

    # Write Manifest: substitute top_layer_id
    manifest_data['layers'][-1]['digest'] = "sha256:" + new_layer_hash
    manifest_data['layers'][-1]['size'] = os.path.getsize(image_dir + '/blobs/sha256/' + new_layer_hash)
    manifest_data['config']['digest'] = "sha256:" + new_config_hash
    manifest_data['config']['size'] = os.path.getsize(image_dir + '/blobs/sha256/' + new_config_hash)
    manifest_handle.seek(0)
    json.dump(manifest_data, manifest_handle, ensure_ascii=False)
    manifest_handle.truncate()
    manifest_handle.close()
    new_manifest_hash = get_sha256(manifest_path)
    os.rename(manifest_path, image_dir + '/blobs/sha256/' + new_manifest_hash)
    print(f"manigest_hash: {config_hash} -> new_manifest_hash: {new_manifest_hash}")

    # Write index.json: substitute config_id
    index_data['manifests'][0]['digest'] = "sha256:" + new_manifest_hash
    index_data['manifests'][0]['size'] = os.path.getsize(image_dir + '/blobs/sha256/' +new_manifest_hash)
    index_handle.seek(0)
    json.dump(index_data, index_handle, ensure_ascii=False)
    index_handle.truncate()
    index_handle.close()

    # package new image
    start_package = time.time()
    subprocess.run(['tar', '-cf', upgraded_image, '-C', image_dir, '.'])
    print(f"Tar new layer time: {(time.time() - start_package) * 1000}")
    return backup_time if backup_way != 'none' else 0
    

# workspace structure
# workspace
#     ├── tmp
#     │   ├── extract_dir
#     │   ├── staged_layer
#     │   ├── staged_diff_content
#     │   │   ├── diff_content
#     ├── backup
#     │   ├── diff_content

layer_time = {}
file_time = {}
if __name__ == "__main__":
    # Get the substitution method
    substitution_way = sys.argv[1] # 'layer' or 'all' or 'inc'
    backup_way = sys.argv[2]      # 'layer' or 'file'  or 'patch' or 'none'

    for _ in range(1):
        # To restore
        remove_dir_children([image_dir, staged_layer, backup_diff_dir, staged_diff_content])
        remove_file_or_dir(["workspace/new_image.tar", "workspace/new_docker_image.tar",
                            "workspace/new_restored_image.tar", "workspace/new_docker_restored_image.tar"])

        all_start_time = time.time() # time
        backup_time = upgrade_image(compressed_diff_content = 'workspace/diff_content.tar.gz',
                                      original_image = "workspace/testappoci.tar",
                                      upgraded_image = "workspace/new_image.tar",
                                      substitution_way = substitution_way,
                                      backup_way = backup_way)
        upgrade_and_backup_time  = (time.time() - all_start_time) * 1000 # time

        # oci_to_docker(oci_image = "new_image.tar", docker_image = "new_docker_image.tar", tag = 'test') # To test middile stage

        # To write time
        if backup_way != 'none':
            with open(f'{substitution_way}_{backup_way}.txt', 'a') as f:
                f.write(f"{upgrade_and_backup_time}\n")
        if backup_way == 'layer':
            with open(f'{substitution_way}_{backup_way}.txt', 'a') as f:
                f.write(f"{backup_time * 1000}\n")
        else:
            start = time.time()
            subprocess.run(['tar', '-czf', f"{backup_dir}/diff_content.tar.gz",'-C', backup_dir, 'diff_content'])
            backup_time += time.time() - start
            with open(f'{substitution_way}_{backup_way}.txt', 'a') as f:
                f.write(f"{backup_time * 1000}\n")