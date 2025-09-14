import os
import json
import subprocess
import shutil
import hashlib
import gzip

staged_layer = "workspace/tmp/staged_layer"
image_dir = "workspace/tmp/extract_dir"
workspace = "workspace"
backup_dir = "workspace/backup"
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


def update_file(target_path, extracted_lines, change_list, is_text):
    if is_text:
        with open(extracted_lines, 'r') as file:
            extracted_lines = file.readlines()
        with open(target_path, 'r+') as file:
            target_file = file.readlines()
            location_target_file = 0
            location_extracted_lines = 0 # location of diff file

            for i in range(2, len(change_list), 2):
                if change_list[i] == 'R':
                    location_target_file += int(change_list[i + 1])
                elif change_list[i] == 'D':
                    del target_file[location_target_file: location_target_file + int(change_list[i + 1])]
                elif change_list[i] == 'I':
                    for j in range(location_target_file, location_target_file + int(change_list[i + 1])):
                        target_file.insert(j, extracted_lines[location_extracted_lines])
                        location_extracted_lines += 1
                    location_target_file += int(change_list[i + 1])
            file.seek(0)
            file.writelines(target_file)
            file.truncate()
    else:
        with open(extracted_lines, 'rb') as file:
            extracted_lines = file.read()
        with open(target_path, 'rb+') as file:
            target_file = bytearray(file.read())
            location_target_file = 0
            location_extracted_lines = 0 # location of diff file

            for i in range(2, len(change_list), 2):
                if change_list[i] == 'R':
                    location_target_file += int(change_list[i + 1])
                elif change_list[i] == 'D':
                    del target_file[location_target_file: location_target_file + int(change_list[i + 1])]
                elif change_list[i] == 'I':
                    for j in range(location_target_file, location_target_file + int(change_list[i + 1])):
                        target_file.insert(j, extracted_lines[location_extracted_lines])
                        location_extracted_lines += 1
                    location_target_file += int(change_list[i + 1])
            file.seek(0)
            file.write(target_file)
            file.truncate()
    print(f"Successfully completed the modification file: {target_path}")

# upgrade image layer and don't back up 
def substitute_layer_without_backup(diff_content = diff_content, substitution_way = 'inc'):
    with open(diff_content + '/diff.txt', 'r') as f:
        lines = f.readlines()
    for line in lines:
        change_list = line.split(" ")
        file_name = change_list[1].strip()
        file_path = staged_layer + '/app/' + file_name
        if line.startswith('-f'):
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Delete file: {file_path}")
            else:
                print(f"file {file_path} doesn't exist")
        elif line.startswith('-d'):
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f"Delete folder: {file_path}")
            else:
                print(f"folder {file_path} doesn't exist")
        elif line.startswith('+f'):
            if not os.path.isfile(file_path):
                os.rename(diff_content + '/' + file_name, file_path)
                print(f"Add file: {file_path}")
            else:
                print(f"The file already exists.")
        elif line.startswith('+d'):
            if not os.path.isdir(file_path):
                os.rename(diff_content + '/' + file_name, file_path)
                print(f"Add folder: {file_path}")
            else:
                print(f"The folder already exists.")
        elif line.startswith('~ '):
            os.remove(file_path)
            os.rename(diff_content + '/' + file_name, file_path)
            print(f"Move file{diff_content + '/' + file_name} to {file_path}")
        elif line.startswith('~t'):
            update_file(file_path, diff_content + '/' + file_name, change_list, is_text = True)
        elif line.startswith('~b'):
            update_file(file_path, diff_content + '/' + file_name, change_list, is_text = False)
        else:
            print(f"Unrecognized line: {line}")

def upgrade_image(compressed_diff_content = 'workspace/diff_content.tar.gz',
                  original_image = "workspace/testappoci.tar",
                  upgraded_image = 'workspace/new_image.tar',
                  substitution_way = 'inc',
                  back_up_way = 'layer'):

    # unpack diff_content and image
    subprocess.run(['tar', '-xzf', compressed_diff_content, '-C', staged_diff_content])
    subprocess.run(['tar', '-xf', original_image , '-C', image_dir])

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


    # extract top layer
    subprocess.run(['tar', '-xf', f"{image_dir}/blobs/sha256/{top_layer_hash}", '-C', staged_layer])
    print("Successfully extacted top layer")

    # back up layer and remove old top layer, its change is diff_content and staged_layer and backupdir
    os.remove(f"{image_dir}/blobs/sha256/{top_layer_hash}")
    substitute_layer_without_backup(substitution_way = substitution_way)


    # repack new layer
    subprocess.run(['tar', '-czf', 'workspace/tmp/new_layer.tar', '-C', staged_layer, '.'])
    new_layer_hash = get_sha256("workspace/tmp/new_layer.tar")
    new_layer_diff_id = get_gzip_layer_diff_id("workspace/tmp/new_layer.tar")
    print(f"new layer_diff_id: {new_layer_diff_id}")

    # move new layer to image
    os.rename("workspace/tmp/new_layer.tar", image_dir + "/blobs/sha256/" + new_layer_hash)
    shutil.copy(image_dir + "/blobs/sha256/" + new_layer_hash, new_layer_hash)
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
    subprocess.run(['tar', '-cf', upgraded_image, '-C', image_dir, '.'])
    

if __name__ == "__main__":
    # To restore
    remove_dir_children([image_dir, staged_layer, backup_dir, staged_diff_content])

    upgrade_image(compressed_diff_content = 'workspace/diff_content.tar.gz',
                      original_image = "workspace/appoci.tar",
                      upgraded_image = "workspace/newappoci.tar",
                      substitution_way = 'inc',
                      back_up_way = 'none')
