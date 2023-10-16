import os
import json
import creating_vector_store

def lazy_list_subfolders(directory):
    software_list = sorted([f.name for f in os.scandir(directory) if f.is_dir()])
    software_version_dict = {}
    for software in software_list:

        result = creating_vector_store.pinecone_create_vector_store(software)
        print(result)

        if '~' not in software:
            versions = ['Not Available']
        else:
            versions = [software.split('~')[1]]
            software = software.split('~')[0]

        if software in software_version_dict.keys():
            software_version_dict[software] = software_version_dict[software] + versions

            if 'Not Available' in software_version_dict[software]:
                versions = software_version_dict[software]
                versions.remove('Not Available')
                software_version_dict[software] = versions

        else:
            software_version_dict[software] = versions
        
    pretty_json_string = json.dumps(software_version_dict, indent=4)
    print(pretty_json_string)
    print(len(software_version_dict.keys()))
    print(len(software_list))

    with open("vector_data.json", "w") as json_file:
        json.dump(software_version_dict, json_file)

target_directory = "/home/yuvraj/projects/docai/test"

if os.path.isdir(target_directory):
    lazy_list_subfolders(target_directory)
else:
    print("Invalid directory path. Please provide a valid directory.")
