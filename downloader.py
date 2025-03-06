import os
import json
import requests
import urllib.request

BASE_URL = 'https://api.figshare.com/v2'
ITEM_ID = '26304175'

def download_data_folder(ITEM_ID, folder):
    if not os.path.exists(folder):
            os.mkdir(folder)
    r=requests.get(BASE_URL + '/articles/' + str(ITEM_ID)) #Load the metadata as JSON
    if r.status_code != 200:
        print('Error:',r.content)
    else:
        metadata=json.loads(r.text)
        files = metadata['files']
        for file in files:
            name = file['name']
            url = file['download_url']
            file_path = folder+'/'+name
            size_Mbs = file['size'] * 1e-6 #byte to MB
            print(f"Downloading {name} to {file_path}...")
            print(f"Total File Size: {size_Mbs:.2f} MBs")
            urllib.request.urlretrieve(url, file_path)
            print("Download Complete.")

download_data_folder(ITEM_ID, 'data')