from __future__ import print_function

import os
import os.path

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive.file']


def upload_files(drive, folder_name, parent_id):
    for file in os.listdir(folder_name):
        with open(folder_name + '/' + file, 'r') as f:
            file_drive = drive.CreateFile({'title': file,
                                           'parents': [{'id': parent_id}],
                                           })
            file_drive.SetContentFile(folder_name + '/' + file)
            file_drive.Upload()


def create_folder(drive, folder_name, parent_id=None):
    if parent_id is None:
        parent_id = '1YkGHMk7hPFwwjwKWhn-yf8fCJysd6e4p'
    # else:
    #     file_list = drive.ListFile(
    #         {'q': ''1_Draj6AqxNXotxv_vfPHI - QkDtyHu2qf' in parents and trashed=false'}).GetList()
    #     for file in file_list:
    #         print('{} {}'.format(file['title'], file['id']))
    #         if file['title'] == folder_name:
    #             parent_id = file['id']

    folder = drive.CreateFile({'title': folder_name,
                               'parents': [{'id': parent_id}],
                               'mimeType': 'application/vnd.google-apps.folder'})
    folder.Upload()
    return folder['id']


def main():
    gauth = GoogleAuth()
    # gauth.LoadCredentialsFile('client_secrets.json')
    # if gauth.credentials is None:
    #     # Authenticate if they're not there
    #     gauth.LocalWebserverAuth()
    # elif gauth.access_token_expired:
    #     # Refresh them if expired
    #     gauth.Refresh()
    # else:
    #     # Initialize the saved creds
    #     gauth.Authorize()
    # # Save the current credentials to a file
    # gauth.SaveCredentialsFile('client_secrets.json')
    # gauth.LoadClientConfigFile('client_secrets.json')

    drive = GoogleDrive(gauth)
    # filename = 'kek'

    mrt_path = '/Users/mkryuchkov/mrt-ds'

    # mrt_parts = ['image', 'mask']
    mrt_parts = ['mask']

    for part in mrt_parts:
        print('processing another part')
        part_folder_id = create_folder(drive, part)
        upload_files(drive, os.path.join(mrt_path, part), part_folder_id)


import shutil


def zip_files():
    lidc_path = ''
    lidc_parts = os.listdir(lidc_path)
    for part in lidc_parts:
        shutil.make_archive('{}.zip'.format(part), 'zip', os.path.join(lidc_path, part))


if __name__ == '__main__':
    main()
