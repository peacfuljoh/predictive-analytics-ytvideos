""" Script that creates a backup of all MySQL tables and MongoDB collections. """

import os
import shutil
import datetime
import subprocess

from src.crawler.crawler.config import DB_MYSQL_CONFIG, DB_INFO, DB_MONGO_CONFIG


UPDATE_BACKUP_SQL = True
UPDATE_BACKUP_MONGO = True
UPDATE_BACKUP_CONFIG = True



date_str = datetime.datetime.now().strftime('%Y-%m-%d')

dir_drive = '/media/nuc/Bork drive'
dir_drive_subdir = os.path.join(dir_drive, 'ytpa', date_str)

dir_config_nuc = '/home/nuc/crawler_config'
dir_config_drive = os.path.join(dir_drive, 'crawler_config')




# backup MySQL tables
if UPDATE_BACKUP_SQL:
    print('\n=== Updating MySQL backup ===')

    pw = DB_MYSQL_CONFIG['password']

    for database in [DB_INFO['DB_VIDEOS_DATABASE']]:
        sql_fname = f"{database}.sql"
        cmd_dump = (f"mysqldump -h {DB_MYSQL_CONFIG['host']} -u {DB_MYSQL_CONFIG['user']} "
                    f"--password='{pw}' {database} > {sql_fname}")
        cmd_mv = f'mv {sql_fname} "{dir_drive_subdir}"'

        print('sudo ' + cmd_dump + ' && ' + cmd_mv) # <-- run this in the terminal

        # input_ = bytes(pw, encoding='utf-8') + b'\n'
        # proc = subprocess.Popen(['sudo', '-S'] + cmd_dump.split(' '),
        #                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        #                         stderr=subprocess.PIPE).communicate(input=input_)
        # print(proc)
        # os.system(cmd_mv)

# backup MongoDB databases
if UPDATE_BACKUP_MONGO:
    databases = ['DB_VIDEOS_NOSQL_DATABASE', 'DB_FEATURES_NOSQL_DATABASE', 'DB_MODELS_NOSQL_DATABASE']
    cmds = []
    for database in [DB_INFO[key] for key in databases]:
        cmd_dump = f'mongodump -h {DB_MONGO_CONFIG["host"]}:{DB_MONGO_CONFIG["port"]} -d {database}'
        cmd_mv = f'mv dump/{database} "{dir_drive_subdir}/{database}"'
        cmds += [cmd_dump, cmd_mv]
    print(' && '.join(cmds))


# backup config files
if UPDATE_BACKUP_CONFIG:
    print('\n=== Updating config files ===')

    if ('home/nuc' in dir_config_drive) or ('media/nuc' not in dir_config_drive):
        raise Exception
    else:
        if os.path.exists(dir_config_drive):
            shutil.rmtree(dir_config_drive)
        shutil.copytree(dir_config_nuc, dir_config_drive)

