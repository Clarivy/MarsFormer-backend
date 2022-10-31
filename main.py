import os
import re
from os.path import join
import uuid
import subprocess

import aiofiles as aiofiles
from fastapi import FastAPI, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_id():
    return str(uuid.uuid4())


def run_job(uid: str, usc_name: str, flame_name: str):
    cwd = './data/{}/output'.format(uid)
    helper_result = open("./data/{}/result".format(uid), 'w')
    subprocess.run(
        ["python3", "../../../zlw/helper.py", 'topo_transfer', '../usc/' + usc_name, '../flame/' + flame_name, 'output',
         '--map_path', 'map.txt'], cwd=cwd, stderr=helper_result)

    subprocess.run(
        ["python3", "../../../cal.py"], cwd=cwd)

    with open("./data/{}/output/VERSION".format(uid), 'w') as f:
        f.write("1")

    helper_result.write('\nJob {} Done.\n'.format(uid))
    helper_result.close()


@app.post("/api/upload")
async def root(flame: UploadFile, usc: UploadFile, background_tasks: BackgroundTasks):
    async def read_file(path: str, ufile: UploadFile):
        os.mkdir(path)
        async with aiofiles.open(join(path, ufile.filename), 'wb') as out:
            while content := await ufile.read(1048576):
                await out.write(content)

    uid = generate_id()
    os.mkdir(join('./data', uid))
    os.mkdir(join('./data', uid, 'output'))

    await read_file(join('./data', uid, 'flame'), flame)
    await read_file(join('./data', uid, 'usc'), usc)

    background_tasks.add_task(run_job, uid, usc.filename, flame.filename)

    return {"uid": uid}


@app.get("/api/download/{uid}")
def download(uid: uuid.UUID):
    dir_name = './data/{}'.format(str(uid))
    zip_path = '{}/{}'.format(dir_name, 'output.zip')
    if not os.path.isdir(dir_name):
        raise HTTPException(status_code=404, detail="Project not found")
    if not os.path.isfile(zip_path):
        shutil.make_archive('./data/{}/output'.format(str(uid)), 'zip', dir_name + '/output')
    return FileResponse(zip_path, filename='{}.dtt'.format(uid))


@app.get("/api/status/{uid}")
def check_status(uid: uuid.UUID):
    output_file = './data/{}/result'.format(str(uid))
    if not os.path.isfile(output_file):
        return {
            'progress': '0%',
            'done': False,
        }
    output_file = open(output_file)
    log = output_file.read()
    if re.findall('Job {} Done\\.'.format(str(uid)), log):
        return {
            'progress': '100%',
            'done': True,
        }
    progress = re.findall('Find closest:\\s*([0-9.]+)%', log)
    if not progress:
        return {
            'progress': '0%',
            'done': False,
        }
    return {
        'progress': '{}%'.format(progress[-1]),
        'done': False,
    }
