import asyncio
import uvicorn
import aiohttp
import aiofiles
from fastapi import FastAPI, HTTPException
from dataclasses import dataclass, asdict
import json
from os import path, rename
import glob
from multiprocessing import shared_memory, Lock
from contextlib import asynccontextmanager

MODEL_DIR = "models"

pending = shared_memory.ShareableList([None] * 100)
pending_lock = Lock()

def release_shared_memory():
    global pending
    pending.shm.close()
    pending.shm.unlink()
    del pending

@asynccontextmanager
async def lifespan(app):
    #stuff before start

    yield

    # Stuff when closing app
    release_shared_memory()

app = FastAPI(lifespan=lifespan)

@dataclass
class Model:
    alias: str
    name: str
    service: str
    model_hash: str
    model_id: int
    version_id: int
    file_id: int
    filename: str
    download_url: str

async def store_model(model, data):
    async with aiofiles.open(path.join(MODEL_DIR, f"{model.filename}.modelcard"), "w") as f:
        await f.write(json.dumps(asdict(model)))

    src = path.join(MODEL_DIR, f"{model.filename}.pending")
    dst = path.join(MODEL_DIR, f"{model.filename}")
    rename(src, dst)


async def fetch_model_card(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers={'Content-type': 'application/json'}) as response:
            r_data = await response.text()

    return r_data

async def download_model(model):
    data = bytearray()
    async with aiohttp.ClientSession() as session:
        async with session.get(model.download_url) as response:
            async with aiofiles.open(path.join(MODEL_DIR, f"{model.filename}.pending"), "wb") as f:
                async for r_data in response.content.iter_chunked(10*1024*1024):
                    await f.write(r_data)

    return data

@app.get("/list")
async def list():
    db = load_models()
    return [m.alias for m in db.values()]

@app.get("/pending")
async def list():
    with pending_lock:
        return {"pending": [p for p in pending if p is not None]}

@app.get("/model/{alias}")
async def get_model(alias):
    db = load_models()
    if alias not in db:
        raise HTTPException(status_code=404, detail="No such model")
    
    async def iterfile():
        async with aiofiles.open(path.join(MODEL_DIR, db[alias].filename), "rb") as f:
            async for chunk in f.aiter_bytes():
                yield chunk

    return StreamingResponse(iterfile(), media_type="application/octet-stream")

@app.get("/download/civitai/{model_hash}/{alias}")
async def download_civitai(model_hash, alias):
    # TODO - check if hash already in db
    db = load_models()

    if alias in db:
        return {alias: "present", "model_info": db[alias]}

    with pending_lock:
        if alias in pending:
            return {alias: "pending"}

        idx = pending.index(None)
        if idx == None:
            raise HTTPException(status_code=500, detail="Too many pending downloads")

        pending[idx] = alias

    resp = await fetch_model_card(f"https://civitai.com/api/v1/model-versions/by-hash/{model_hash}")
    model_card = json.loads(resp)

    if model_card["model"]["type"] != "Checkpoint":
        raise HTTPException(status_code=422, detail="Model was of wrong type")

    if model_card["baseModel"] not in ["SD 1.5"]:
        raise HTTPException(status_code=422, detail="Base model type not implemented")
    
    name = model_card["model"]["name"]
    model_id = model_card["modelId"]
    version_id = model_card["id"]
    
    primary_file_obj = [f for f in model_card["files"] if f.get("primary") == True][0]

# Skip this since we check pickle and virus scan results below
#    if primary_file_obj["metadata"]["format"] != "SafeTensor":
#        raise HTTPException(status_code=422, detail="Model was not a safetensor")

    if primary_file_obj["pickleScanResult"] != "Success":
        raise HTTPException(status_code=422, detail="Model has failed pickle scan")

    if primary_file_obj["virusScanResult"] != "Success":
        raise HTTPException(status_code=422, detail="Model has failed virus scan")

    filename = primary_file_obj["name"]
    file_id = primary_file_obj["id"]
    file_size = primary_file_obj["sizeKB"]
    download_url = primary_file_obj["downloadUrl"]

    try:
        model = Model(alias, name, "civitai", model_hash, model_id, version_id, file_id, filename, download_url)
    except:
        #TODO - cleanup pending file
        raise HTTPException(status_code=500, detail="Failed to download model")

    data = await download_model(model)

    await store_model(model, data)
    with pending_lock:
        idx = pending.index(alias)
        pending[idx] = None

    return {alias: "present", "model_info": model} 

def load_models():
    db = {}
    model_card_files = glob.glob(f"{MODEL_DIR}/*.modelcard")
    for mcf in model_card_files:
        with open(mcf, "r") as f:
            mcf_data = f.read()
            model = json.loads(mcf_data)
            model = Model(model["alias"], model["name"], model["service"], model["model_hash"], model["model_id"], model["version_id"], model["file_id"], model["filename"], model["download_url"])
            db[model.alias] = model

    return db


async def main():
    load_models()
    config = uvicorn.Config("bandolier:app", host="192.168.1.43", port=5000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()
    release_shared_memory()

if __name__=="__main__":
    asyncio.run(main())
