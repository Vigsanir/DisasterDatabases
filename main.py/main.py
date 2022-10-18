from fastapi import FastAPI
import usgs
import catnat2
import glcdb
import disasterdb1
import idmc

app = FastAPI()

@app.get("/usgs_earthquake")
def usgs_earthquake():
   response = usgs.return_json()
   return response

@app.get("/BDcatnat")
def BDcatnat():
    response = catnat2.return_json()
    return response


@app.get("/glc")
def glc():
    response = glcdb.return_json()
    return response

@app.get("/DisasterDataBase")
def DisasterDataBase():
    response = disasterdb1.return_json()
    return response

@app.get("/IDMC_ID")
def IDMC_ID():
   response = idmc.return_json()
   return response