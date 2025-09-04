import sys
sys.path.append("../")

import requests,pandas as pd,numpy as np
from src.steps.utils import prod_path
from fastapi.testclient import TestClient
from fastapi import FastAPI


input = pd.read_csv(f"{prod_path}",index_col=0)

app = FastAPI()


client = TestClient(app)


def test_read_main(location):
    response = client.get(f"/predict/{location}")
    assert response.status_code == 200


if "__main__" == __name__:
    locs = input["Location"].unique()

    for i in locs:
        test_read_main(i)