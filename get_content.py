import time
import pandas as pd
import multiprocessing
import requests
from requests.auth import HTTPBasicAuth
from json import loads
from tqdm import tqdm
from itertools import product
import kblab 
import time
kblab.VERIFY_CA=False


def get_content(dark_id, backoff_factor=0.1):
    with open('Exjobb/api_cred', 'r') as file:
        pw = file.read().replace('\n', '')

    for i in range(5):
        backoff_time = backoff_factor * (2 ** i)

        content_structure = requests.get(
            f"https://betalab.kb.se/{dark_id}/content.json", auth=HTTPBasicAuth("demo", pw), verify=False
        )

        if content_structure.status_code == 200:
            content_json = loads(content_structure.text)
            content_json = [x["content"] for x in content_json]
            data = {
                "content": " ".join(content_json),
                "dark_id": dark_id
            }
            print(f"{dark_id} done")
            return data

        else:
            print(f"{dark_id} failed.")

        time.sleep(backoff_time)



if __name__ == "__main__":
    start = time.time()
    df = pd.read_feather("data/all_metadata.feather")
    pool = multiprocessing.Pool()

    content = pool.starmap(
        get_content,
        tqdm(list(product(list(df["dark_id"])))),
        chunksize=20,
    )
    pool.close()

    res = {"content":[], "dark_id":[]}
    for x in content:
        res["content"].append(x["content"])
        res["dark_id"].append(x["dark_id"])
    
    df2 = pd.DataFrame(res)

    df2.to_feather("data/df_justcontent.feather")

    #df_content = df_content.reset_index(drop=False)

    df_content = pd.merge(df2, df[["dark_id", "title", "created"]], how="left")
    #df_content = df_content.rename(columns={"created": "date"})
    df_content.to_feather("data/df_content.feather")
    end = time.time()
    print(end-start)

    """
    df = pd.read_feather("data/sampled_editions.feather")

    with open('Exjobb/api_cred', 'r') as file:
        pw = file.read().replace('\n', '')

    print(pw)

    backoff_factor = 0.02
    pool = multiprocessing.Pool()

    df_list = pool.starmap(
        join_content_structure,
        tqdm(list(product(list(df["dark_id"]), [backoff_factor]))),
        chunksize=20,
    )
    pool.close()

    df_content = pd.concat(df_list)
    df_content = df_content.reset_index(drop=False)

    df_content = pd.merge(df_content, df[["dark_id", "title", "created"]], how="left")
    df_content = df_content.rename(columns={"created": "date"})

    df_content.to_feather("data/df_content.feather")
    """



