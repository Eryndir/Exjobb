import os
import multiprocessing

from json import loads

# from time import sleep
import pandas as pd
import kblab 
from pandas.core.frame import DataFrame
from tqdm import tqdm
from urllib3.util import Retry
from urllib3 import PoolManager, make_headers
from kblab import Archive
from itertools import product
kblab.VERIFY_CA=False


# Create custom API call to download all metadata files instead of using kblab package
# Retry after failed attempts
def get_metadata(dark_id, headers):
    """Custom API call to download every package's metadata (meta.json) files to filter API content.
    kblab package's .search()-method returns incomplete results, where some some ids are missing
    from search results. Downloading metadata of all id's is the most secure way of ensuring all
    packages are included when filtering for dates or newspaper names.

    Args:
        dark_id (str): URI of package in betalab.kb.se or datalab.kb.se
        headers (list): API authentication details passed as header.

    Returns:
        [dict]: dict with fields of meta.json file.
    """

    http = PoolManager(cert_reqs='CERT_NONE')
    kblab.VERIFY_CA=False
    try:
        print(f"trying https://datalab.kb.se/{dark_id}/meta.json")
        meta_json = http.request(
            "GET",
            f"https://datalab.kb.se/{dark_id}/meta.json",
            headers=headers,
            retries=Retry(connect=5, read=4, redirect=5, backoff_factor=0.02),
        )

        meta_json = loads(meta_json.data.decode("utf-8"))
        meta_json["dark_id"] = dark_id
        return meta_json

    except:
        print("failed getting")
        return {
            "dark_id": dark_id,
            "title": "failed",
            "year": "failed",
        }


if __name__ == "__main__":
    year = "2010"
    decade = year[:-1]
    print("START")
    kblab.VERIFY_CA=False

    with open('api_cred', 'r') as file:
        pw = file.read().replace('\n', '')

    a = Archive("https://datalab.kb.se/", auth=("demo", pw))

    print(f"""tags: "issue" (label:"DAGENS NYHETER" or label:"AFTONBLADET" or label:"Svenska dagbladet") meta.created: ({year} or {decade}1 or {decade}2 or {decade}3 or {decade}4 or {decade}5 or {decade}6 or {decade}7 or {decade}8 or {decade}9)""")
    dark_ids = a.search(f"""tags: "issue" (label:"DAGENS NYHETER" or label:"AFTONBLADET" or label:"Svenska dagbladet") meta.created: ({year} or {decade}1 or {decade}2 or {decade}3 or {decade}4 or {decade}5 or {decade}6 or {decade}7 or {decade}8 or {decade}9 or 2020)""")
    #dark_ids = [dark_id for dark_id in a]

    headers = make_headers(basic_auth=f"demo:{pw}")
    
    pool = multiprocessing.Pool()
    df_meta = pool.starmap(get_metadata, tqdm(list(product(dark_ids, [headers]))), chunksize=5000)
    pool.close()

    os.makedirs("data", exist_ok=True)
    df_meta = pd.DataFrame(df_meta)

    # Keep only newspaper name, throw away date
    df_meta["title"] = df_meta["title"].str.extract(r"(\D*) (\D*)?", expand=False).loc[:, 0]
    nr_failed = len(df_meta[df_meta["year"] == "failed"])
    print(f"A total of {nr_failed} failed.")

    df_meta["year"] = pd.to_numeric(df_meta["year"], errors="coerce")
    df_meta["year"] = df_meta["year"].astype("Int16")
    df_meta = df_meta[~df_meta["issue"].isna()]  # Remove ids that aren't newspapers
    df_meta = df_meta.reset_index(drop=True)

    df_meta.to_feather(f"data/all_metadata_{year}s.feather", compression=None)

    df: DataFrame = pd.read_feather(f"data/all_metadata_{year}s.feather")
    df = df.sample(n=200, random_state=1)

    df = df.reset_index(drop=True)
    df.to_feather(f"data/sampled_editions_{year}s.feather")
