from wildlife_datasets import datasets
from lib.experimentClass import ExpSetup

wildlife_datasets = [
    "OpenCows2020",
    "ATRW",
    "BelugaID",
    "CTai",
    "GiraffeZebraID",
    "Giraffes",
    "HumpbackWhaleID",
    "HyenaID2022",
    "IPanda50",
    "LeopardID2022",
    "MacaqueFaces",
    "NyalaData",
    "SealID",
    "SeaTurtleIDHeads",
    # "StripeSpotter", # Downloaded externally
    "WhaleSharkID",
    "ZindiTurtleRecall"
]

exper = ExpSetup()
print(exper.data_dir)
for dataset in wildlife_datasets:
    dataset_func = getattr(datasets, dataset)

    if dataset == "SealID":
        dataset_func.get_data(
            f"{exper.datasets_dir}/{dataset}", force=False, 
            url="https://download.fairdata.fi:443/download?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE3MzI4MTkyNTAsImRhdGFzZXQiOiIyMmI1MTkxZS1mMjRiLTQ0NTctOTNkMy05NTc5N2M5MDBmYzAiLCJwYWNrYWdlIjoiMjJiNTE5MWUtZjI0Yi00NDU3LTkzZDMtOTU3OTdjOTAwZmMwX3JfX2VfM202LnppcCIsImdlbmVyYXRlZF9ieSI6IjViYzEzMTNiLWU2NjctNDc5NS04NWMzLWZhYjJiODk4OWRkZCIsInJhbmRvbV9zYWx0IjoiZDkxYjY1ZDIifQ.8DEcbfxCtTYhHAb0L-8AMk-y0cS9iMY_Ha_qLmxs-7c"
            )
    else:
        dataset_func.get_data(f"{exper.datasets_dir}/{dataset}", force=False)