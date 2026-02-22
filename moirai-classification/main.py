# %%
from tslearn.datasets import UCR_UEA_datasets
from uni2ts.model.moirai import MoiraiModule

# %%
ds = UCR_UEA_datasets()
X_train, y_train, X_test, y_test = ds.load_dataset("LSST")

# %%
module = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small")

# %%