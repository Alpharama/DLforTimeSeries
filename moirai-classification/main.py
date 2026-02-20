# %%
from uni2ts.eval_util.data import get_gluonts_test_dataset
from uni2ts.eval_util.plot import plot_next_multi
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
# %%
model = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small")


# %%