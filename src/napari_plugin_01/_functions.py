#%%
from magicgui import magic_factory
from pathlib import Path

from ._transfer_learning import Run_Transfer_Learning_Train

# %%
@magic_factory(directory={"mode": "d", "label": "Folder with images for training:"})
def Train(Debug = False, directory=Path.home()):
    Run_Transfer_Learning_Train("d://Programovani//MachineLearning2024_Prague_Conference_Transfer_Learning//images")