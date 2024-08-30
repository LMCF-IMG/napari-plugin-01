#%%
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import napari

from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget

from magicgui import magic_factory

# %%
class MyQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btn)

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")


# %%
@magic_factory
def MyExampleMagicGUI(x: int, y="hi"):
    """Basic example function."""
    return x, y
