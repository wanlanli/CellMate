import matplotlib as mpl


class COLOR():
    def __init__(self, colorlist) -> None:
        self.colorlist = colorlist

    def linear(self):
        return mpl.colors.LinearSegmentedColormap.from_list("", self.colorlist)

    def list(self):
        return mpl.colors.ListedColormap(self.colorlist)


black_pink_colors = ["black", "indigo", "magenta", "pink"]
gaudi_colors = ['#f1a533', '#0f2ad4', '#4aa22e', '#f9de4b', '#3f8ef0', '#80ba38', '#f7f0dc', '#97c8e9', '#4faea7', '#87ebe9']
mont_colors = ['#eeeeee', '#164487', '#fdd619', '#d72235', '#ea8902', '#a8c84a', '#716489', '#111111']
patch_colors = ['#eeeeee', '#d72235', '#a8c84a', '#164487']
blue_pink_color = ['#88D4E7', '#F0B4D0']
blue_pink_color_long = ["#88D4E7","#BAE3E2","#F2F8F6","#F8D6DE","#F0B4D0"]
default = mpl.colormaps['nipy_spectral']


COLORMAP = {
    'default': default,
    'blackpink': COLOR(black_pink_colors),
    'gaudi': COLOR(gaudi_colors),
    'mont': COLOR(mont_colors),
    'patch': COLOR(patch_colors),
    'bp': COLOR(blue_pink_color),
    'bp_long': COLOR(blue_pink_color_long)
}


if __name__ == 'main':
    import numpy as np
    x, y, c = zip(*np.random.rand(30, 3)*4-2)
    norm = mpl.pyplot.Normalize(-2, 2)
    mpl.pyplot.scatter(x, y, c=c, cmap=COLORMAP['default'], norm=norm)
    mpl.pyplot.colorbar()
    mpl.pyplot.show()
