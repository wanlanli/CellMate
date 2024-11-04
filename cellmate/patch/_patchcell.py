

from cellmate.mating import CellNetwork90


class CellNetworkPatch(CellNetwork90):
    def __init__(self, image, time_network, tracker, threshold, *args, **kwargs):
        super().__init__(image, time_network, tracker, threshold, *args, **kwargs)

    def raw_patch(self, cell_id):
        pass
