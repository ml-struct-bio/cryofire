"""
Masks
"""

import utils

log = utils.log


def get_circular_mask(lattice, radius):
    """
    lattice: Lattice
    radius: float

    output: [resolution**2]
    """
    coords = lattice.coords
    resolution = lattice.D
    assert 2 * radius + 1 <= resolution, 'Mask with radius {} too large for lattice with size {}'.format(radius,
                                                                                                         resolution)
    r = radius / (resolution // 2) * lattice.extent
    mask = coords.pow(2).sum(-1) <= r ** 2
    if lattice.ignore_DC:
        mask[resolution ** 2 // 2] = 0
    return mask


class CircularMask:
    def __init__(self, lattice, radius):
        """
        lattice: Lattice
        radius: float
        """
        self.lattice = lattice
        self.binary_mask = get_circular_mask(lattice, radius)
        self.current_radius = radius

    def update_radius(self, radius):
        """
        radius: float
        """
        self.binary_mask = get_circular_mask(self.lattice, radius)
        self.current_radius = radius


class FrequencyMarchingMask(CircularMask):
    def __init__(self, lattice, radius_max, radius=3, add_one_every=100000):
        """
        lattice: Lattice
        radius: float
        add_one_every: int
        """
        super().__init__(lattice, radius)
        self.add_one_every = add_one_every
        self.radius_init = radius
        self.radius_max = radius_max
        log("Frequency marching initialized at r = {}".format(radius))

    def update(self, total_images_count):
        """
        total_images_count
        """
        new_radius = self.radius_init + int(total_images_count / self.add_one_every)
        if new_radius > self.current_radius and new_radius <= self.radius_max:
            self.update_radius(new_radius)
            log("Mask updated. New radius = {}".format(self.current_radius))
