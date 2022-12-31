import numpy as np
from matplotlib import pyplot as plt


class Domain:
    """
    Generic representation of the geometry of a 2D + channel rectangular domain
    """
    def __init__(self, coords=None, center=None, height=None, width=None):
        """
        Args:
            coords: ((top, bottom), (left, right))

                OR

            center: (i, j)
            height: height
            width: width
        """

        assert (coords is not None) ^ all([x is not None for x in (center, height, width)])

        if coords is not None:
            self.coords = coords
        else:
            self.coords = coords_from_center_height_width(center, height, width)

    @property
    def t(self):
        return self.coords[0][0]

    @property
    def b(self):
        return self.coords[0][1]

    @property
    def l(self):
        return self.coords[1][0]

    @property
    def r(self):
        return self.coords[1][1]

    @property
    def width(self):
        return self.r - self.l

    @property
    def height(self):
        return self.b - self.t

    @property
    def center(self):
        return (self.t + (self.b - self.t) / 2,
                self.l + (self.r - self.l) / 2)

    @property
    def shape(self):
        return (self.height, self.width)

    @property
    def polygon(self):
        """
        Convert to polygon, corners in clockwise order:

            1 2
            4 3

        Returns:
            polygon: 4x2 numpy array, 4 (i,j) tuples of vertex positions.
        """

        ij = ((self.t, self.l),
              (self.t, self.r),
              (self.b, self.r),
              (self.b, self.l))
        return np.array(ij)


class ReplicationDomain:

    def __init__(self, x_shape, buffer=0, yard=0):

        assert x_shape[0] == x_shape[1], "we only support square X!"

        self.x_shape = x_shape
        self.buffer = buffer
        self.yard = yard

        self._x_w = x_shape[0]
        self._x_hat_w = self._x_w + 2 * self.buffer  # Width of X and it's buffer.

        # All coords and padding are in ((top, bottom), (left, right))

        # Our domain is big enough for two daughters side-by-side and their yard.
        self.full_domain = Domain(coords=((0, self._x_hat_w), (0, 2 * self._x_hat_w + self.yard)))

        # And the parent is centered within this domain
        center = self.full_domain.center
        self.parent_domain = Domain(center=center, height=self._x_w, width=self._x_w)

        # The daughters are shifted left and right such that their buffers will be adjacent
        daughter_offset = int(self._x_hat_w / 2)
        self.daughter_domains = (Domain(center=(center[0], center[1] - daughter_offset),
                                        height=self._x_w, width=self._x_w),
                                 Domain(center=(center[0], center[1] + daughter_offset),
                                        height=self._x_w, width=self._x_w))


def get_domain_padding(domain, subdomain, buffer=0):
    """
    Return the padding that would be required to convert the sub-Domain into the enclosing Domain
    or, equivalently, the cropping required to crop the sub-domain from within the enclosing domain.
    Args:
        domain: Domain to crop from / pad to
        subdomain: sub-Domain enclosed by domain
        buffer: optionally crop/pad with a buffer of this width outside the sub-domain.

    Returns:

    """

    padding = ((subdomain.t - domain.t, domain.b - subdomain.b), (subdomain.l - domain.l, domain.r - subdomain.r))
    padding = tuple(map(tuple, np.array(padding) - buffer))
    assert np.all(np.array(padding) >= 0), "The input domain pairs result in negative padding!"

    return padding


def coords_from_center_height_width(c, h, w):
    return ((int(c[0]-h/2), int(c[0]+h/2)),
            (int(c[1]-w/2), int(c[1]+w/2)))


def draw_domain(domain):
    ij = domain.polygon
    plt.fill(ij[:, 1], ij[:, 0], alpha=0.2)
