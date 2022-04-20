#From https://gist.github.com/davidstutz/88a12b73813d0e054ece8ab1b53e58a9
import os
import math
import numpy as np

class BoundingBox:
    """
    Represents a bounding box by center, size and orientation. Note that the input will be
    directly read from the KITTI XML files; thus, the z-axis corresponds to height instead
    of depth and the y-axis corresponds to depth instead of height.

    This class will take care of swapping the axes correspondingly.
    """

    # https://gist.github.com/phn/1111712/35e8883de01916f64f7f97da9434622000ac0390
    @staticmethod
    def _normalize_angle(num, lower=0.0, upper=360.0, b=False):
        """Normalize number to range [lower, upper) or [lower, upper].
        Parameters
        ----------
        num : float
            The number to be normalized.
        lower : float
            Lower limit of range. Default is 0.0.
        upper : float
            Upper limit of range. Default is 360.0.
        b : bool
            Type of normalization. See notes.
        Returns
        -------
        n : float
            A number in the range [lower, upper) or [lower, upper].
        Raises
        ------
        ValueError
            If lower >= upper.
        Notes
        -----
        If the keyword `b == False`, the default, then the normalization
        is done in the following way. Consider the numbers to be arranged
        in a circle, with the lower and upper marks sitting on top of each
        other. Moving past one limit, takes the number into the beginning
        of the other end. For example, if range is [0 - 360), then 361
        becomes 1. Negative numbers move from higher to lower
        numbers. So, -1 normalized to [0 - 360) becomes 359.
        If the keyword `b == True` then the given number is considered to
        "bounce" between the two limits. So, -91 normalized to [-90, 90],
        becomes -89, instead of 89. In this case the range is [lower,
        upper]. This code is based on the function `fmt_delta` of `TPM`.
        Range must be symmetric about 0 or lower == 0.
        Examples
        --------
        >>> normalize(-270,-180,180)
        90
        >>> import math
        >>> math.degrees(normalize(-2*math.pi,-math.pi,math.pi))
        0.0
        >>> normalize(181,-180,180)
        -179
        >>> normalize(-180,0,360)
        180
        >>> normalize(36,0,24)
        12
        >>> normalize(368.5,-180,180)
        8.5
        >>> normalize(-100, -90, 90, b=True)
        -80.0
        >>> normalize(100, -90, 90, b=True)
        80.0
        >>> normalize(181, -90, 90, b=True)
        -1.0
        >>> normalize(270, -90, 90, b=True)
        -90.0
        """
        from math import floor, ceil
        # abs(num + upper) and abs(num - lower) are needed, instead of
        # abs(num), since the lower and upper limits need not be 0. We need
        # to add half size of the range, so that the final result is lower +
        # <value> or upper - <value>, respectively.
        res = num
        if not b:
            if lower >= upper:
                raise ValueError("Invalid lower and upper limits: (%s, %s)" % (lower, upper))

            res = num
            if num > upper or num == lower:
                num = lower + abs(num + upper) % (abs(lower) + abs(upper))
            if num < lower or num == upper:
                num = upper - abs(num - lower) % (abs(lower) + abs(upper))

            res = lower if res == upper else num
        else:
            total_length = abs(lower) + abs(upper)
            if num < -total_length:
                num += ceil(num / (-2 * total_length)) * 2 * total_length
            if num > total_length:
                num -= floor(num / (2 * total_length)) * 2 * total_length
            if num > upper:
                num = total_length - num
            if num < lower:
                num = -total_length - num

            res = num * 1.0  # Make all numbers float, to be consistent

        return res

    def __init__(self, size, translation, rotation, source = 'kitti_raw', type = '', meta = ''):
        """
        Constructor.

        :param size: list of length 3 corresponding to (width, height, length)
        :type size: [float]
        :param translation: list of length 3 corresponding to translation (x, y, z
        :type translation: [float]
        :param rotation: list of length 3 corresponding to rotation (x, y, z)
        :type rotation: [float]
        :param kitti: whether to convert from kitti axes
        :type kitti: bool
        :param meta: meta information
        :type meta:
        """

        self.size = []
        """ ([float]) Size of bounding box. """

        if source == 'kitti_raw':
            # original h w l
            # change to l w h
            self.size = [size[2], size[1], size[0]]
        elif source == 'kitti':
            # original h w l
            # change to l w h
            self.size = [size[2], size[1], size[0]]
        else:
            self.size = [size[0], size[1], size[2]]

        # bounding boxes always start at zero height, i.e. its center is
        # determined from height
        self.translation = []
        """ ([float]) Translation, i.e. center, of bounding box. """

        R0_rect = np.array([9.999239000000e-01, 9.837760000000e-03, -7.445048000000e-03,
                            -9.869795000000e-03, 9.999421000000e-01, -4.278459000000e-03,
                            7.402527000000e-03, 4.351614000000e-03, 9.999631000000e-01]).reshape(3, 3)
        R0_rect_inv = np.linalg.inv(R0_rect)
        R_velo_to_cam = np.array([7.533745000000e-03, -9.999714000000e-01, -6.166020000000e-04,
                                    1.480249000000e-02, 7.280733000000e-04, -9.998902000000e-01,
                                    9.998621000000e-01, 7.523790000000e-03, 1.480755000000e-02]).reshape(3, 3)
        R_velo_to_cam_inv = np.linalg.inv(R_velo_to_cam)
        Tr_velo_to_cam = np.array([-4.069766000000e-03,
                                    -7.631618000000e-02,
                                    -2.717806000000e-01]).reshape(3)

        if source == 'kitti_raw':
            # bounding boxes are in camera coordinate system
            # x = right, y = down, z = forward, see paper
            # point clods are in velodyne coordinate system
            # x = forward, y = left, z = up

            # incoming translation is in velodyne coordinates

            self.translation = [-translation[1], translation[2] + self.size[1] / 2, translation[0]]
        elif source == 'kitti':
            # bounding boxes are in camera coordinate system
            # x = right, y = down, z = forward, see paper
            # point clods are in velodyne coordinate system
            # x = forward, y = left, z = up

            # incoming translation is in camera coordinates

            self._translation = np.dot(R_velo_to_cam_inv, np.dot(R0_rect_inv, np.array(translation)) - Tr_velo_to_cam)

            # now in velodyne coordinates

            self.translation = [-self._translation[1], self._translation[2] + self.size[1] / 2, self._translation[0]]
        else:
            self.translation = [translation[0], translation[1], translation[2]]

        self.rotation = []
        """ ([float]) Rotation of bounding box. """

        if source == 'kitti_raw':
            # incoming rotation is in velodyne coordinates
            self.rotation = [rotation[1], self._normalize_angle(rotation[2] + math.pi/2, -math.pi, math.pi), rotation[0]]
        elif source == 'kitti':
            # incoming rotation is in camera coordinates
            self.rotation = [rotation[0], -rotation[2], rotation[1]]
        else:
            self.rotation = [rotation[0], rotation[1], rotation[2]]

        self.type = type
        """ (string) Type. """

        self.meta = meta
        """ (string) Meta information. """

        #print('[Data] (%.2f, %.2f, %.2f), (%.2f, %.2f, %.2f), (%.2f, %.2f, %.2f)' \
        #      % (self.size[0], self.size[1], self.size[2],
        #         self.translation[0], self.translation[1], self.translation[2],
        #         self.rotation[0], self.rotation[1], self.rotation[2]))

    def copy(self):
        """
        Copy the bounding box (deep copy).

        :return: copied bounding box
        :rtype:BoundingBox
        """

        return BoundingBox(self.size, self.translation, self.rotation, False, self.meta)

    @staticmethod
    def from_kitti(filepath):
        """
        Read bounding boxes from KITTI.

        :param filepath: bounding box file
        :type filepath: str
        :return: list of bounding boxes
        :rtype: [BoundingBox]
        """

        bounding_boxes = []
        with open(filepath, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if line.strip() != '']

            for line in lines:
                parts = line.split(' ')
                assert len(parts) == 15 or len(parts) == 16, "invalid number of parts in line %s" % filepath

                bounding_boxes.append(BoundingBox([float(parts[8]), float(parts[9]), float(parts[10])],
                                    [float(parts[11]), float(parts[12]), float(parts[13])],
                                    [0., 0., float(parts[14])], 'kitti', parts[0], filepath))

        return bounding_boxes

    def __str__(self):
        """
        Convert to string representation.

        :return: bounding box as string
        :rtype: str
        """

        return '%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %s' % (
            round(self.size[0], 2), round(self.size[1], 2), round(self.size[2], 2),
            round(self.translation[0], 2), round(self.translation[1], 2), round(self.translation[2], 2),
            round(self.rotation[0], 2), round(self.rotation[1], 2), round(self.rotation[2], 2),
            self.meta
        )

def write_bounding_boxes(file, bounding_boxes):
    """
    Write bounding boxes to file.

    :param file: path to file to write
    :type file: str
    :param bounding_boxes: bounding boxes to write
    :type bounding_boxes: [BoundingBox]
    """

    with open(file, 'w') as f:
        f.write('%d\n' % len(bounding_boxes))
        for bounding_box in bounding_boxes:
            f.write(str(bounding_box) + '\n')

def read_bounding_boxes(file):
    """
    Reads bounding boxes.

    :param file: path to file to read
    :type file: str
    :return: bounding boxes
    :rtype: [BoundingBox]
    """

    assert os.path.exists(file), 'file %s not found' % file

    with open(file, 'r') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines if line.strip()]
        lines = lines[1:]

        bounding_boxes = []
        for line in lines:
            parts = line.split(' ')
            parts = [part.strip() for part in parts if part]

            assert len(parts) == 9 or len(parts) == 10, 'invalid bounding box line: %s' % line

            size = (float(parts[0]), float(parts[1]), float(parts[2]))
            translation = (float(parts[3]), float(parts[4]), float(parts[5]))
            rotation = (float(parts[6]), float(parts[7]), float(parts[8]))

            meta = ''
            if len(parts) > 9:
                meta = parts[9]

            bounding_boxes.append(BoundingBox(size, translation, rotation, '', '', meta))

        return bounding_boxes