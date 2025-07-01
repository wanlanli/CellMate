import SimpleITK as sitk
import numpy as np


def rigid(image: np.array, reference_channel=0, maxStep=1.0, minStep=0.01, numberOfIterations=1000,
          relaxationFactor=0.05, tolerance=1e-8):
    """
    Rigid regisitration
    ----------
    Parameters:
    ----------
    image: np.array, [Time x Chaneel x width x heigh]

    Returns:
    ---------
    moved_data: rigied image array with same shape
    """
    # set method
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation()
    R.SetOptimizerAsRegularStepGradientDescent(maxStep,
                                               minStep,
                                               numberOfIterations,
                                               relaxationFactor,
                                               gradientMagnitudeTolerance=tolerance)
    R.SetInterpolator(sitk.sitkLinear)
    # loop rigid
    for i in range(0, image.shape[0]):
        if i == 0:
            # init fixed image by #0 frame
            fixed = sitk.GetImageFromArray(image[0, reference_channel].astype(np.float32))
            R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
            # init resampler
            resampler = sitk.ResampleImageFilter()
            resampler.SetDefaultPixelValue(0)
            resampler.SetInterpolator(sitk.sitkLinear)
            # set fixed frame
            moved_data = np.zeros(image.shape, dtype=np.uint16)
            moved_data[0] = image[0]
            continue
        moving = sitk.GetImageFromArray(image[i, 0].astype(np.float32))
        outTx = R.Execute(fixed, moving)
        resampler.SetReferenceImage(fixed)
        resampler.SetTransform(outTx)
        for j in range(0, image.shape[1]):
            out = sitk.GetImageFromArray(image[i, j].astype(np.float32))
            out = resampler.Execute(out)
            if j == 0:
                fixed = out
            moved_data[i, j] = sitk.GetArrayFromImage(out).astype(np.uint16)
    return moved_data


if __name__ == '__main__':
    from ..io import imread, imsave
    image = imread("./cellmating/data/example_for_rigid.tif")
    image = np.moveaxis(image, -1, 1)
    moved = rigid(image)
    imsave("./cellmating/data/example_for_rigid_moved.tif",
           moved.astype(np.uint16),
           imagej=True)
# python -m python -m cellmating.registration._rigid
