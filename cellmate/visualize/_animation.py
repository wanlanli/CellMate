import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate_images(image, *args, **kwargs):
    """
    Create an animation of the image frames.

    This function takes a 3D image array and creates an animation where each frame of the
    animation corresponds to a slice along the first dimension of the image array.

    Parameters:
    -----------
    image : np.array
        A 3D numpy array with dimensions [Time x Width x Height].

    Returns:
    --------
    ani : matplotlib.animation.ArtistAnimation
        The created animation object.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_yticks([])
    ims = []
    for i in range(image.shape[0]):
        im = ax.imshow(image[i], animated=True, *args, **kwargs)
        title = ax.text(0.5, 1.05, f'Frame {i+1}', ha="center", va="baseline", 
                        fontsize=12, transform=ax.transAxes, animated=True)
        ims.append([im, title])
    plt.close()
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    return ani


def animate_tracking_with_annotation(mask, annotation=False):
    """
    Create an animation of the image frames.

    This function takes a 3D image array and creates an animation where each frame of the
    animation corresponds to a slice along the first dimension of the image array.

    Parameters:
    -----------
    image : np.array
        A 3D numpy array with dimensions [Time x Width x Height].

    Returns:
    --------
    ani : matplotlib.animation.ArtistAnimation
        The created animation object.
    """
    from ._utils import label2rgb
    image = label2rgb(mask)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_yticks([])
    ims = []
    for i in range(image.shape[0]):
        im_i = []
        im = ax.imshow(image[i], animated=True)
        title = ax.text(0.5, 1.05, f'Frame {i+1}', ha="center", va="baseline", 
                        fontsize=12, transform=ax.transAxes, animated=True)
        im_i += [im, title]

        if annotation:
            labels = np.unique(mask[i])[1:]
            for label in labels:
                x, y = np.where(mask[i] == label)
                ann = ax.text(y.mean()-5, x.mean(), f'{label}', fontsize=8, c='w', animated=True)
                im_i.append(ann)
        ims.append(im_i)
    plt.close()
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    return ani


def animate_patch(intensity, properties=None, peaks=None, *args, **kwargs):
    """
    Create an animation of the image frames.

    This function takes a 3D image array and creates an animation where each frame of the
    animation corresponds to a slice along the first dimension of the image array.

    Parameters:
    -----------
    image : np.array
        A 3D numpy array with dimensions [Time x Width x Height].

    Returns:
    --------
    ani : matplotlib.animation.ArtistAnimation
        The created animation object.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ims = []
    for i in range(intensity.shape[0]):
        im_i = []
        im = ax.plot(intensity[i], animated=True, *args, **kwargs)
        im_i += [im[0]]
        if properties is not None:
            prop = properties[i]
            x_min = prop["left_ips"]
            x_max = prop["right_ips"]
            y_i = prop["width_heights"]
            switch = np.where(x_min > x_max)[0]
            if len(switch) > 0:
                y_i = np.concatenate([y_i, y_i[switch]])
                x_min = np.concatenate([x_min, [0]*len(switch)])
                x_max = np.concatenate([x_max, x_max[switch]])
                x_max[switch] = [len(intensity[i])]*len(switch)
            marker = ax.hlines(y=y_i, xmin=x_min, xmax=x_max, color="C1")
            im_i += [marker]
        if peaks is not None:
            peak = peaks[i]
            peak_im = ax.plot(peak, intensity[i][peak],  "*", markersize=20, label='Detected Peaks', color='red')
            im_i += peak_im

        title = ax.text(0.5, 1.05, f'Frame {i+1}', ha="center", va="baseline",
                        fontsize=12, transform=ax.transAxes, animated=True)
        im_i += [title]
        ims.append(im_i)
    plt.close()
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    return ani
