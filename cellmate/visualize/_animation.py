import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate_images(image):
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
        im = ax.imshow(image[i], animated=True)
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
