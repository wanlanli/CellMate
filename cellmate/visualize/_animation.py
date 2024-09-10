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
    fig, ax = plt.subplots()
    ims = []
    for i in range(image.shape[0]):
        im = ax.imshow(image[i], animated=True)
        ims.append([im])
    plt.close()
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    return ani
