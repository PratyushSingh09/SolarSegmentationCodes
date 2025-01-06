'''INCOMPLETE and missing stuff'''

import numpy as np

def Plot_pts_on_image(image_data:np.ndarray,pts_list:list,cmap= 'V_K'):
    if cmap == 'V_K':
        c_file = np.loadtxt(r"E:\AIA Data\Coronal Loop\Extra\IDLcolortable4.txt")
        c_file=c_file/255
        cmap = ListedColormap(c_file)

    # fig = plt.figure(dpi=300)
    fig = plt.figure()
    ax = fig.add_subplot(projection=test)

    ax.coords.grid(color='yellow', linestyle='solid', alpha=0.5)

# Coronal loop point :266 237
# Diffuse point :275 236
    # pixel_coord = [236,284] * u.pix
    i=1
    for pixel_coord in pts_list:
        ax.plot(pixel_coord[0], pixel_coord[1], 'x',
            label=f'[Pt {i}: {pixel_coord[0]}, {pixel_coord[1]}]',ms=10,mew=2)
        i+=1
    ax.legend(bbox_to_anchor=(1.37,1.17),ncols=5)

# map_coord = ([-160, 472] * u.arcsec)

# ax.plot(map_coord[0].to('deg'), map_coord[1].to('deg'), 'o', color='white',
#         transform=ax.get_transform('world'),
#         label=f'Map coordinate [{map_coord[0]}, {map_coord[1]}]')
    # ax.legend()

    ax.imshow(image_data,cmap=cmap)
    plt.show()
    # test.plot(axes=ax)