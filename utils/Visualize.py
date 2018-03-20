import numpy as np
import matplotlib.pyplot as plt



def diff(img, img1):  # returns just the difference of the two images
    return cv2.absdiff(img, img1)


def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()




def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


def plot_data(pred_all, y_all):
    c = ['#ff0000', '#009999']
    for i in range(2):
        f = pred_all[np.where(y_all == i)]
        plt.plot(f[:, 0], f[:, 1], '.', c=c[i])
    plt.legend(['0', '1'])
    plt.savefig('result.png')
