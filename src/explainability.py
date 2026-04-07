
import numpy as np

def denormalize(img):
    img = img.permute(1,2,0).cpu().numpy()
    img = (img * 0.5) + 0.5
    return np.clip(img, 0, 1)

def normalize_map(attr_map):
    p99 = np.percentile(attr_map, 99)
    return np.clip(attr_map / (p99 + 1e-8), 0, 1)


def visualize_attr_map(orig_img, attr, title, ax, title_style):
    attr_map = attr.squeeze().cpu().detach().numpy()

    if len(attr_map.shape) > 2:
        attr_map = np.mean(np.abs(attr_map), axis=0)

    attr_map = normalize_map(attr_map)
    img_show = denormalize(orig_img)

    ax.imshow(img_show[:,:,0], cmap='gray')
    ax.imshow(attr_map, cmap='jet', alpha=0.4)

    ax.set_title(title, **title_style)
    ax.axis("off")