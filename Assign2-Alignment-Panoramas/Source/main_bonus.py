# ================================================
# Skeleton codes for HW4
# Read the skeleton codes carefully and put all your
# codes into main function
# ================================================

import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.data import astronaut
from skimage.util import img_as_float
import maxflow
from scipy.spatial import Delaunay


def help_message():
    print("Usage: [Input_Image] [Input_Marking] [Output_Directory]")
    print("[Input_Image]")
    print("Path to the input image")
    print("[Input_Marking]")
    print("Path to the input marking")
    print("[Output_Directory]")
    print("Output directory")
    print("Example usages:")
    print(sys.argv[0] + " astronaut.png " + "astronaut_marking.png " + "./")


# Calculate the SLIC superpixels, their histograms and neighbors
def superpixels_histograms_neighbors(img):
    # SLIC
    segments = slic(img, n_segments=500, compactness=18.5)
    segments_ids = np.unique(segments)

    # centers
    centers = np.array([np.mean(np.nonzero(segments == i), axis=1) for i in segments_ids])

    # H-S histograms for all superpixels
    hsv = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2HSV)
    bins = [20, 20]  # H = S = 20
    ranges = [0, 360, 0, 1]  # H: [0, 360], S: [0, 1]
    colors_hists = np.float32(
        [cv2.calcHist([hsv], [0, 1], np.uint8(segments == i), bins, ranges).flatten() for i in segments_ids])

    # neighbors via Delaunay tesselation
    tri = Delaunay(centers)

    return (centers, colors_hists, segments, tri.vertex_neighbor_vertices)


# Get superpixels IDs for FG and BG from marking
def find_superpixels_under_marking(marking, superpixels):
    fg_segments = np.unique(superpixels[marking[:, :, 0] != 255])
    bg_segments = np.unique(superpixels[marking[:, :, 2] != 255])
    return (fg_segments, bg_segments)


# Sum up the histograms for a given selection of superpixel IDs, normalize
def cumulative_histogram_for_superpixels(ids, histograms):
    h = np.sum(histograms[ids], axis=0)
    return h / h.sum()


# Get a bool mask of the pixels for a given selection of superpixel IDs
def pixels_for_segment_selection(superpixels_labels, selection):
    pixels_mask = np.where(np.isin(superpixels_labels, selection), True, False)
    return pixels_mask


# Get a normalized version of the given histograms (divide by sum)
def normalize_histograms(histograms):
    return np.float32([h / h.sum() for h in histograms])


# Perform graph cut using superpixels histograms
def do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors):
    num_nodes = norm_hists.shape[0]
    # Create a graph of N nodes, and estimate of 5 edges per node
    g = maxflow.Graph[float](num_nodes, num_nodes * 5)
    # Add N nodes
    nodes = g.add_nodes(num_nodes)

    hist_comp_alg = cv2.HISTCMP_KL_DIV

    # Smoothness term: cost between neighbors
    indptr, indices = neighbors
    for i in range(len(indptr) - 1):
        N = indices[indptr[i]:indptr[i + 1]]  # list of neighbor superpixels
        hi = norm_hists[i]  # histogram for center
        for n in N:
            if (n < 0) or (n > num_nodes):
                continue
            # Create two edges (forwards and backwards) with capacities based on
            # histogram matching
            hn = norm_hists[n]  # histogram for neighbor
            g.add_edge(nodes[i], nodes[n], 20 - cv2.compareHist(hi, hn, hist_comp_alg),
                       20 - cv2.compareHist(hn, hi, hist_comp_alg))

    # Match term: cost to FG/BG
    for i, h in enumerate(norm_hists):
        if i in fgbg_superpixels[0]:
            g.add_tedge(nodes[i], 0, 1000)  # FG - set high cost to BG
        elif i in fgbg_superpixels[1]:
            g.add_tedge(nodes[i], 1000, 0)  # BG - set high cost to FG
        else:
            g.add_tedge(nodes[i], cv2.compareHist(fgbg_hists[0], h, hist_comp_alg),
                        cv2.compareHist(fgbg_hists[1], h, hist_comp_alg))

    g.maxflow()
    return g.get_grid_segments(nodes)


def RMSD(target, master):
    # Note: use grayscale images only

    # Get width, height, and number of channels of the master image
    master_height, master_width = master.shape[:2]
    master_channel = len(master.shape)

    # Get width, height, and number of channels of the target image
    target_height, target_width = target.shape[:2]
    target_channel = len(target.shape)

    # Validate the height, width and channels of the input image
    if (master_height != target_height or master_width != target_width or master_channel != target_channel):
        return -1
    else:

        total_diff = 0.0;
        dst = cv2.absdiff(master, target)
        dst = cv2.pow(dst, 2)
        mean = cv2.mean(dst)
        total_diff = mean[0] ** (1 / 2.0)

        return total_diff;

def interactiveSeg():
    print "Testing"
    fg_segments, bg_segments = find_superpixels_under_marking(img_marking, superpixels)
    fg_hists = cumulative_histogram_for_superpixels(fg_segments, color_hists)
    bg_hists = cumulative_histogram_for_superpixels(bg_segments, color_hists)
    norm_hists = normalize_histograms(color_hists)

    fgbg_superpixels = [fg_segments, bg_segments]
    fgbg_hists = [fg_hists, bg_hists]

    graph_cut = do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors)
    mask = pixels_for_segment_selection(superpixels, np.nonzero(graph_cut))
    mask = np.uint8(mask * 255)
    cv2.imshow('Mask', mask)
# mouse callback function
def draw(event,former_x,former_y,flags,param):
    global current_former_x,current_former_y,drawing, drawingBlue,color
    if event==cv2.EVENT_LBUTTONDOWN:
        print "lctrl"
        drawing = True
        color = (0,0,255)
        current_former_x,current_former_y=former_x,former_y
    elif event == cv2.EVENT_LBUTTONDBLCLK:
            print "ctrl"
            color = (255, 0, 0)
            drawingBlue = True
            current_former_x, current_former_y = former_x, former_y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(img,(current_former_x,current_former_y),(former_x,former_y),color,5)
            cv2.line(img_marking, (current_former_x, current_former_y), (former_x, former_y), color, 5)
            current_former_x = former_x
            current_former_y = former_y
        if drawingBlue == True:
            cv2.line(img, (current_former_x, current_former_y), (former_x, former_y), color, 5)
            cv2.line(img_marking, (current_former_x, current_former_y), (former_x, former_y),color, 5)
            current_former_x = former_x
            current_former_y = former_y
                #print former_x,former_y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        drawingBlue =False
        cv2.line(img,(current_former_x,current_former_y),(former_x,former_y),color,5)
        cv2.line(img_marking, (current_former_x, current_former_y), (former_x, former_y), color, 5)
        current_former_x = former_x
        current_former_y = former_y
        interactiveSeg()
        return former_x,former_y


if __name__ == '__main__':

    # # validate the input arguments
    # if (len(sys.argv) != 4):
    #     help_message()
    #     sys.exit()
    #
    # img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    # img_marking = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)

    # ======================================== #
    # write all your codes here
    drawing = False
    drawingBlue = False
    img = cv2.imread("astronaut.png", cv2.IMREAD_COLOR)
    img_copy =np.copy(img)
    img_marking = np.ones(img.shape, np.uint8)*255

    centers, color_hists, superpixels, neighbors = superpixels_histograms_neighbors(img_copy)


    cv2.namedWindow("Astro")
    cv2.setMouseCallback('Astro', draw)
    while (1):
        cv2.imshow('Astro', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()



    # ======================================== #

    # read video file
    # output_name = sys.argv[3] + "mask.png"
    # cv2.imwrite(output_name, mask); 	