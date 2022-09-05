import numpy as np

def divide_frame(frame):
    """
    Returns the number of macroblocks the given frame has vertically and horizontally.
    """
    height, width = frame.shape
    vertical_mblocks = int(height/ 16)
    horizontal_mblocks = int(width/ 16)
    total_mblocks = int(vertical_mblocks * horizontal_mblocks)

    return vertical_mblocks, horizontal_mblocks

def find_center(x, y):
    """
    Determines the center of a block whose top left corner coordinates are x, y.
    e.g. for the block whose top left corner coordinates are (0, 0), the center is (8, 8).
    """
    return(int(x + 8), int(y + 8))

def find_search_area(x, y, frame):
    """
    Returns the search area of a frame based to the coordinates given.
    - x, y are the top left corner's coordinates of the given macroblock.
    """
    height, width = frame.shape
    center_x, center_y = find_center(x, y)
    
    # ensure search area is within bounds
    # find the top left corner's coordinates of the search area
    search_x = max(0, center_x - 24)    
    search_y = max(0, center_y - 24)

    # slice frame within bounds to produce the search area
    search_area = frame[search_y:min(search_y + 48, height), search_x:min(search_x + 48, width)]

    return search_area

def get_block_zone(p, search_area, current_block):
    """
    Retrieves the block searched in the search area of the initial I-frame to be compared with the current_block of the current frame.
    - (x, y): coordinates of the current frame's macroblock center.
    """   
    px, py = p                          # coordinates of the macroblock's center
    px, py = px - 8, py - 8             # top left corner
    px, py = max(0, px), max(0, py)     # ensure within bounds

    a_block = search_area[py:py + 16, px:px + 16]

    # make sure the two blocks have the same shape
    try:
        assert a_block.shape == current_block.shape
    except:
        print("The blocks should have the same shape!")

    return a_block

def find_mad(current_block, a_block):
    """
    Returns Mean Absolute Difference between current_block and a_block.
    - current_block: current frame macroblock.
    - a_block: I-frame macroblock.
    """
    return np.sum(np.abs(np.subtract(current_block, a_block)))/ (current_block.shape[0] * current_block.shape[1])

def find_match(current_block, search_area):
    """
    Algorithm to find the best pair of blocks between the current frame's block and the I-frame's block. 
    Returns the macroblock from the I-frame's search area with the least Mean Absolute Difference.
    """
    step = 4
    sa_height, sa_width = search_area.shape
    # find the center of the I-frame's search area
    sa_centerY, sa_centerX = int(sa_height/2), int(sa_width/2)

    min_mad = float("+inf")
    minP = None

    while step >= 1:
        p1 = (sa_centerX, sa_centerY)
        p2 = (sa_centerX + step, sa_centerY)
        p3 = (sa_centerX, sa_centerY + step)
        p4 = (sa_centerX + step, sa_centerY + step)
        p5 = (sa_centerX - step, sa_centerY)
        p6 = (sa_centerX, sa_centerY - step)
        p7 = (sa_centerX - step, sa_centerY - step)
        p8 = (sa_centerX + step, sa_centerY - step)
        p9 = (sa_centerX - step, sa_centerY + step)

        # retrieve the nine search points
        pointList = [p1, p2, p3, p4, p5, p6, p7, p8, p9]

        for p in range(len(pointList)):
            # get the I-frame's macroblock
            a_block = get_block_zone(pointList[p], search_area, current_block)
            mad = find_mad(current_block, a_block)
            if mad < min_mad:
                min_mad = mad
                minP = pointList[p]
            
        step = int(step/2)

    px, py = minP                       # center of the macroblock with the least M.A.D.
    px, py = px - 8, py - 8             # top left corner
    px, py = max(0, px), max(0, py)     # ensure it's within bounds

    match = search_area[py:py + 16, px:px + 16]

    return match 

def create_predicted(frame, target_frame):
    """
    Helps create the predicted frame based on the initial I-frame and the target-frame.
    """
    height, width = frame.shape
    vertical_mblocks, horizontal_mblocks = divide_frame(frame)

    predicted = np.ones((height, width)) * 255
    bcount = 0

    for y in range(0, int(vertical_mblocks * 16), 16):
        for x in range(0, int(horizontal_mblocks * 16), 16):
            bcount += 1
            target_block = target_frame[y:y + 16, x:x + 16]
            search_area = find_search_area(x, y, frame)
            iframe_block = find_match(target_block, search_area)
            
            # add iframe_block to the predicted frame
            predicted[y:y + 16, x:x + 16] = iframe_block

    assert bcount == int(horizontal_mblocks * vertical_mblocks)

    return predicted

def find_residual(target_frame, predicted_frame):
    """
    Find the residual frame via target_frame - predicted_frame.
    """
    return np.subtract(target_frame, predicted_frame)

def reconstruct_target(residual_frame, predicted_frame):
    """
    Reconstruct the target frame by adding the residual_frame to the predicted_frame.
    """
    return np.add(residual_frame, predicted_frame)
