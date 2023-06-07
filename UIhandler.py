import cv2
import copy
import numpy as np

class EventHandler:
    """
    Class for handling user input during segmentation iterations 
    """

    def __init__(self, flags, img, _mask, colors):
        
        self.FLAGS = flags
        self.ix = -1
        self.iy = -1
        self.img = img
        self.img2 = copy.deepcopy(img)
        self._mask = _mask
        self.COLORS = colors
        self.circSize = 2

    @property
    def image(self):
        return self.img

    @image.setter
    def image(self, img):
        self.img = img
        
    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, _mask):
        self._mask = _mask

    @property
    def flags(self):
        return self.FLAGS 

    @flags.setter
    def flags(self, flags):
        self.FLAGS = flags

    def handler(self, event, x, y, flags, param):
        
        # Draw rectangular mask for missing pixels (and invert later)
        if event == cv2.EVENT_RBUTTONDOWN:
            self.FLAGS['DRAW_RECT'] = True
            self.ix, self.iy = x,y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.FLAGS['DRAW_RECT'] == True:
                self.img = copy.deepcopy(self.img2)
                cv2.rectangle(self.img, (self.ix, self.iy), (x, y), self.COLORS['BLUE'], 2)
                self.FLAGS['RECT'] = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))
                self.FLAGS['rect_or_mask'] = 0

        elif event == cv2.EVENT_RBUTTONUP:
            self.FLAGS['DRAW_RECT'] = False
            self.FLAGS['rect_over'] = True
            cv2.rectangle(self.img, (self.ix, self.iy), (x, y), self.COLORS['BLUE'], 2)
            self.FLAGS['RECT'] = (min(self.ix, x), min(self.iy, y), abs(self.ix - x), abs(self.iy - y))
            self.FLAGS['rect_or_mask'] = 0
            fr = self.FLAGS['RECT']
            # print(mask.shape)
            self._mask[fr[1]:fr[1] + fr[3], fr[0]:fr[0] + fr[2]] = 255 #here
        # Draw strokes for refinement - sets values in mask at those places
        # Uncomment the following for manual refinement part 
        # **NOT MASKING, BUT REFINEMENT**

        if event == cv2.EVENT_LBUTTONDOWN:
            self.FLAGS['DRAW_STROKE'] = True
            cv2.circle(self.img, (x,y), self.FLAGS['circSize'], self.FLAGS['value']['color'], -1)
            cv2.circle(self._mask, (x,y), self.FLAGS['circSize'], self.FLAGS['value']['val'], -1)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.FLAGS['DRAW_STROKE'] == True:
                cv2.circle(self.img, (x, y), self.FLAGS['circSize'], self.FLAGS['value']['color'], -1)
                cv2.circle(self._mask, (x, y), self.FLAGS['circSize'], self.FLAGS['value']['val'], -1)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.FLAGS['DRAW_STROKE'] == True:
                self.FLAGS['DRAW_STROKE'] = False
                cv2.circle(self.img, (x, y), self.FLAGS['circSize'], self.FLAGS['value']['color'], -1)
                cv2.circle(self._mask, (x, y), self.FLAGS['circSize'], self.FLAGS['value']['val'], -1)


def drawMask(img):
    """
    Getting our manual mask. 

    Input
    -----
    filename (str) : Path to image
    """

    COLORS = {
    'BLACK' : [0,0,0],
    'RED'   : [0, 0, 255],
    'GREEN' : [0, 255, 0],
    'BLUE'  : [255, 0, 0],
    'WHITE' : [255,255,255]
    }

    DRAW_BG = {'color' : COLORS['RED'], 'val' : 2}
    DRAW_FG = {'color' : COLORS['GREEN'], 'val' : 255}

    FLAGS = {
        'RECT' : (0, 0, 1, 1),
        'DRAW_STROKE': False,         # flag for drawing strokes
        'DRAW_RECT' : False,          # flag for drawing rectangle
        'rect_over' : False,          # flag to check if rectangle is  drawn
        'rect_or_mask' : -1,          # flag for selecting rectangle or stroke mode
        'value' : DRAW_FG,            # drawing strokes initialized to mark foreground
        'circSize' : 2,
    }

    # img = cv2.imread(filename)
    img2 = img.copy()                                
    mask = np.zeros(img.shape[:2], dtype = np.uint8) # mask is a binary array with : 0 - background pixels
                                                        #                               1 - foreground pixels 
    output = np.zeros(img.shape, np.uint8)           # output image to be shown

    # Input and segmentation windows
    cv2.namedWindow('Input Image',cv2.WINDOW_GUI_NORMAL)
    # cv2.namedWindow('Segmented output')

    EventObj = EventHandler(FLAGS, img, mask, COLORS)
    cv2.setMouseCallback('Input Image', EventObj.handler)
    cv2.moveWindow('Input Image', img.shape[1] + 10, 90)

    maskarr = []
    fno = 0

    while(1):
        
        img = EventObj.image
        mask = EventObj.mask
        FLAGS = EventObj.flags
        # cv2.imshow('Segmented image', output)
        cv2.imshow('Input Image', img)
        
        k = cv2.waitKey(1)

        # key bindings
        if k == 27:
            # esc to exit
            break
        
        elif k == ord('p'):
            FLAGS['circSize'] += 1

        elif k == ord('o'):
            FLAGS['circSize'] = max(1,FLAGS['circSize']-1) 

        elif k == ord('b'): 
            # Strokes for background 2
            FLAGS['value'] = DRAW_BG
        
        elif k == ord('f'):
            # FG drawing 3
            FLAGS['value'] = DRAW_FG
        
        elif k == ord('r'):
            # reset everything
            FLAGS['RECT'] = (0, 0, 1, 1)
            FLAGS['DRAW_STROKE'] = False
            FLAGS['DRAW_RECT'] = False
            FLAGS['rect_or_mask'] = -1
            FLAGS['rect_over'] = False
            FLAGS['value'] = DRAW_FG
            FLAGS['circSize'] = 2
            img = copy.deepcopy(img2)
            mask = np.zeros(img.shape[:2], dtype = np.uint8) 
            EventObj.image = img
            EventObj.mask = mask
            output = np.zeros(img.shape, np.uint8)
            fno=0
        
        elif k == 13:
            # Press carriage return to exit
            cv2.destroyAllWindows()
            return img, mask


        EventObj.flags = FLAGS
    
    return img, mask