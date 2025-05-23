import numpy as np

BOX_COLORS = list(np.array([(0, 169, 252), (255, 131, 0), (255, 0, 0)]) / 255)  # blue, orange, red
BOX_COLORS_DARK = list(np.array([(0, 109, 163), (207, 107, 0), (207, 0, 0)]) / 255)
# BOX_COLORS = list(np.array([(111, 255, 0), (255, 131, 0), (255, 0, 0)]) / 255)  # green, orange, red
# BOX_COLORS_DARK = list(np.array([(60, 138, 0), (207, 107, 0), (207, 0, 0)]) / 255)

BOX_LABELS = ['fast','medium','slow']
BOX_POSITIONS = ['S', 'NE', 'NW']
BOX_POSITIONS_ORDER = [2, 0, 1] # NW S NE ordering