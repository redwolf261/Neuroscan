"""
Training Data Extractor and Graph Plotter
Extracts metrics from training logs and creates publication-quality graphs
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from datetime import datetime

# Training data extracted from your terminal output
# Epoch: Train Dice, Train Loss, Val Dice, Precision, Recall, F1, Time(s)
training_data = {
    'epoch': [],
    'train_dice': [],
    'train_loss': [],
    'val_dice': [],
    'precision': [],
    'recall': [],
    'f1_score': [],
    'time_seconds': []
}

# Your actual training data (epochs 1-200)
epochs_data = [
    # Epoch, Train_D, Train_Loss, Val_D, Prec, Rec, F1, Time
    (1, 0.2336, 0.7192, 0.2555, 0.5986, 0.8245, 0.6951, 89.5),
    (2, 0.2788, 0.6747, 0.2940, 0.5984, 0.8271, 0.6957, 87.1),
    (3, 0.3162, 0.6330, 0.3266, 0.5988, 0.8291, 0.6962, 86.9),
    (4, 0.3497, 0.5962, 0.3549, 0.5992, 0.8305, 0.6965, 86.9),
    (5, 0.3595, 0.5772, 0.3657, 0.5995, 0.8311, 0.6966, 87.2),
    (6, 0.3839, 0.5087, 0.3903, 0.5999, 0.8315, 0.6968, 86.6),
    (7, 0.4100, 0.4821, 0.4142, 0.5984, 0.8357, 0.6972, 82.2),
    (8, 0.4356, 0.4570, 0.4374, 0.5974, 0.8382, 0.6974, 82.4),
    (9, 0.4627, 0.4295, 0.4594, 0.5968, 0.8396, 0.6975, 84.7),
    (10, 0.4849, 0.4088, 0.4802, 0.5966, 0.8406, 0.6977, 85.6),
    (11, 0.5080, 0.3859, 0.4995, 0.5967, 0.8413, 0.6980, 85.8),
    (12, 0.5285, 0.3660, 0.5174, 0.5972, 0.8417, 0.6984, 85.8),
    (13, 0.5494, 0.3460, 0.5338, 0.5977, 0.8418, 0.6989, 83.8),
    (14, 0.5663, 0.3302, 0.5488, 0.5985, 0.8420, 0.6995, 82.6),
    (15, 0.5838, 0.3138, 0.5624, 0.5993, 0.8420, 0.7000, 82.2),
    (16, 0.5975, 0.3010, 0.5748, 0.6001, 0.8419, 0.7006, 86.4),
    (17, 0.6093, 0.2903, 0.5859, 0.6009, 0.8419, 0.7011, 84.9),
    (18, 0.6230, 0.2772, 0.5959, 0.6015, 0.8419, 0.7015, 85.2),
    (19, 0.6323, 0.2687, 0.6049, 0.6020, 0.8418, 0.7018, 85.8),
    (20, 0.6421, 0.2591, 0.6130, 0.6025, 0.8417, 0.7021, 85.2),
    (21, 0.6531, 0.2495, 0.6204, 0.6031, 0.8417, 0.7025, 85.2),
    (22, 0.6603, 0.2430, 0.6270, 0.6035, 0.8417, 0.7028, 85.7),
    (23, 0.6674, 0.2356, 0.6329, 0.6038, 0.8416, 0.7030, 85.3),
    (24, 0.6738, 0.2303, 0.6384, 0.6043, 0.8416, 0.7033, 84.9),
    (25, 0.6808, 0.2236, 0.6433, 0.6049, 0.8415, 0.7037, 85.3),
    (26, 0.6880, 0.2172, 0.6479, 0.6053, 0.8415, 0.7039, 85.0),
    (27, 0.6905, 0.2152, 0.6521, 0.6059, 0.8414, 0.7043, 81.7),
    (28, 0.6956, 0.2102, 0.6560, 0.6065, 0.8412, 0.7047, 82.6),
    (29, 0.7001, 0.2059, 0.6595, 0.6071, 0.8411, 0.7050, 85.3),
    (30, 0.7014, 0.2044, 0.6628, 0.6077, 0.8410, 0.7054, 84.8),
    (31, 0.7064, 0.2007, 0.6658, 0.6084, 0.8407, 0.7058, 85.1),
    (32, 0.7113, 0.1950, 0.6686, 0.6093, 0.8405, 0.7063, 85.0),
    (33, 0.7159, 0.1920, 0.6712, 0.6101, 0.8402, 0.7067, 84.5),
    (34, 0.7168, 0.1916, 0.6737, 0.6109, 0.8399, 0.7072, 84.5),
    (35, 0.7194, 0.1894, 0.6761, 0.6120, 0.8395, 0.7077, 84.3),
    (36, 0.7223, 0.1859, 0.6784, 0.6131, 0.8390, 0.7083, 85.4),
    (37, 0.7267, 0.1829, 0.6805, 0.6142, 0.8386, 0.7089, 85.4),
    (38, 0.7263, 0.1833, 0.6824, 0.6153, 0.8380, 0.7094, 86.3),
    (39, 0.7305, 0.1784, 0.6843, 0.6165, 0.8374, 0.7099, 84.2),
    (40, 0.7324, 0.1775, 0.6862, 0.6179, 0.8367, 0.7106, 86.0),
    (41, 0.7373, 0.1734, 0.6880, 0.6194, 0.8359, 0.7114, 85.0),
    (42, 0.7402, 0.1714, 0.6897, 0.6209, 0.8352, 0.7121, 85.6),
    (43, 0.7410, 0.1713, 0.6914, 0.6227, 0.8344, 0.7129, 85.1),
    (44, 0.7390, 0.1729, 0.6930, 0.6243, 0.8335, 0.7137, 84.2),
    (45, 0.7444, 0.1689, 0.6945, 0.6260, 0.8326, 0.7145, 85.4),
    (46, 0.7468, 0.1658, 0.6960, 0.6278, 0.8318, 0.7153, 84.0),
    (47, 0.7463, 0.1676, 0.6975, 0.6296, 0.8308, 0.7161, 83.0),
    (48, 0.7504, 0.1642, 0.6989, 0.6314, 0.8298, 0.7169, 87.3),
    (49, 0.7497, 0.1642, 0.7002, 0.6331, 0.8288, 0.7176, 85.5),
    (50, 0.7537, 0.1608, 0.7015, 0.6348, 0.8278, 0.7183, 84.5),
    (51, 0.7521, 0.1635, 0.7028, 0.6366, 0.8269, 0.7191, 85.3),
    (52, 0.7562, 0.1603, 0.7039, 0.6382, 0.8260, 0.7198, 84.1),
    (53, 0.7580, 0.1585, 0.7050, 0.6397, 0.8251, 0.7204, 83.7),
    (54, 0.7595, 0.1569, 0.7062, 0.6415, 0.8241, 0.7212, 82.8),
    (55, 0.7596, 0.1583, 0.7072, 0.6431, 0.8232, 0.7218, 83.0),
    (56, 0.7629, 0.1549, 0.7082, 0.6445, 0.8224, 0.7224, 82.5),
    (57, 0.7616, 0.1566, 0.7092, 0.6458, 0.8216, 0.7229, 82.4),
    (58, 0.7644, 0.1541, 0.7101, 0.6473, 0.8207, 0.7235, 83.3),
    (59, 0.7663, 0.1526, 0.7110, 0.6486, 0.8199, 0.7240, 82.4),
    (60, 0.7652, 0.1543, 0.7119, 0.6500, 0.8191, 0.7245, 83.8),
    (61, 0.7676, 0.1537, 0.7127, 0.6514, 0.8182, 0.7251, 82.5),
    (62, 0.7652, 0.1542, 0.7135, 0.6527, 0.8175, 0.7255, 83.7),
    (63, 0.7703, 0.1499, 0.7143, 0.6541, 0.8166, 0.7261, 80.1),
    (64, 0.7713, 0.1499, 0.7150, 0.6554, 0.8157, 0.7265, 80.2),
    (65, 0.7730, 0.1510, 0.7157, 0.6565, 0.8150, 0.7270, 84.0),
    (66, 0.7755, 0.1474, 0.7164, 0.6577, 0.8143, 0.7274, 82.9),
    (67, 0.7722, 0.1512, 0.7171, 0.6590, 0.8136, 0.7279, 80.0),
    (68, 0.7761, 0.1475, 0.7177, 0.6599, 0.8131, 0.7282, 80.5),
    (69, 0.7759, 0.1487, 0.7182, 0.6608, 0.8125, 0.7286, 80.4),
    (70, 0.7794, 0.1446, 0.7189, 0.6618, 0.8119, 0.7290, 84.2),
    (71, 0.7822, 0.1430, 0.7194, 0.6629, 0.8114, 0.7294, 83.0),
    (72, 0.7777, 0.1473, 0.7200, 0.6638, 0.8109, 0.7298, 83.7),
    (73, 0.7794, 0.1452, 0.7206, 0.6646, 0.8104, 0.7300, 82.0),
    (74, 0.7790, 0.1462, 0.7211, 0.6652, 0.8101, 0.7303, 82.2),
    (75, 0.7819, 0.1433, 0.7217, 0.6661, 0.8095, 0.7306, 82.1),
    (76, 0.7834, 0.1423, 0.7222, 0.6670, 0.8091, 0.7309, 83.7),
    (77, 0.7829, 0.1407, 0.7227, 0.6680, 0.8085, 0.7313, 83.5),
    (78, 0.7843, 0.1407, 0.7232, 0.6690, 0.8079, 0.7317, 83.2),
    (79, 0.7825, 0.1429, 0.7237, 0.6699, 0.8073, 0.7320, 82.4),
    (80, 0.7841, 0.1412, 0.7241, 0.6706, 0.8069, 0.7322, 88.9),
    (81, 0.7850, 0.1414, 0.7245, 0.6711, 0.8067, 0.7324, 84.3),
    (82, 0.7852, 0.1415, 0.7249, 0.6716, 0.8065, 0.7326, 83.4),
    (83, 0.7880, 0.1392, 0.7252, 0.6723, 0.8060, 0.7329, 85.8),
    (84, 0.7876, 0.1414, 0.7256, 0.6729, 0.8056, 0.7331, 89.9),
    (85, 0.7881, 0.1384, 0.7260, 0.6737, 0.8051, 0.7333, 83.5),
    (86, 0.7866, 0.1398, 0.7265, 0.6746, 0.8046, 0.7337, 82.1),
    (87, 0.7902, 0.1367, 0.7268, 0.6754, 0.8041, 0.7339, 83.7),
    (88, 0.7908, 0.1379, 0.7273, 0.6762, 0.8036, 0.7342, 82.7),
    (89, 0.7897, 0.1388, 0.7276, 0.6769, 0.8032, 0.7345, 82.2),
    (90, 0.7887, 0.1381, 0.7280, 0.6776, 0.8028, 0.7347, 82.0),
    (91, 0.7893, 0.1390, 0.7283, 0.6779, 0.8027, 0.7348, 84.7),
    (92, 0.7906, 0.1377, 0.7286, 0.6782, 0.8025, 0.7349, 84.1),
    (93, 0.7941, 0.1339, 0.7288, 0.6785, 0.8024, 0.7350, 83.5),
    (94, 0.7902, 0.1379, 0.7291, 0.6790, 0.8020, 0.7352, 82.1),
    (95, 0.7925, 0.1363, 0.7294, 0.6797, 0.8017, 0.7354, 84.5),
    (96, 0.7921, 0.1360, 0.7298, 0.6803, 0.8012, 0.7356, 120.0),
    (97, 0.7941, 0.1367, 0.7301, 0.6810, 0.8007, 0.7358, 103.2),
    (98, 0.7924, 0.1368, 0.7304, 0.6817, 0.8002, 0.7360, 79.8),
    (99, 0.7932, 0.1365, 0.7307, 0.6823, 0.7998, 0.7362, 79.9),
    (100, 0.7907, 0.1394, 0.7309, 0.6826, 0.7996, 0.7363, 80.6),
    (101, 0.7977, 0.1324, 0.7311, 0.6829, 0.7995, 0.7365, 79.4),
    (102, 0.7916, 0.1377, 0.7313, 0.6831, 0.7994, 0.7365, 81.5),
    (103, 0.7958, 0.1347, 0.7316, 0.6836, 0.7991, 0.7367, 82.7),
    (104, 0.7946, 0.1361, 0.7318, 0.6840, 0.7989, 0.7368, 81.8),
    (105, 0.7959, 0.1336, 0.7320, 0.6843, 0.7988, 0.7369, 82.4),
    (106, 0.7981, 0.1319, 0.7322, 0.6847, 0.7985, 0.7370, 82.9),
    (107, 0.7993, 0.1308, 0.7325, 0.6855, 0.7980, 0.7373, 82.9),
    (108, 0.7983, 0.1334, 0.7328, 0.6863, 0.7975, 0.7376, 82.6),
    (109, 0.8002, 0.1322, 0.7330, 0.6870, 0.7970, 0.7378, 82.5),
    (110, 0.7989, 0.1328, 0.7332, 0.6876, 0.7967, 0.7379, 79.6),
    (111, 0.7974, 0.1345, 0.7335, 0.6882, 0.7963, 0.7381, 80.0),
    (112, 0.8016, 0.1313, 0.7337, 0.6886, 0.7961, 0.7383, 79.7),
    (113, 0.7980, 0.1339, 0.7339, 0.6891, 0.7959, 0.7385, 81.8),
    (114, 0.8022, 0.1298, 0.7341, 0.6896, 0.7957, 0.7387, 79.5),
    (115, 0.8000, 0.1319, 0.7343, 0.6896, 0.7959, 0.7388, 79.4),
    (116, 0.7998, 0.1314, 0.7345, 0.6900, 0.7957, 0.7389, 81.5),
    (117, 0.8034, 0.1295, 0.7347, 0.6903, 0.7955, 0.7390, 82.5),
    (118, 0.8004, 0.1327, 0.7349, 0.6907, 0.7953, 0.7392, 80.0),
    (119, 0.8022, 0.1290, 0.7351, 0.6912, 0.7950, 0.7393, 80.2),
    (120, 0.8047, 0.1277, 0.7353, 0.6917, 0.7947, 0.7395, 81.9),
    (121, 0.8029, 0.1312, 0.7355, 0.6923, 0.7944, 0.7397, 79.3),
    (122, 0.8028, 0.1300, 0.7357, 0.6929, 0.7939, 0.7398, 79.8),
    (123, 0.8051, 0.1288, 0.7360, 0.6935, 0.7935, 0.7400, 79.4),
    (124, 0.8031, 0.1302, 0.7362, 0.6943, 0.7931, 0.7403, 80.7),
    (125, 0.8046, 0.1305, 0.7364, 0.6950, 0.7926, 0.7405, 80.0),
    (126, 0.8035, 0.1306, 0.7367, 0.6958, 0.7922, 0.7407, 85.8),
    (127, 0.8043, 0.1305, 0.7369, 0.6962, 0.7920, 0.7409, 80.3),
    (128, 0.8048, 0.1287, 0.7371, 0.6968, 0.7916, 0.7410, 79.8),
    (129, 0.8058, 0.1284, 0.7372, 0.6968, 0.7917, 0.7411, 80.2),
    (130, 0.8059, 0.1286, 0.7373, 0.6969, 0.7917, 0.7411, 82.9),
    (131, 0.8044, 0.1295, 0.7374, 0.6971, 0.7917, 0.7412, 80.6),
    (132, 0.8070, 0.1275, 0.7375, 0.6970, 0.7919, 0.7413, 79.8),
    (133, 0.8065, 0.1297, 0.7377, 0.6972, 0.7918, 0.7414, 81.1),
    (134, 0.8056, 0.1284, 0.7378, 0.6975, 0.7916, 0.7415, 80.4),
    (135, 0.8062, 0.1283, 0.7380, 0.6979, 0.7915, 0.7416, 80.5),
    (136, 0.8087, 0.1267, 0.7381, 0.6982, 0.7913, 0.7417, 80.2),
    (137, 0.8101, 0.1248, 0.7383, 0.6984, 0.7912, 0.7418, 80.0),
    (138, 0.8064, 0.1274, 0.7385, 0.6989, 0.7908, 0.7419, 79.8),
    (139, 0.8097, 0.1267, 0.7386, 0.6993, 0.7906, 0.7420, 81.8),
    (140, 0.8075, 0.1268, 0.7388, 0.6998, 0.7902, 0.7422, 80.6),
    (141, 0.8112, 0.1241, 0.7390, 0.7004, 0.7898, 0.7423, 80.2),
    (142, 0.8092, 0.1270, 0.7391, 0.7006, 0.7898, 0.7424, 80.9),
    (143, 0.8110, 0.1258, 0.7392, 0.7007, 0.7899, 0.7425, 80.2),
    (144, 0.8085, 0.1286, 0.7393, 0.7011, 0.7897, 0.7426, 82.3),
    (145, 0.8102, 0.1253, 0.7395, 0.7015, 0.7894, 0.7428, 80.4),
    (146, 0.8097, 0.1259, 0.7397, 0.7018, 0.7893, 0.7429, 82.5),
    (147, 0.8136, 0.1229, 0.7398, 0.7023, 0.7891, 0.7430, 92.1),
    (148, 0.8078, 0.1258, 0.7400, 0.7027, 0.7889, 0.7432, 82.2),
    (149, 0.8094, 0.1270, 0.7401, 0.7030, 0.7887, 0.7433, 83.8),
    (150, 0.8109, 0.1245, 0.7403, 0.7033, 0.7885, 0.7434, 82.9),
    (151, 0.8039, 0.1309, 0.7404, 0.7036, 0.7883, 0.7435, 83.7),
    (152, 0.8102, 0.1260, 0.7406, 0.7041, 0.7880, 0.7436, 82.8),
    (153, 0.8082, 0.1262, 0.7407, 0.7045, 0.7878, 0.7437, 82.6),
    (154, 0.8075, 0.1301, 0.7408, 0.7048, 0.7876, 0.7438, 83.5),
    (155, 0.8125, 0.1259, 0.7409, 0.7050, 0.7876, 0.7439, 83.0),
    (156, 0.8131, 0.1229, 0.7411, 0.7051, 0.7875, 0.7440, 82.6),
    (157, 0.8125, 0.1261, 0.7412, 0.7055, 0.7873, 0.7441, 82.3),
    (158, 0.8103, 0.1259, 0.7413, 0.7058, 0.7873, 0.7442, 83.3),
    (159, 0.8116, 0.1271, 0.7414, 0.7059, 0.7873, 0.7443, 83.7),
    (160, 0.8146, 0.1225, 0.7415, 0.7059, 0.7874, 0.7444, 82.6),
    (161, 0.8112, 0.1265, 0.7416, 0.7062, 0.7873, 0.7445, 83.2),
    (162, 0.8097, 0.1271, 0.7417, 0.7066, 0.7872, 0.7446, 81.6),
    (163, 0.8138, 0.1240, 0.7418, 0.7067, 0.7872, 0.7447, 83.1),
    (164, 0.8103, 0.1265, 0.7419, 0.7068, 0.7873, 0.7448, 82.8),
    (165, 0.8111, 0.1265, 0.7420, 0.7070, 0.7871, 0.7448, 82.6),
    (166, 0.8124, 0.1246, 0.7421, 0.7071, 0.7871, 0.7449, 90.2),
    (167, 0.8125, 0.1231, 0.7422, 0.7075, 0.7868, 0.7450, 83.2),
    (168, 0.8156, 0.1236, 0.7423, 0.7077, 0.7867, 0.7451, 84.0),
    (169, 0.8166, 0.1209, 0.7424, 0.7077, 0.7868, 0.7451, 82.6),
    (170, 0.8128, 0.1244, 0.7425, 0.7081, 0.7865, 0.7452, 82.7),
    (171, 0.8166, 0.1232, 0.7426, 0.7084, 0.7864, 0.7453, 91.9),
    (172, 0.8133, 0.1239, 0.7427, 0.7087, 0.7863, 0.7454, 80.2),
    (173, 0.8128, 0.1254, 0.7428, 0.7087, 0.7863, 0.7454, 80.0),
    (174, 0.8160, 0.1242, 0.7429, 0.7089, 0.7862, 0.7455, 82.5),
    (175, 0.8123, 0.1236, 0.7430, 0.7090, 0.7862, 0.7455, 83.9),
    (176, 0.8134, 0.1271, 0.7431, 0.7093, 0.7861, 0.7456, 80.3),
    (177, 0.8165, 0.1228, 0.7432, 0.7093, 0.7862, 0.7457, 79.9),
    (178, 0.8170, 0.1223, 0.7433, 0.7097, 0.7859, 0.7458, 80.5),
    (179, 0.8133, 0.1232, 0.7434, 0.7102, 0.7856, 0.7459, 83.4),
    (180, 0.8148, 0.1234, 0.7435, 0.7103, 0.7855, 0.7459, 80.0),
    (181, 0.8141, 0.1233, 0.7436, 0.7107, 0.7852, 0.7460, 80.4),
    (182, 0.8138, 0.1227, 0.7437, 0.7110, 0.7850, 0.7461, 81.6),
    (183, 0.8147, 0.1232, 0.7438, 0.7112, 0.7849, 0.7462, 83.0),
    (184, 0.8130, 0.1257, 0.7438, 0.7112, 0.7849, 0.7462, 82.7),
    (185, 0.8157, 0.1237, 0.7439, 0.7114, 0.7848, 0.7463, 84.0),
    (186, 0.8163, 0.1218, 0.7440, 0.7115, 0.7847, 0.7463, 82.4),
    (187, 0.8166, 0.1221, 0.7441, 0.7119, 0.7844, 0.7464, 84.1),
    (188, 0.8167, 0.1235, 0.7442, 0.7120, 0.7844, 0.7464, 83.9),
    (189, 0.8153, 0.1220, 0.7442, 0.7121, 0.7845, 0.7465, 85.0),
    (190, 0.8208, 0.1199, 0.7443, 0.7123, 0.7844, 0.7466, 84.6),
    (191, 0.8140, 0.1267, 0.7444, 0.7127, 0.7841, 0.7466, 84.7),
    (192, 0.8129, 0.1248, 0.7445, 0.7130, 0.7838, 0.7467, 84.1),
    (193, 0.8145, 0.1245, 0.7445, 0.7132, 0.7836, 0.7467, 85.5),
    (194, 0.8138, 0.1246, 0.7446, 0.7133, 0.7836, 0.7467, 86.6),
    (195, 0.8194, 0.1208, 0.7447, 0.7135, 0.7836, 0.7468, 83.4),
    (196, 0.8148, 0.1236, 0.7447, 0.7136, 0.7835, 0.7469, 82.7),
    (197, 0.8185, 0.1228, 0.7448, 0.7137, 0.7836, 0.7469, 83.5),
    (198, 0.8189, 0.1221, 0.7449, 0.7140, 0.7834, 0.7470, 84.8),
    (199, 0.8193, 0.1191, 0.7450, 0.7142, 0.7832, 0.7471, 82.8),
    (200, 0.8174, 0.1204, 0.7451, 0.7143, 0.7834, 0.7472, 83.0),
]

# Populate the dictionary
for epoch_data in epochs_data:
    training_data['epoch'].append(epoch_data[0])
    training_data['train_dice'].append(epoch_data[1])
    training_data['train_loss'].append(epoch_data[2])
    training_data['val_dice'].append(epoch_data[3])
    training_data['precision'].append(epoch_data[4])
    training_data['recall'].append(epoch_data[5])
    training_data['f1_score'].append(epoch_data[6])
    training_data['time_seconds'].append(epoch_data[7])

# Create DataFrame
df = pd.DataFrame(training_data)

# Save to CSV for easy access
df.to_csv('training_metrics.csv', index=False)
print("✅ Saved training data to: training_metrics.csv")

# Save to JSON as well
with open('training_metrics.json', 'w') as f:
    json.dump(training_data, f, indent=2)
print("✅ Saved training data to: training_metrics.json")

# Calculate statistics
print("\n" + "="*70)
print("📊 TRAINING STATISTICS")
print("="*70)
print(f"Total epochs: {len(df)}")
print(f"Best Val Dice: {df['val_dice'].max():.4f} (Epoch {df.loc[df['val_dice'].idxmax(), 'epoch']:.0f})")
print(f"Final Val Dice: {df['val_dice'].iloc[-1]:.4f}")
print(f"Improvement: {df['val_dice'].iloc[-1] - df['val_dice'].iloc[0]:.4f} ({((df['val_dice'].iloc[-1] / df['val_dice'].iloc[0]) - 1) * 100:.1f}%)")
print(f"\nAverage epoch time: {df['time_seconds'].mean():.1f}s")
print(f"Fastest epoch: {df['time_seconds'].min():.1f}s")
print(f"Slowest epoch: {df['time_seconds'].max():.1f}s")
print(f"Total training time: {df['time_seconds'].sum() / 3600:.2f} hours")

# Create publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(16, 12))

# 1. Dice Score Progress (Train vs Val)
ax1 = plt.subplot(2, 3, 1)
ax1.plot(df['epoch'], df['train_dice'], 'b-', linewidth=2, label='Training Dice', alpha=0.8)
ax1.plot(df['epoch'], df['val_dice'], 'r-', linewidth=2, label='Validation Dice', alpha=0.8)
ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Dice Score', fontsize=12, fontweight='bold')
ax1.set_title('Dice Score Progression (PATH 3 BALANCED)', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 200)
ax1.set_ylim(0.2, 0.85)

# 2. Training Loss
ax2 = plt.subplot(2, 3, 2)
ax2.plot(df['epoch'], df['train_loss'], 'g-', linewidth=2, label='Training Loss', alpha=0.8)
ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax2.set_title('Training Loss Reduction', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 200)

# 3. Precision, Recall, F1 Score
ax3 = plt.subplot(2, 3, 3)
ax3.plot(df['epoch'], df['precision'], 'c-', linewidth=2, label='Precision', alpha=0.8)
ax3.plot(df['epoch'], df['recall'], 'm-', linewidth=2, label='Recall', alpha=0.8)
ax3.plot(df['epoch'], df['f1_score'], 'y-', linewidth=2, label='F1 Score', alpha=0.8)
ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
ax3.set_title('Classification Metrics', fontsize=14, fontweight='bold')
ax3.legend(loc='lower right', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 200)
ax3.set_ylim(0.55, 0.85)

# 4. Validation Dice with Milestones
ax4 = plt.subplot(2, 3, 4)
ax4.plot(df['epoch'], df['val_dice'], 'r-', linewidth=2.5, label='Validation Dice', alpha=0.9)
# Mark milestones
milestones = [50, 100, 150, 200]
for m in milestones:
    if m <= len(df):
        dice_val = df.loc[df['epoch'] == m, 'val_dice'].values[0]
        ax4.axvline(x=m, color='gray', linestyle='--', alpha=0.3)
        ax4.plot(m, dice_val, 'ro', markersize=8)
        ax4.text(m, dice_val + 0.01, f'{dice_val:.3f}', ha='center', fontsize=9)
ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax4.set_ylabel('Validation Dice Score', fontsize=12, fontweight='bold')
ax4.set_title('Validation Dice with Milestones', fontsize=14, fontweight='bold')
ax4.legend(loc='lower right', fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 200)
ax4.set_ylim(0.25, 0.76)

# 5. Training Time per Epoch
ax5 = plt.subplot(2, 3, 5)
ax5.plot(df['epoch'], df['time_seconds'], 'orange', linewidth=1.5, alpha=0.7)
ax5.axhline(y=df['time_seconds'].mean(), color='r', linestyle='--', 
            label=f'Mean: {df["time_seconds"].mean():.1f}s', linewidth=2)
ax5.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax5.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
ax5.set_title('Training Time per Epoch', fontsize=14, fontweight='bold')
ax5.legend(loc='upper right', fontsize=10)
ax5.grid(True, alpha=0.3)
ax5.set_xlim(0, 200)

# 6. Improvement Rate (Dice per epoch)
ax6 = plt.subplot(2, 3, 6)
dice_improvement = df['val_dice'].diff().rolling(window=10).mean()
ax6.plot(df['epoch'][10:], dice_improvement[10:], 'purple', linewidth=2, alpha=0.8)
ax6.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax6.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax6.set_ylabel('Dice Improvement (10-epoch avg)', fontsize=12, fontweight='bold')
ax6.set_title('Learning Rate (Dice Improvement)', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.set_xlim(0, 200)

plt.tight_layout()
plt.savefig('training_graphs_full.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved comprehensive graph to: training_graphs_full.png")

# Create individual high-resolution plots for paper
# Individual Plot 1: Dice Score (larger, cleaner)
fig1, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['epoch'], df['train_dice'], 'b-', linewidth=2.5, label='Training Dice', alpha=0.8)
ax.plot(df['epoch'], df['val_dice'], 'r-', linewidth=2.5, label='Validation Dice', alpha=0.8)
ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Dice Score', fontsize=14, fontweight='bold')
ax.set_title('Dice Score Progression - PATH 3 BALANCED Configuration', fontsize=16, fontweight='bold')
ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, 200)
ax.set_ylim(0.2, 0.85)
plt.tight_layout()
plt.savefig('paper_dice_score.png', dpi=300, bbox_inches='tight')
print("✅ Saved paper-ready Dice plot to: paper_dice_score.png")

# Individual Plot 2: All Metrics Combined
fig2, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['epoch'], df['val_dice'], 'r-', linewidth=2.5, label='Dice Score', alpha=0.9)
ax.plot(df['epoch'], df['precision'], 'c-', linewidth=2.5, label='Precision', alpha=0.9)
ax.plot(df['epoch'], df['recall'], 'm-', linewidth=2.5, label='Recall', alpha=0.9)
ax.plot(df['epoch'], df['f1_score'], 'y-', linewidth=2.5, label='F1 Score', alpha=0.9)
ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Score', fontsize=14, fontweight='bold')
ax.set_title('Comprehensive Evaluation Metrics - PATH 3 BALANCED', fontsize=16, fontweight='bold')
ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, 200)
ax.set_ylim(0.25, 0.85)
plt.tight_layout()
plt.savefig('paper_all_metrics.png', dpi=300, bbox_inches='tight')
print("✅ Saved paper-ready metrics plot to: paper_all_metrics.png")

# Individual Plot 3: Loss Curve
fig3, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['epoch'], df['train_loss'], 'g-', linewidth=2.5, label='Training Loss', alpha=0.9)
ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Loss', fontsize=14, fontweight='bold')
ax.set_title('Training Loss Reduction - Hybrid Loss Function', fontsize=16, fontweight='bold')
ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, 200)
plt.tight_layout()
plt.savefig('paper_loss_curve.png', dpi=300, bbox_inches='tight')
print("✅ Saved paper-ready loss plot to: paper_loss_curve.png")

# Create summary statistics table
summary_stats = pd.DataFrame({
    'Metric': ['Initial Val Dice', 'Final Val Dice', 'Best Val Dice', 'Improvement', 
               'Final Precision', 'Final Recall', 'Final F1 Score', 
               'Avg Epoch Time', 'Total Training Time'],
    'Value': [
        f"{df['val_dice'].iloc[0]:.4f}",
        f"{df['val_dice'].iloc[-1]:.4f}",
        f"{df['val_dice'].max():.4f}",
        f"+{df['val_dice'].iloc[-1] - df['val_dice'].iloc[0]:.4f} ({((df['val_dice'].iloc[-1] / df['val_dice'].iloc[0]) - 1) * 100:.1f}%)",
        f"{df['precision'].iloc[-1]:.4f}",
        f"{df['recall'].iloc[-1]:.4f}",
        f"{df['f1_score'].iloc[-1]:.4f}",
        f"{df['time_seconds'].mean():.1f}s",
        f"{df['time_seconds'].sum() / 3600:.2f} hours"
    ]
})

summary_stats.to_csv('training_summary.csv', index=False)
print("✅ Saved training summary to: training_summary.csv")

print("\n" + "="*70)
print("📊 ALL FILES CREATED SUCCESSFULLY!")
print("="*70)
print("\nFiles created:")
print("  1. training_metrics.csv - Full data table")
print("  2. training_metrics.json - JSON format")
print("  3. training_summary.csv - Summary statistics")
print("  4. training_graphs_full.png - Comprehensive 6-panel figure")
print("  5. paper_dice_score.png - Publication-ready Dice plot")
print("  6. paper_all_metrics.png - Publication-ready metrics plot")
print("  7. paper_loss_curve.png - Publication-ready loss plot")
print("\n🎯 All graphs are saved at 300 DPI for publication quality!")
print("="*70)

# Display summary
print("\n" + "="*70)
print("📈 QUICK SUMMARY FOR RESEARCH PAPER")
print("="*70)
print(f"Configuration: PATH 3 BALANCED")
print(f"  - Spatial Size: (64, 64, 64)")
print(f"  - Batch Size: 3 (effective: 6 with gradient accumulation)")
print(f"  - Model: HybridMiniSwin3D (embed_dim=112, depth=4, heads=7)")
print(f"  - Loss: Hybrid (70% Focal Tversky + 30% Dice)")
print(f"\nPerformance:")
print(f"  - Initial Validation Dice: {df['val_dice'].iloc[0]:.4f}")
print(f"  - Final Validation Dice: {df['val_dice'].iloc[-1]:.4f}")
print(f"  - Best Validation Dice: {df['val_dice'].max():.4f} (Epoch {df.loc[df['val_dice'].idxmax(), 'epoch']:.0f})")
print(f"  - Total Improvement: +{df['val_dice'].iloc[-1] - df['val_dice'].iloc[0]:.4f} ({((df['val_dice'].iloc[-1] / df['val_dice'].iloc[0]) - 1) * 100:.1f}%)")
print(f"\nFinal Metrics (Epoch 200):")
print(f"  - Precision: {df['precision'].iloc[-1]:.4f}")
print(f"  - Recall: {df['recall'].iloc[-1]:.4f}")
print(f"  - F1 Score: {df['f1_score'].iloc[-1]:.4f}")
print(f"\nTraining Efficiency:")
print(f"  - Average Time per Epoch: {df['time_seconds'].mean():.1f} seconds")
print(f"  - Total Training Time: {df['time_seconds'].sum() / 3600:.2f} hours")
print(f"  - Speedup vs Original: ~10-15x faster")
print("="*70)
