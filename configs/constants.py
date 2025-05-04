import numpy as np

class INDEXING:
    ROOT_JNT_IDX=np.array([2])
    UNILATERAL_JNT_IDX=np.array([39, 43, 47, 51, 55])
    LEG_JNT_IDX=np.array([[7, 23], [11, 27]])
    FOOT_JNT_IDX=np.array([[15, 31], [19, 35]])
    BILATERAL_JNT_IDX=np.array([[59, 79], [63, 83], [67, 87], [71, 91], [75, 95]])

    ROOT_GEOM_IDX=np.array([1])
    UNILATERAL_GEOM_IDX=np.array([10, 11, 12, 13, 14])
    LEG_GEOM_IDX=np.array([[2, 6], [3, 7]])
    FOOT_GEOM_IDX=np.array([[4, 8], [5, 9]])
    BILATERAL_GEOM_IDX=np.array([[15, 20], [16, 21], [17, 22], [18, 23], [19, 24]])

    TRANSL_JNT_IDXS=np.array([7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95])
    ROT_JNT_IDX=np.array([8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 40, 41, 42, 44, 45, 46, 48, 49, 50, 52, 53, 54, 56, 57, 58, 60, 61, 62, 64, 65, 66, 68, 69, 70, 72, 73, 74, 76, 77, 78, 80, 81, 82, 84, 85, 86, 88, 89, 90, 92, 93, 94, 96, 97, 98])

    R6D_TRANSL_IDXS=np.array([9, 16, 23, 30, 37, 44, 51, 58, 65, 72, 79, 86, 93, 100, 107, 114, 121, 128, 135, 142, 149, 156, 163])
    R6D_ROT_IDXS=np.array([10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 91, 92, 94, 95, 96, 97, 98, 99, 101, 102, 103, 104, 105, 106, 108, 109, 110, 111, 112, 113, 115, 116, 117, 118, 119, 120, 122, 123, 124, 125, 126, 127, 129, 130, 131, 132, 133, 134, 136, 137, 138, 139, 140, 141, 143, 144, 145, 146, 147, 148, 150, 151, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 164, 165, 166, 167, 168, 169])

class RESHAPED_INDEXING:
    ROOT_JNT_IDX=np.array([0])
    UNILATERAL_JNT_IDX=np.array([9, 10, 11, 12, 13])
    LEG_JNT_IDX=np.array([[1, 5], [2, 6], [3, 7]])
    FOOT_JNT_IDX=np.array([[4, 8]])
    BILATERAL_JNT_IDX=np.array([[14, 19], [15, 20], [16, 21], [17, 22], [18, 23]])

class PARENTS:
    ROOT_JNT_IDX=np.array([0])
    UNILATERAL_JNT_IDX=np.array([1, 10, 11, 12, 13])
    LEG_JNT_IDX=np.array([[1, 1], [2, 6], [3, 7]])
    FOOT_JNT_IDX=np.array([[4, 8]])
    BILATERAL_JNT_IDX=np.array([[11, 11], [15, 20], [16, 21], [17, 22], [18, 23]])

class CHILDREN:
    DATA=[[1], 
          [2, 6, 10], 
          [3], 
          [4], 
          [5], 
          [-1], 
          [7], 
          [8], 
          [9], 
          [-1], 
          [11], 
          [12, 15, 20], 
          [13], 
          [14],
          [-1],
          [16],
          [17],
          [18],
          [19],
          [-1],
          [21],
          [22],
          [23],
          [24],
          [-1],]
    
class ROBOT:
    BODIES = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']

    FROM_SMPL_MAP={
                    0 : 8, 1 : 9, 2 : 7,
                    3 : 20, 4 : 21, 5 : 19,
                    6 : 32, 7 : 33, 8 : 31,
                    9 : 11, 10 : 12, 11 : 10,
                    12 : 23, 13 : 24, 14 : 22,
                    15 : 35, 16 : 36, 17 : 34,
                    18 : 14, 19 : 15, 20 : 13,
                    21 : 26, 22 : 27, 23 : 25,
                    24 : 38, 25 : 39, 26 : 37,
                    27 : 17, 28 : 18, 29 : 16,
                    30 : 29, 31 : 30, 32 : 28,
                    33 : 41, 34 : 42, 35 : 40,
                    36 : 47, 37 : 48, 38 : 46,
                    39 : 62, 40 : 63, 41 : 61,
                    42 : 44, 43 : 45, 44 : 43,
                    45 : 50, 46 : 51, 47 : 49,
                    48 : 65, 49 : 66, 50 : 64,
                    51 : 53, 52 : 54, 53 : 52,
                    54 : 68, 55 : 69, 56 : 67,
                    57 : 56, 58 : 57, 59 : 55,
                    60 : 71, 61 : 72, 62 : 70,
                    63 : 59, 64 : 60, 65 : 58,
                    66 : 74, 67 : 75, 68 : 73
                    }
    
    REVERSE_MAPPING = {v: k for k, v in sorted(FROM_SMPL_MAP.items(), key=lambda x: x[1])}
    
class SMPL:
    JOINTS = [
        'Pelvis', 'Left Hip', 'Right Hip', 'Spine 1 (Lower)', 'Left Knee',
        'Right Knee', 'Spine 2 (Middle)', 'Left Ankle', 'Right Ankle',
        'Spine 3 (Upper)', 'Left Foot', 'Right Foot', 'Neck',
        'Left Shoulder (Inner)', 'Right Shoulder (Inner)', 'Head',
        'Left Shoulder (Outer)', 'Right Shoulder (Outer)', 'Left Elbow',
        'Right Elbow', 'Left Wrist', 'Right Wrist', 'Left Hand', 'Right Hand']
    
    NAME_MAPPING = {
        # Previous Name       : New Name
        'Pelvis'             : 'Pelvis',
        'L_Hip'              : 'Left Hip',
        'R_Hip'              : 'Right Hip',
        'Torso'              : 'Spine 1 (Lower)',  # Closest match
        'L_Knee'             : 'Left Knee',
        'R_Knee'             : 'Right Knee',
        'Spine'              : 'Spine 2 (Middle)', # Closest match
        'L_Ankle'            : 'Left Ankle',
        'R_Ankle'            : 'Right Ankle',
        'Chest'              : 'Spine 3 (Upper)', # Closest match
        'L_Toe'              : 'Left Foot',       # 'Toe' → 'Foot'
        'R_Toe'              : 'Right Foot',      # 'Toe' → 'Foot'
        'Neck'               : 'Neck',
        'L_Thorax'           : 'Left Shoulder (Inner)',
        'R_Thorax'           : 'Right Shoulder (Inner)',
        'Head'               : 'Head',
        'L_Shoulder'         : 'Left Shoulder (Outer)',
        'R_Shoulder'         : 'Right Shoulder (Outer)',
        'L_Elbow'            : 'Left Elbow',
        'R_Elbow'            : 'Right Elbow',
        'L_Wrist'            : 'Left Wrist',
        'R_Wrist'            : 'Right Wrist',
        'L_Hand'             : 'Left Hand',
        'R_Hand'             : 'Right Hand'
    }

    REVERSE_MAPPING = {v: k for k, v in NAME_MAPPING.items()}

    NUMERICAL=[0, 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, 14, 19, 13, 15, 20, 16, 21, 17, 22, 18, 23]

    TO_ROBOT_MAPPING={}

class CAMERAS:
    FOCAL_LENGTH = 421.2
    CX = 424.0
    CY = 240.0

class CONTROL:
    GAINS = {
            "L_Hip":      [800, 80, 1, 1000],
            "L_Knee":     [800, 80, 1, 1000],
            "L_Ankle":    [800, 80, 1, 1000],
            "L_Toe":      [500, 50, 1, 500],
            "R_Hip":      [800, 80, 1, 1000],
            "R_Knee":     [800, 80, 1, 1000],
            "R_Ankle":    [800, 80, 1, 1000],
            "R_Toe":      [500, 50, 1, 500],
            "Torso":      [1000, 100, 1, 500],
            "Spine":      [1000, 100, 1, 500],
            "Chest":      [1000, 100, 1, 500],
            "Neck":       [500, 50, 1, 250],
            "Head":       [500, 50, 1, 250],
            "L_Thorax":   [500, 50, 1, 1000],
            "L_Shoulder": [500, 50, 1, 1000],
            "L_Elbow":    [500, 50, 1, 250],
            "L_Wrist":    [300, 30, 1, 250],
            "L_Hand":     [300, 30, 1, 250],
            "R_Thorax":   [500, 50, 1, 1000],
            "R_Shoulder": [500, 50, 1, 1000],
            "R_Elbow":    [500, 50, 1, 250],
            "R_Wrist":    [300, 30, 1, 250],
            "R_Hand":     [300, 30, 1, 250],
            }
