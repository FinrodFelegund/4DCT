
TEST_ROI_CENTERS = {
    'case_01': (44, 231, 244),
    'case_02': (56, 231, 230),
    'case_03': (52, 224, 220),
    'case_04': (46, 238, 222),
    'case_05': (53, 233, 206),
    'case_06': (64, 210, 247),
    'case_07': (66, 220, 233),
    'case_08': (57, 247, 279),
    'case_09': (80, 233, 232),
    'case_10': (77, 232, 229),
}

CALIBRATION_ROI_CENTERS = {
    '022_4DCT_Lunge_amplitudebased_complete': (65, 230, 223),
    '024_4DCT_Lunge_amplitudebased_complete': (66, 238, 236),
    '032_4DCT_Lunge_amplitudebased_complete': (54, 227, 206),
    '033_4DCT_Lunge_amplitudebased_complete': (61, 224, 204),
    '068_4DCT_Lunge_amplitudebased_complete': (62, 233, 220),
    '074_4DCT_Lunge_amplitudebased_complete': (68, 233, 218),
    '078_4DCT_Lunge_amplitudebased_complete': (61, 229, 220),
    '091_4DCT_Lunge_amplitudebased_complete': (64, 222, 213), 
    '092_4DCT_Lunge_amplitudebased_complete': (63, 232, 204), 
    '104_4DCT_Lunge_amplitudebased_complete': (66, 246, 189), 
    '106_4DCT_Lunge_amplitudebased_complete': (58, 230, 219), 
    '109_4DCT_Lunge_amplitudebased_complete': (50, 230, 202), 
    '115_4DCT_Lunge_amplitudebased_complete': (49, 270, 176), 
    '116_4DCT_Lunge_amplitudebased_complete': (64, 224, 202), 
    '121_4DCT_Lunge_amplitudebased_complete': (63, 224, 226), 
    '124_4DCT_Lunge_amplitudebased_complete': (57, 228, 179), 
    '132_4DCT_Lunge_amplitudebased_complete': (62, 240, 213), 
    '142_4DCT_Lunge_amplitudebased_complete': (54, 238, 229), 
    '145_4DCT_Lunge_amplitudebased_complete': (61, 235, 206), 
    '146_4DCT_Lunge_amplitudebased_complete': (70, 230, 213),
    '520_Lunge_amplitudebased': (88, 248, 237),
    '522_Lunge_amplitudebased': (83, 245, 231),
    '523_Lunge_amplitudebased': (70, 245, 230),
    '525_Lunge_amplitudebased': (67, 243, 222),
    '528_Lunge_amplitudebased': (79, 227, 220),
    '532_Lunge_amplitudebased': (74, 239, 224),
    '536_Lunge_amplitudebased': (142, 214, 216),
    '540_Lunge_amplitudebased': (78, 230, 217),
    '542_Lunge_amplitudebased': (80, 232, 220),
    '543_Lunge_amplitudebased': (73, 226, 225),
    '544_Lunge_amplitudebased': (75, 237, 225),
    '545_Lunge_amplitudebased': (89, 227, 205),
    '546_Lunge_amplitudebased': (89, 252, 221),
    '547_Lunge_amplitudebased': (69, 233, 230),
    '550_Lunge_amplitudebased': (76, 245, 220),
    '551_Lunge_amplitudebased': (74, 222, 216),
    '552_Lunge_amplitudebased': (67, 237, 206),
    '553_Lunge_amplitudebased': (76, 219, 205),
    '555_Lunge_amplitudebased': (79, 237, 214),
    '561_Lunge_amplitudebased': (69, 237, 230),
    '562_Lunge_amplitudebased': (83, 229, 203),
    '563_Lunge_amplitudebased': (67, 241, 220),
}

def roi_from_centre(d, r, c, size_z=8, size_xy=12):
    return (
        d - size_z // 2, d - size_z // 2 + size_z,
        r - size_xy // 2, r - size_xy // 2 + size_xy,
        c - size_xy // 2, c - size_xy // 2 + size_xy
    )

AORTA_ROI = {s: roi_from_centre(*c) for s, c in TEST_ROI_CENTERS.items()}
CALIBRATION_ROI = {s: roi_from_centre(*c) for s, c in CALIBRATION_ROI_CENTERS.items()}