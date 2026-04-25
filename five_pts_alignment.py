from face_align import crop_5pts_vipl_256   # import the function crop_align

def face_align_crop(image, keypoints):
    '''
    image: numpy.ndarray, uint8, [0,255], RGB, (H,W,3), 读取后的原始图片
    keypoints: 5x2 np.array,
    人脸关键点坐标, each row is a pair of coordinates (x, y)
    关键点坐标依次为：左眼中心，右眼中心，鼻尖，左嘴角， 右嘴角
    '''

    keypoints = keypoints.astype(int)
    aligned_frame = crop_5pts_vipl_256.crop_align(image, keypoints)
    return aligned_frame #return the numpy array of the aligned face
