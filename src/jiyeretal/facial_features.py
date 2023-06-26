import numpy as np

d = np.linalg.norm


class FacialRatio:
    def __init__(self, l: np.ndarray) -> None:
        """
        l: landmarks
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            self.under_eyes_interocular = d(l[48] - l[56]) / d(l[42] - l[54])
            self.under_eyes_nose_width = d(l[48] - l[56]) / d(l[64] - l[58])
            self.mouth_width_interocular = d(l[79] - l[85]) / d(l[42] - l[54])
            self.upper_lip_jaw_interocular = d(l[76] - l[11]) / d(l[42] - l[54])
            self.upper_lip_jaw_nose_width = d(l[76] - l[11]) / d(l[64] - l[58])
            self.interocular_lip_height = d(l[42] - l[54]) / d(l[76] - l[82])
            self.nose_width_interocular = d(l[64] - l[68]) / d(l[42] - l[54])
            self.nose_width_upper_lip_height = d(l[64] - l[68]) / d(l[76] - l[83]) / 2
            self.interocular_nose_mouth_height = d(l[42] - l[54]) / d(l[66] - l[76])
            self.face_top_eyebrows_eyebrows_nose = d(
                l[0] - d(l[22] - l[36]) / l[1]
            ) / d(d(l[22] - l[36]) / l[1] - l[66])
            self.eyebrows_nose_nose_jaw = d(d(l[22] - l[36]) / l[1] - l[66]) / d(
                l[66] - l[11]
            )
            self.face_top_eyebrows_nose_jaw = d(l[0] - d(l[22] - l[36]) / l[1]) / d(
                l[66] - l[11]
            )
            self.interocular_nose_width = d(l[42] - l[54]) / d(l[64] - l[68])
            self.face_height_face_width = d(l[0] - l[11]) / d(l[6] - l[16])

    def to_numpy(self) -> np.ndarray:
        return np.array(
            [
                self.under_eyes_interocular,
                self.under_eyes_nose_width,
                self.mouth_width_interocular,
                self.upper_lip_jaw_interocular,
                self.upper_lip_jaw_nose_width,
                self.interocular_lip_height,
                self.nose_width_interocular,
                self.nose_width_upper_lip_height,
                self.interocular_nose_mouth_height,
                self.face_top_eyebrows_eyebrows_nose,
                self.eyebrows_nose_nose_jaw,
                self.face_top_eyebrows_nose_jaw,
                self.interocular_nose_width,
                self.face_height_face_width,
            ]
        )

    def __str__(self):
        return """
        under_eyes_interocular          : {}
        under_eyes_nose_width           : {}
        mouth_width_interocular         : {}
        upper_lip_jaw_interocular       : {}    
        upper_lip_jaw_nose_width        : {}    
        interocular_lip_height          : {}
        nose_width_interocular          : {}
        nose_width_upper_lip_height     : {}    
        interocular_nose_mouth_height   : {}        
        face_top_eyebrows_eyebrows_nose : {}        
        eyebrows_nose_nose_jaw          : {}
        face_top_eyebrows_nose_jaw      : {}    
        interocular_nose_width          : {}
        face_height_face_width          : {}
        """.format(
            self.under_eyes_interocular,
            self.under_eyes_nose_width,
            self.mouth_width_interocular,
            self.upper_lip_jaw_interocular,
            self.upper_lip_jaw_nose_width,
            self.interocular_lip_height,
            self.nose_width_interocular,
            self.nose_width_upper_lip_height,
            self.interocular_nose_mouth_height,
            self.face_top_eyebrows_eyebrows_nose,
            self.eyebrows_nose_nose_jaw,
            self.face_top_eyebrows_nose_jaw,
            self.interocular_nose_width,
            self.face_height_face_width,
        )


class SymmetriRatios:
    def __init__(self, l: np.ndarray) -> None:
        """
        l: landmarks
        """
        with np.errstate(divide="ignore"):
            self.lower_eyebrow_length = d(l[26] - l[31]) / d(l[37] - l[41])
            self.lower_lip_length = d(l[79] - l[83]) / d(l[73] - l[83])
            self.upper_eyebrow = d(l[22] - d(l[22] - l[36]) / l[1]) / d(
                l[34] - d(l[22] - l[36]) / 2
            )
            self.upper_lip = d(l[79] - l[76]) / d(l[73] - l[76])
            self.nose = d(l[64] - l[66]) / d(l[68] - l[66])

    def to_numpy(self) -> np.ndarray:
        return np.array(
            [
                self.lower_eyebrow_length,
                self.lower_lip_length,
                self.upper_eyebrow,
                self.upper_lip,
                self.nose,
            ]
        )

    def __str__(self):
        return """
        lower_eyebrow_length: {} 
        lower_lip_length    : {} 
        upper_eyebrow       : {} 
        upper_lip           : {} 
        nose                : {}""".format(
            self.lower_eyebrow_length,
            self.lower_lip_length,
            self.upper_eyebrow,
            self.upper_lip,
            self.nose,
        )
