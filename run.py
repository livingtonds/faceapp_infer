import cv2
from model import AttractiveModel


if __name__ == "__main__":
    attractive_model = AttractiveModel(model_a_path="models/model_a.pt",
                                       model_b_path="models/model_b.pt")
    
    good_img_path = "ava.jpg"
    good_image = cv2.imread(good_img_path)
    
    bad_img_path = "bad.jpg"
    bad_image = cv2.imread(bad_img_path)

    print("good_image res: ", attractive_model.get_score_from_image(good_image))
    print("bad_image res: ", attractive_model.get_score_from_image(bad_image))