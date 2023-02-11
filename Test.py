import Classifier_Model
import Test_Images

def Test_is_a_one():
    # given
    test_img = Test_Images.Test_Image.is_one()
    # when
    model = Classifier_Model.model
    pred = model.predict(test_img)
    #then
    assert pred == '1'
def Test_is_not_one():
    # given
    test_img = Test_Images.Test_Image.is_not_one()
    # when
    model = Classifier_Model.model
    pred = model.predict(test_img)
    #then
    assert pred == '0'
