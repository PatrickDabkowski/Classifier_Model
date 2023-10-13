import Classifier_Model
import Test_Images
import argparse

def Test_is_a_one(model):
    # given
    test_img = Test_Images.Test_Image.is_one()
    # when
    pred = model.predict(test_img.reshape((1, 28, 28)))
    #then
    assert pred == '1'
def Test_is_not_one(model):
    # given
    test_img = Test_Images.Test_Image.is_not_one()
    # when
    pred = model.predict(test_img.reshape((1, 28, 28)))
    #then
    assert pred == '0'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Model')
    parser.add_argument('test_type', type=int ,default=1, help="1 = Test is a one, 2 = Test is not one")
    parser.add_argument('model_path', type=str, default="Model.CNN", help="path to your trained model")
    parser.add_argument('target_path', type=str, default="Model.CNN", help="path to save your model")

    model = Classifier_Model.get_model(parser.model_path, parser.target_path)

    if parser.test_type == 1:
        Test_is_a_one(model)

    elif parser.test_type == 2:
        Test_is_not_one(model)

    else:
        print('Invalid test number in parser')


