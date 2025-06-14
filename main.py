
from loading import load_dataset
from visualizing import visualize, show_accuracy_loss_graphs
from training import train
from testing import start_webcam_inference

# PARAMETERS
from PIL import Image


classes_csv_path = r"D:\watsapp\Traffic-Sign-Recognition-master\Traffic-Sign-Recognition-master\Datasets\GTSRB\Classes.csv"
sample_images_path = r"D:\watsapp\Traffic-Sign-Recognition-master\Traffic-Sign-Recognition-master\Datasets\GTSRB\Samples"

train_images_path = r'D:\watsapp\Traffic-Sign-Recognition-master\Traffic-Sign-Recognition-master\Datasets\GTSRB\Train'
train_csv_path = r'D:\watsapp\Traffic-Sign-Recognition-master\Traffic-Sign-Recognition-master\Datasets\GTSRB\Train'
validation_images_path = r'D:\watsapp\Traffic-Sign-Recognition-master\Traffic-Sign-Recognition-master\Datasets\GTSRB\Test'
validation_csv_path = r"D:\watsapp\Traffic-Sign-Recognition-master\Traffic-Sign-Recognition-master\Datasets\GTSRB\Test.csv"

ver = "5.0"
epochs = 10
batch_size = 32
image_dimensions = (32, 32, 3)

model_name = f'traffic_sign_classifier_v{ver}_e{epochs}_b{batch_size}.model'
#model_name = f"D:\watsapp\Traffic-Sign-Recognition-master\Traffic-Sign-Recognition-master\Models\traffic_sign_classifier_v{ver}_e{epochs}_b{batch_size}.model"

# BOOLEAN PARAMETERS
condition_load = True
condition_preload = True
condition_visualize = False
condition_train = True
condition_inference = False
condition_prerecorded = False


if __name__ == "__main__":
    # LOADING DATASET
    if condition_load:
        X_train, Y_train, X_validation, Y_validation = load_dataset(train_images_path,
                                                                    validation_images_path,
                                                                    validation_csv_path,
                                                                    image_dimensions)#,
                                                                    #preloaded=condition_preload)

    # VISUALIZING DATASET
    if condition_visualize:
        visualize(classes_csv_path, sample_images_path, train_csv_path, validation_csv_path, image_dimensions)

    # TRAINING THE MODEL
    if condition_train:
        # noinspection PyUnboundLocalVariable
        history = train(dimensions=image_dimensions,
                        X_train=X_train, Y_train=Y_train,
                        X_validation=X_validation, Y_validation=Y_validation,
                        epochs=epochs, batch_size=batch_size,
                        model_name=model_name)

        # SHOWING GRAPHS
        show_accuracy_loss_graphs(history, model_name)

    # INFERENCE
    if condition_inference:
        start_webcam_inference(model_name, condition_prerecorded)


