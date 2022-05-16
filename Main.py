import cv2 
import os
import numpy as np

def get_train_image(path):
    '''
        To get a list of train images, images label, and images index using the given path

        Parameters
        ----------
        path : str
            Location of train root directory
        
        Returns
        -------
        list
            List containing all the train images
        list
            List containing all train images label
        list
            List containing all train images indexes
    '''
    persons_name = os.listdir(path)

    # gambar muka
    img_list = []

    # label
    class_list = []

    # index
    index_list = []

    for idx, person_name in enumerate(persons_name):
        image_name_path = os.path.join(path,person_name)

        for image_name in os.listdir(image_name_path):
            image_path = os.path.join(image_name_path, image_name)
            image = cv2.imread(image_path)

            height = image.shape[0]
            width = int(image.shape[1] * (200 / height))
            image = cv2.resize(image, (width, 200))

            img_list.append(image)
            index_list.append(idx)
        
        class_list.append(person_name)
    return img_list, class_list, index_list 


def get_all_test_folders(path):
    '''
        To get a list of test subdirectories using the given path

        Parameters
        ----------
        path : str
            Location of test root directory
        
        Returns
        -------
        list
            List containing all the test subdirectories
    '''
    test_dir = [i for i in os.listdir(path)]
    return test_dir


def get_all_test_images(path):
    '''
        To load a list of test images from given path list. Resize image height 
        to 200 pixels and image width to the corresponding ratio for train images

        Parameters
        ----------
        path : str
            Location of images root directory
        
        Returns
        -------
        list
            List containing all image that has been resized for each Test Folders
    '''
    img_list = []
    for image_name in os.listdir(path):
        image_path = os.path.join(path, image_name)
        image = cv2.imread(image_path)
        height = image.shape[0]
        width = int(image.shape[1] * (200 / height))
        image = cv2.resize(image, (width, 200))

        img_list.append(image)

    return img_list


def detect_faces_and_filter(faces_list, labels_list=None):
    '''
        To detect a face from given image list and filter it if the face on
        the given image is not equals to one

        Parameters
        ----------
        faces_list : list
            List containing all loaded images
        labels_list : list
            List containing all image classes labels
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            list containing image gray face location
        list
            List containing all filtered image classes label
    '''
    image = []
    face_locs = []
    label = []

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


    for i, j in enumerate(faces_list):
        gray_img = cv2.cvtColor(j, cv2.COLOR_BGR2GRAY)

        detected_faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.3, minNeighbors = 5)
        if len(detected_faces) < 1:
            continue

        (x, y, w, h) = detected_faces[0]
        face_img = gray_img[y:y+h, x:x+w]
        image.append(face_img)
        face_locs.append(detected_faces[0])

        if labels_list is not None:
            label.append(labels_list[i])

    return image, face_locs, label


def train(grayed_images_list, grayed_labels_list):
    '''
        To create and train face recognizer object

        Parameters
        ----------
        grayed_images_list : list
            List containing all filtered and cropped face images in grayscale
        grayed_labels : list
            List containing all filtered image classes label
        
        Returns
        -------
        object
            Recognizer object after being trained with cropped face images
    '''
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(grayed_images_list, np.array(grayed_labels_list))

    return face_recognizer


def predict(recognizer, gray_test_image_list):
    '''
        To predict the test image with the recognizer

        Parameters
        ----------
        recognizer : object
            Recognizer object after being trained with cropped face images
        gray_test_image_list : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''
    prediction_results=[]
    for i in gray_test_image_list:
        prediction_results.append(recognizer.predict(i))

    return prediction_results
    


def check_attandee(predicted_name, room_number):
    '''
        To check the predicted user is in the designed room or not

        Parameters
        ----------
        predicted_name : str
            The name result from predicted user
        room_number : int
            The room number that the predicted user entered

        Returns
        -------
        bool
            If predicted user entered the correct room return True otherwise False
    '''
    room_dictionary = {
        1: ['Elon Musk', 'Steve Jobs', 'Benedict Cumberbatch', 'Donald Trump'],
        2: ['IU', 'Kim Se Jeong', 'Kim Seon Ho', 'Rich Brian']
    }
    return predicted_name in room_dictionary[room_number]
    
    
def write_prediction(predict_results, test_image_list, test_faces_rects, train_names, room):
    '''
        To draw prediction and validation results on the given test images

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories
        room: int
            The room number

        Returns
        -------
        list
            List containing all test images after being drawn with
            its prediction and validation results
    '''
    img_list = []
    for i, image in enumerate(test_image_list):
        x,y,w,h = test_faces_rects[i]
        face_img = cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,255), 2)
        image_idx = int(predict_results[i][0])
        label_color = None
        label = train_names[image_idx]
        if check_attandee(label, room):
            label += ' - Present'
            label_color = (0, 255, 0)
        else:
            label += " - Shouldn't be here"
            label_color = (0, 0, 255)

        cv2.putText(face_img, label, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 0.7, label_color, 1)
        img_list.append(face_img)
    
    return img_list


def combine_and_show_result(room, predicted_test_image_list):
    '''
        To show the final image that already combine into one image

        Parameters
        ----------
        room : str
            The room number in string format (e.g. 'Room 1')
        predicted_test_image_list : nparray
            Array containing image data
    '''
    result = np.concatenate(predicted_test_image_list, axis=1)
    cv2.imshow(room, result)
    cv2.waitKey(0)


def main():
    
    train_path = "./Dataset/Train"

    faces_list, labels_list, indexes_list = get_train_image(train_path)
    grayed_trained_images_list, _, grayed_trained_labels_list = detect_faces_and_filter(faces_list, indexes_list)
    recognizer = train(grayed_trained_images_list, grayed_trained_labels_list)

    test_path = "./Dataset/Test"

    test_images_folder = get_all_test_folders(test_path)

    for index, room in enumerate(test_images_folder):
        test_images_list = get_all_test_images(test_path + '/' + room)
        grayed_test_image_list, grayed_test_location, _ = detect_faces_and_filter(test_images_list)
        predict_results = predict(recognizer, grayed_test_image_list)
        predicted_test_image_list = write_prediction(predict_results, test_images_list, grayed_test_location, labels_list, index+1)
        combine_and_show_result(room, predicted_test_image_list)


if __name__ == "__main__":
    main()