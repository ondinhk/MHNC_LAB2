import os
from flask import Flask, request, render_template
import time
import numpy as np
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "/home/onkhang/PycharmProjects/Demo_Image/uploads"
MODEL = tf.keras.models.load_model('/home/onkhang/PycharmProjects/Demo_Image/Root/Model')
IMG_HEIGHT = 180
IMG_WIDTH = 180
CLASS_NAME = ['s1', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's2', 's20', 's21', 's22',
              's23', 's24', 's25', 's26', 's27', 's28', 's29', 's3', 's30', 's31', 's32', 's33', 's34', 's35', 's36',
              's37', 's38', 's39', 's4', 's40', 's5', 's6', 's7', 's8', 's9']


###############


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/uploads', methods=['GET', 'POST'])
def upload_file():
    # Kiểm tra phương thức gửi lên
    if request.method == 'POST':
        # Kiểm tra có tệp được gửi lên không
        if request.files:
            # Nhận file -> file này là ảnh mình đã nhận
            image = request.files['imageFile']
            # Lấy đường dẫn trong máy
            pathFile = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            # Lưu file vào địa chỉ UPLOAD_FOLDER
            image.save(pathFile)
            time.sleep(1)
            # Demo
            # Code mọi thứ tại đây
            arrResult = pre(image.filename)
            return render_template("index.html", label=str("Nhãn:" + arrResult[0]),
                                   score=[str("Score: "), arrResult[1]])
    return render_template("index.html")


###############

# Hàm đánh giá
def pre(fileName):
    # Load file
    path = app.config['UPLOAD_FOLDER'] + "/" + fileName
    img = tf.keras.utils.load_img(
        path, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    # Chuyển đổi một phiên bản Hình ảnh PIL thành một mảng Numpy.
    img_array = tf.keras.utils.img_to_array(img)
    # Trả về một tensor có chiều dài 1 trục được chèn vào trục chỉ mục.
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Dự đoán
    predictions = MODEL.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Score là 1 mảng các chỉ số
    # Tìm index của giá trị lớn nhất dọc theo trục. ->[np.argmax(score)]
    # Từ chỉ số đem so với class name
    # Sau đó lấy max score trả về
    return [CLASS_NAME[np.argmax(score)], 100 * np.max(score)]


if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True)

# Đầu tiên nhận file (S9)
# Sau đó lưu file lên local
# Cuối cùng truyền đường dẫn file đó qua hàm Predict() -> trả về 1 mảng 2 phần tử
