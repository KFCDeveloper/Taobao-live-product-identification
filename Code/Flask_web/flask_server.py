# coding=utf-8
import base64
import os

from flask import Flask, request

app = Flask(__name__)
app.run(host='0.0.0.0')


@app.route('/get-picture', methods=['POST'])
def get_picture():
    # 接收图片
    upload_file = request.files['file']
    # 获取图片名
    file_name = upload_file.filename
    # 文件保存目录（桌面）
    file_path = r'../../Temp-File/Data/'
    if upload_file:
        # 地址拼接
        file_paths = os.path.join(file_path, file_name)
        # 保存接收的图片到桌面
        upload_file.save(file_paths)
        print('succeed')
        # 随便打开一张其他图片作为结果返回，
        # with open(r'C:/Users/Administrator/Desktop/1001.jpg', 'rb') as f:
        #     res = base64.b64encode(f.read())
        #     return res
        return "xihuan hh"


if __name__ == "__main__":
    app.run()
    print('运行前检测两个设备是否在同一wifi下，并且ip要设置好')
