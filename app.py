import os
import json
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import config
from visualizationMethods import convert

opt = config.parse_opt()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
# UPLOAD_FOLDER = '../upload'
# ALLOWED_EXTENSIONS = {'json'}

app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#设置编码
app.config['JSON_AS_ASCII'] = False


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/success', methods=['POST'])

def success():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        save_path = os.path.join(basepath, 'save_files/origin',secure_filename(f.filename))
        save_path = os.path.abspath(save_path) # 将路径转换为绝对路径
        f.save(save_path)
        convert.convert(opt)
        final_path = os.path.abspath(os.path.join(basepath, 'save_files/final',secure_filename(f.filename)))
        with open(final_path, "r", encoding="utf-8") as rf:
            datas = json.load(rf)
        info = dict()
        info["nodes"] = datas["nodes"]
        info["edges"] = datas["edges"]
        data = json.dumps(info, ensure_ascii=False)
        return render_template('success.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)