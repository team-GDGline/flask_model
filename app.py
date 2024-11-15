import torch
import torchvision
import torchvision.ops
import yaml
import base64
import cv2
import numpy as np
import os
from flask import Flask, request, jsonify
from io import BytesIO

app = Flask(__name__)

# 모델 로드 및 초기화
def initialize_model():
    try:
        model_dir = "./"  # 모델 디렉토리 경로
        model_path = os.path.join(model_dir, "best.torchscript")
        data_yaml_path = os.path.join(model_dir, "data.yaml")
        
        # TorchScript 모델 로드
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        model = torch.jit.load(model_path)
        model.eval()
        
        # 클래스 이름 로드
        if not os.path.exists(data_yaml_path):
            raise FileNotFoundError(f"Data configuration file not found at: {data_yaml_path}")
        
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            class_names = data.get("names", [])
            if not isinstance(class_names, list):
                raise ValueError("The 'names' field in data.yaml must be a list of class names.")
        
        return model, class_names
    
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model: {e}")

# 이미지 전처리
def preprocess(image_data):
    """
    Decode Base64 image and prepare tensor input.
    """
    img_bgr = cv2.imdecode(np.frombuffer(base64.b64decode(image_data), np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    return torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0

# 모델 추론
def inference(model, inputs):
    """
    Run inference on the model.
    """
    with torch.no_grad():
        return model(inputs)

# 후처리
def postprocess(outputs, class_names):
    """
    Extract class names and bounding boxes from the model output.
    """
    conf_threshold = 0.5
    iou_threshold = 0.5
    
    outputs = outputs[0]  # 배치 차원 제거 (shape: [84, 8400])

    # 바운딩 박스 좌표와 클래스 확률 분리
    boxes = outputs[:4, :].transpose(0, 1)  # 첫 4개 값은 바운딩 박스 좌표 ([8400, 4])
    class_scores = outputs[4:, :]  # 클래스 확률 ([80, 8400])

    # 각 바운딩 박스에서 최고 확률의 클래스 ID 및 해당 확률 추출
    class_confidences, class_labels = torch.max(class_scores, dim=0)
    mask = class_confidences > conf_threshold  # 신뢰도 임계값 적용
    
    # 필터링된 바운딩 박스 및 해당 클래스 정보
    filtered_boxes = boxes[mask]
    filtered_scores = class_confidences[mask]
    filtered_labels = class_labels[mask]

    # 비최대 억제 적용
    if filtered_boxes.size(0) > 0:
        indices = torchvision.ops.nms(filtered_boxes, filtered_scores, iou_threshold)
        final_boxes = filtered_boxes[indices]
        final_scores = filtered_scores[indices]
        final_labels = filtered_labels[indices]

        # 바운딩 박스 좌표 변환 (x_center, y_center, w, h → x1, y1, x2, y2)
        final_boxes[:, 0] = final_boxes[:, 0] - final_boxes[:, 2] / 2  # x_center - width / 2
        final_boxes[:, 1] = final_boxes[:, 1] - final_boxes[:, 3] / 2  # y_center - height / 2
        final_boxes[:, 2] = final_boxes[:, 0] + final_boxes[:, 2]  # x_center + width / 2
        final_boxes[:, 3] = final_boxes[:, 1] + final_boxes[:, 3]  # y_center + height / 2

        # 결과 준비
        results = list(set([class_names[class_id.item()] for class_id in final_labels]))
        
        return {"detections": results}
    else:
        return {"detections": []}

# 모델 초기화
model, class_names = initialize_model()

@app.route("/", methods=["GET", "POST"])
def predict():
    try:
        if request.method == "GET":
            # GET 요청에 대한 응답: 단순히 "OK" 반환
            return jsonify({"status": "OK"}), 200
        
        # POST 요청 처리
        data = request.get_json()
        if 'image' not in data:
            return jsonify({"error": "'image' field is required."}), 400
        
        image_data = data.get("image")

        # 전처리
        model_input = preprocess(image_data)
        
        # 추론
        model_output = inference(model, model_input)
        
        # 후처리
        result = postprocess(model_output, class_names)
        
        # 결과 반환
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
