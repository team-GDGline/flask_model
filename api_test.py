import base64
import requests

def test_yolo_api(image_path, api_url):
    # 이미지 파일 읽기

    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    # API 요청 데이터 준비
    payload = {
        "image": image_base64
    }
    
    # POST 요청 보내기
    response = requests.post(api_url, json=payload)
    
    # 결과 출력
    print("Status Code:", response.status_code)
    print("Response:", response.json())

# 테스트
test_yolo_api("image.png", "http://127.0.0.1:8080")
