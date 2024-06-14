import base64

try:
    with open("./Alzheimer_s Dataset/test/NonDemented/26 (65).jpg", "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        print("Exito!!!")
        print(base64_image)
except Exception as e:
    print(f"Error: {str(e)}")
