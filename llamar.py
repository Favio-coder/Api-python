import requests

def llamar_api():
    url = "https://api-python-o067.onrender.com"  # Reemplaza esta URL con la URL de tu API
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("La solicitud GET fue exitosa. La API está funcionando correctamente.")
        else:
            print(f"Error al realizar la solicitud: {response.status_code}")
    except Exception as e:
        print(f"Error al realizar la solicitud: {e}")

if __name__ == "__main__":
    llamar_api()
