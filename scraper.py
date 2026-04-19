import trafilatura
import requests
import urllib3

# Desactivar advertencias de SSL (común en entornos de aprendizaje)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def run_scraper(url, output_file):
    print(f"🕵️ Intentando conexión robusta con: {url}")
    
    # Definimos cabeceras para parecer un navegador real
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }

    try:
        # Usamos requests primero para tener más control que con trafilatura.fetch_url
        response = requests.get(url, headers=headers, verify=False, timeout=15)
        response.raise_for_status() # Lanza error si la página no carga
        
        # 2. Extraer contenido del HTML obtenido
        content = trafilatura.extract(response.text, output_format='markdown', include_tables=True)
        
        if content:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ ¡Éxito! Archivo generado: {output_file}")
        else:
            print("⚠️ El sitio cargó, pero no se encontró contenido legible (posible carga por JS).")
            
    except Exception as e:
        print(f"❌ Error crítico: {e}")

if __name__ == "__main__":
    target = "https://alimentoscarnicos.com.co/"
    run_scraper(target, "alimentos_carnicos.md")
