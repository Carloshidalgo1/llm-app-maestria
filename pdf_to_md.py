import fitz  # PyMuPDF
import os

def process_pdfs_from_folder():
    input_folder = "pdf"
    output_folder = "dataset_carnicos"
    
    # Crear carpetas si no existen
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    
    # Verificar si la carpeta de entrada existe
    if not os.path.exists(input_folder):
        print(f"❌ Error: No encontré la carpeta '{input_folder}'")
        return

    pdf_files = [f for f in os.listdir(input_folder) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"⚠️ No hay archivos PDF en la carpeta '{input_folder}'")
        return

    print(f"🚀 Se encontraron {len(pdf_files)} PDFs. Iniciando extracción...")

    for pdf in pdf_files:
        pdf_path = os.path.join(input_folder, pdf)
        output_name = pdf.replace(".pdf", ".md")
        output_path = os.path.join(output_folder, output_name)
        
        try:
            doc = fitz.open(pdf_path)
            markdown_content = f"--- SOURCE PDF: {pdf} ---\n\n"
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                markdown_content += f"## Página {page_num + 1}\n\n{page.get_text()}\n\n"
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            print(f"✅ Procesado: {pdf} -> {output_name}")
            
        except Exception as e:
            print(f"❌ Error procesando {pdf}: {e}")

if __name__ == "__main__":
    process_pdfs_from_folder()
