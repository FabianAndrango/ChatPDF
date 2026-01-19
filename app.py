import streamlit as st # interfaz web
import os # manejo de archivos y variables de entorno
import hashlib # hash del pdf para detectar cambios
import chromadb # base de datos vectorial
import google.generativeai as genai # cliente de Gemini

from pypdf import PdfReader # extracción de texto de PDFs
from sentence_transformers import SentenceTransformer # libreria de texto plano a embeddings
from dotenv import load_dotenv # carga variables de entorno desde .env

import csv # manejo de archivos csv
import chardet # detección de codificación de archivos
import io # manejo de flujos de datos en memoria
import docx  # manejo de archivos docx
# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(page_title="Chat Universal PDF+CSV+DOCX+TXT con Gemini")

# Carga variables de entorno desde .env
# Aquí se espera GOOGLE_API_KEY=xxxx
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Modelo de embeddings local
# Se puede cambiar por otros modelos de sentence-transformers
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Se Inicializa el Cliente de ChromaDB
client = chromadb.Client()

# ============================================================
# SESSION STATE
# ============================================================
# session_state nos permite "recordar" cosas entre reruns.
if "collection" not in st.session_state:
    st.session_state.collection = None

if "file_processed" not in st.session_state:
    st.session_state.file_processed = False

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None


# ============================================================
# FUNCIONES
# ============================================================
def hash_file(file) -> str:
    return hashlib.sha256(file.getvalue()).hexdigest()

def extract_text_from_pdf(pdf_file):
    """
    Extrae texto de un PDF digital (no escaneado).
    Incluye el número de página como marcador.
    """
    reader = PdfReader(pdf_file)
    text = ""

    for i, page in enumerate(reader.pages):
        content = page.extract_text()
        if content:
            text += f"\n[Página {i+1}]\n{content}"

    return text

def extraxt_text_from_csv(file):
    """
    Extraer texto de un archivo CSV.
    Cada registro se convierte en una línea de texto identificada por su número de fila.

    Devuelve:
        Texto completo extraído del CSV.

    """
    text = ""
    #Leer el archivo CSV
    raw_data = file.read()

    # Revisar la codificación del archivo
    enconding_result = chardet.detect(raw_data)
    enconding = enconding_result['encoding']
    # Decodificar el contenido
    decoded_content = raw_data.decode(enconding)
    #Convertir el texto en archivo virtual
    string_io = io.StringIO(decoded_content)

    # 5. Detectar el dialecto (separador, comillas, etc.)
    try:
        dialect = csv.Sniffer().sniff(string_io.read(1024))
    except Exception:
        # En caso de que falle el sniffer, usamos una coma por defecto
        dialect = 'excel' 
    
    string_io.seek(0) # Volver al inicio del archivo virtual

    #  Leer el contenido
    reader = csv.reader(string_io, dialect=dialect)
    text = ""
    for i, row in enumerate(reader):
        line = ' '.join(str(r) for r in row)
        text += f"Fila {i+1}: {line}\n"

    return text
def extract_text_from_docx(file):
    """
    Extraer texto de un archivo DOCX.
    Devuelve:
        Texto completo extraído del DOCX.
    """
    text = ""
    # Leer el archivo DOCX
    doc = docx.Document(file)
    for i, para in enumerate(doc.paragraphs):
        if para.text.strip():  # Evitar párrafos vacíos
         text += f"[Párrafo {i+1}] {para.text}\n"
    return text

def extract_text_from_txt(file):
    """
    Extraer texto de un archivo TXT.

    Devuelve:
        Texto completo extraído del TXT.
    """
    text = ""
    #Leer el archivo TXT
    raw_data = file.read()

    # Revisar la codificación del archivo
    enconding_result = chardet.detect(raw_data)
    enconding = enconding_result['encoding']
    # Decodificar el contenido
    text = raw_data.decode(enconding)

    return text

def get_text_dispatch(uplaoded_file):
    """
    Función despachadora para extraer texto según el tipo de archivo.
    """
    file_type = uplaoded_file.name.split(".")[-1].lower() # Obtener extensión del archivo
    uplaoded_file.seek(0)  # Asegurarse de que el puntero del archivo esté al inicio    
    if file_type == "pdf":
        return extract_text_from_pdf(uplaoded_file)
    elif file_type == "csv":
        return extraxt_text_from_csv(uplaoded_file)
    elif file_type == "docx":
        return extract_text_from_docx(uplaoded_file)
    elif file_type == "txt":
        return extract_text_from_txt(uplaoded_file)
    else:
        raise ValueError("Tipo de archivo no soportado.")

    

def chunk_text(text):
    """
    Divide un texto largo en fragmentos (chunks) con solapamiento.

    chunk_size:
        - Número máximo de caracteres por fragmento
        - Valores típicos: 400–800
        - Más grande = más contexto, pero embeddings más caros

    overlap:
        - Número de caracteres que se repiten entre chunks consecutivos
        - Evita que una idea quede cortada entre fragmentos
        - Regla común: 10–20% del chunk_size

    Devuelve:
        Lista de diccionarios, cada uno representando un chunk con:
        - id           -> identificador único
        - content      -> texto del fragmento
        - start_index  -> posición donde comienza en el texto original
        - size         -> longitud real del chunk
    """
    chunk_size = 500 
    overlap = 100
    chunks = []          # Aquí guardaremos todos los fragmentos
    start = 0            # Puntero que indica desde dónde empezamos a cortar
    chunk_id = 0         # Contador para asignar IDs únicos

    # El while se ejecuta mientras NO hayamos llegado al final del texto
    while start < len(text):

        # 1️⃣ Cortamos el texto desde 'start' hasta 'start + chunk_size'
        #    Python corta automáticamente si se pasa del largo del texto
        chunk_text = text[start:start + chunk_size]

        # 2️⃣ Guardamos el chunk junto con metadata útil
        chunks.append({
            "id": f"chunk_{chunk_id}",   # Identificador único del fragmento
            "content": chunk_text,       # Texto real del fragmento
            "start_index": start,        # Posición en el texto original
            "size": len(chunk_text)      # Tamaño real del fragmento
        })

        # 3️⃣ Incrementamos el ID para el próximo chunk
        chunk_id += 1

        # 4️⃣ Avanzamos el puntero 'start'
        #    No avanzamos chunk_size completo,
        #    sino (chunk_size - overlap) para que haya solapamiento
        #
        #    Ejemplo:
        #    chunk_size = 500
        #    overlap    = 100
        #    start avanza 400 caracteres
        #
        #    Los últimos 100 caracteres del chunk actual
        #    aparecerán también al inicio del siguiente
        start += chunk_size - overlap

    # 5️⃣ Cuando start >= len(text), el while termina
    #    y devolvemos todos los fragmentos creados
    return chunks



def create_chroma_collection(chunks):
    """
    Crea una colección nueva en ChromaDB a partir de los chunks generados.

    Cada chunk se almacena junto con:
    - su embedding (vector numérico)
    - su texto original
    - metadata útil
    """

    # ------------------------------
    # 1️⃣ Borrado defensivo
    # ------------------------------
    # Si ya existe una colección con el mismo nombre ("file_rag"),
    try:
        client.delete_collection("file_rag")
    except:
        # Si la colección no existe, Chroma lanza error.
        # Lo ignoramos porque es un caso esperado.
        pass

    # ------------------------------
    # 2️⃣ Crear colección nueva
    # ------------------------------
    # Aquí Chroma crea:
    # - una tabla de documentos
    # - un índice vectorial
    # - espacio para metadatos
    collection = client.create_collection(name="file_rag")

    # ------------------------------
    # 3️⃣ Separar texto de metadata
    # ------------------------------
    if not chunks:
        return None

    # Extraemos SOLO el contenido textual de cada chunk.
    # Esto es lo que se convertirá en embeddings.
    texts = [c["content"] for c in chunks]

    # ------------------------------
    # 4️⃣ Generar embeddings
    # ------------------------------
    # El modelo de SentenceTransformers convierte cada texto
    # en un vector numérico.
    #
    # Cada vector representa el significado del chunk.
    embeddings = EMBEDDING_MODEL.encode(texts)

    # ------------------------------
    # 5️⃣ Insertar datos en Chroma
    # ------------------------------
    collection.add(
        # Texto original del chunk
        documents=texts,

        # Vectores que permiten búsqueda semántica
        embeddings=embeddings.tolist(),

        # IDs únicos
        # Sirven para identificar cada chunk internamente
        ids=[c["id"] for c in chunks],

        # Metadata asociada a cada chunk
        metadatas=[
            {
                "chunk_index": i,         # Orden del chunk
                "start_index": c["start_index"],  # Posición en el texto original
                "chunk_size": c["size"]   # Tamaño real del fragmento
            }
            for i, c in enumerate(chunks)
        ]
    )

    # ------------------------------
    # 6️⃣ Devolver colección lista
    # ------------------------------
    # La colección ya puede:
    # - recibir queries (preguntas)
    # - devolver chunks relevantes
    return collection

def retrieve_context(collection, query, k=4):
    """
    Recupera los k chunks más similares a la pregunta.
    Devuelve tanto el texto como la metadata asociada.
    """
    query_embedding = EMBEDDING_MODEL.encode([query])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=k
    )

    return results


def ask_gemini(context, question):
    """
    Llama a Gemini usando el contexto recuperado.
    El prompt fuerza comportamiento RAG (no inventar).
    """
    model = genai.GenerativeModel("models/gemini-2.5-flash-lite")

    prompt = f"""
Eres un asistente que responde SOLO con la información del contexto.
Si la respuesta no está en el contexto, di: "No se encuentra en el documento".

Contexto:
{context}

Pregunta:
{question}
"""

    response = model.generate_content(prompt)
    return response.text

# ============================================================
# INTERFAZ DE USUARIO (Simplificada)
# ============================================================

st.title("📄 Chat Multi-Documento con IA + Gemini")

uploaded_file = st.file_uploader("Sube un tu archivo aquí", type=["pdf","csv","docx","txt"])

# 🔄 Detectar cambio del file y resetear estado
if uploaded_file:
    current_hash = hash_file(uploaded_file)

    if st.session_state.file_hash != current_hash:
        st.session_state.file_hash = current_hash
        st.session_state.file_processed = False
        st.session_state.collection = None

# ------------------------------
# BOTÓN PROCESAR FILE
# ------------------------------
if uploaded_file and not st.session_state.file_processed:
    if st.button("📥 Procesar File"):
        with st.spinner("Procesando Archivo..."):
            #Determinar el tipo de archivo y extraer texto 
            text = get_text_dispatch(uploaded_file)
            if not text:
                st.error("No se pudo extraer texto del archivo.")
            chunks = chunk_text(text)
            st.session_state.collection = create_chroma_collection(chunks)
            st.session_state.file_processed = True

        st.success(f" procesado ✅ ({len(chunks)} fragmentos)")

# ------------------------------
# SECCIÓN DE PREGUNTAS
# ------------------------------
if st.session_state.file_processed and st.session_state.collection:
    st.divider()
    st.subheader("❓ Pregunta al documento")

    question = st.text_input("Escribe tu pregunta")

    if st.button("🤖 Preguntar") and question:
        with st.spinner("Buscando respuesta..."):
            results = retrieve_context(st.session_state.collection, question)

            # Unimos los documentos para Gemini
            context_text = "\n\n".join(results["documents"][0])

            answer = ask_gemini(context_text, question)

        st.subheader("🤖 Respuesta")
        st.write(answer)

        # ------------------------------
        # DETALLE DEL CONTEXTO USADO
        # ------------------------------
        with st.expander("📚 Contexto usado (detallado)"):
            for i, (doc, meta) in enumerate(
                zip(results["documents"][0], results["metadatas"][0])
            ):
                st.markdown(f"""
**Chunk #{meta['chunk_index']}**
- 📍 Inicio en texto: `{meta['start_index']}`
- 📏 Tamaño: `{meta['chunk_size']}` caracteres

```text
{doc}
""")