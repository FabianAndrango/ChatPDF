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
# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(page_title="Chat PDF y CSV con Gemini")

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

if "collection_csv" not in st.session_state:
    st.session_state.collection_csv = None

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "csv_processed" not in st.session_state:
    st.session_state.csv_processed = False

if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None

if "csv_hash" not in st.session_state:
    st.session_state.csv_hash = None


# ============================================================
# FUNCIONES
# ============================================================
def hash_pdf(file) -> str:
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
    # Si ya existe una colección con el mismo nombre ("pdf_rag"),
    try:
        client.delete_collection("pdf_rag")
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
    collection = client.create_collection(name="pdf_rag")

    # ------------------------------
    # 3️⃣ Separar texto de metadata
    # ------------------------------
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

def create_chroma_collection_csv(chunks):
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
    # Si ya existe una colección con el mismo nombre ("csv_rag"),
    try:
        client.delete_collection("csv_rag")
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
    collection = client.create_collection(name="csv_rag")

    # ------------------------------
    # 3️⃣ Separar texto de metadata
    # ------------------------------
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
# INTERFAZ
# ============================================================

st.title("📄 Chat con PDF y CSV + ChromaDB + Gemini")

uploaded_pdf = st.file_uploader("Sube un PDF", type="pdf")
uploaded_csv = st.file_uploader("Sube un CSV", type="csv")

# 🔄 Detectar cambio de PDF y resetear estado
if uploaded_pdf:
    current_hash = hash_pdf(uploaded_pdf)

    if st.session_state.pdf_hash != current_hash:
        st.session_state.pdf_hash = current_hash
        st.session_state.pdf_processed = False
        st.session_state.collection = None

# 🔄 Detectar cambio de CSV y resetear estado
if uploaded_csv:
    current_hash = hash_pdf(uploaded_csv)

    if st.session_state.csv_hash != current_hash:
        st.session_state.csv_hash = current_hash
        st.session_state.csv_processed = False
        st.session_state.collection_csv = None


# ------------------------------
# BOTÓN PROCESAR PDF
# ------------------------------
if uploaded_pdf and not st.session_state.pdf_processed:
    if st.button("📥 Procesar PDF"):
        with st.spinner("Procesando PDF..."):
            text = extract_text_from_pdf(uploaded_pdf)
            chunks = chunk_text(text)
            st.session_state.collection = create_chroma_collection(chunks)
            st.session_state.pdf_processed = True

        st.success(f"PDF procesado ✅ ({len(chunks)} fragmentos)")

# ------------------------------
# BOTÓN PROCESAR CSV
# ------------------------------
if uploaded_csv and not st.session_state.csv_processed:
    if st.button("📥 Procesar CSV"):
        with st.spinner("Procesando CSV..."):
            text = extraxt_text_from_csv(uploaded_csv)
            chunks = chunk_text(text)
            st.session_state.collection_csv = create_chroma_collection_csv(chunks)
            st.session_state.csv_processed = True

        st.success(f"CSV procesado ✅ ({len(chunks)} fragmentos)")

# ------------------------------
# SECCIÓN DE PREGUNTAS
# ------------------------------
if st.session_state.pdf_processed and st.session_state.collection:
    st.divider()
    st.subheader("❓ Pregunta al documento")

    question = st.text_input("Escribe tu pregunta",key="pdf_query")

    if st.button("🤖 Preguntar",key="pdf_btn") and question:
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
                
# ------------------------------
# SECCIÓN DE PREGUNTAS CSV
# ------------------------------
if st.session_state.csv_processed and st.session_state.collection_csv:
    st.divider()
    st.subheader("❓ Pregunta al documento")

    question = st.text_input("Escribe tu pregunta",key="csv_query")

    if st.button("🤖 Preguntar",key="csv_btn") and question:
        with st.spinner("Buscando respuesta..."):
            results = retrieve_context(st.session_state.collection_csv, question)

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