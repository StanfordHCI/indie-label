""""
Intented for use by Dockerfile; downloads pytorch model and caches it 
in the image, so the download doesn't have to run when app starts.
"""

from sentence_transformers import SentenceTransformer

model_name = "paraphrase-MiniLM-L6-v2"
model = SentenceTransformer(model_name)


