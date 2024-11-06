import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pathlib import Path
from transformers import pipeline

app = FastAPI()

checkpoint = Path(r'C:\Users\user\Desktop\scrap\NLP\checkpoint-5268')
try:
    token_classifier = pipeline(
        "token-classification", model=checkpoint, aggregation_strategy="simple"
    )
except Exception as e:
    raise RuntimeError(f"Failed to load model from checkpoint: {str(e)}")

@app.get('/')
def index():
    return {'message': "Hello, It is an API that handles NER and sentiment analysis task.",
             "For NER" : " input your text at the end of the url http://127.0.0.1:8000/ner/?data=",
             "For Sentiment Analysis" : " input your text at the end of the url http://127.0.0.1:8000/classify/?data="
             }

@app.get('/ner/')
def ner(data: str = Query(..., description="The text to perform NER on")):
    try:

        output = token_classifier(data)
        
  
        result = [{"word": i["word"], "entity": i["entity_group"]} for i in output]
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get('/classify/')
def classify(data: str = Query(..., description="The text to perform Sentiment Analysis on")):
    try:
     sentiment_model_path = Path(r'C:\Users\user\Desktop\scrap\checkpoint-2500')

     sequence_classifier = pipeline(
    "sentiment-analysis",
     model=sentiment_model_path,)
     output = sequence_classifier(data)

     result = [{"Sentiment": i["label"]} for i in output]
     return result
  
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
