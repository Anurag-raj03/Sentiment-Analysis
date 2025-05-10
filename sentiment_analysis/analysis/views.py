from django.shortcuts import render
from django.http import JsonResponse
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def index(request):
    if request.method == 'POST':
        text = request.POST.get('text')
        prediction = predict_sentiment(text)
        return JsonResponse({'prediction': prediction})
    return render(request, 'analysis/index.html')

def predict_sentiment(text):
    stop = stopwords.words('english')
    if 'not' in stop:
        stop.remove('not')
    pt = PorterStemmer()

    def preprocessing(text):
        clean_txt = []
        tex = re.sub('[^a-zA-Z0-9]', ' ', text)
        for i in tex.split():
            if i not in stop:
                stemm = pt.stem(i.lower())
                clean_txt.append(stemm)
        return " ".join(clean_txt)

    vect_or = joblib.load(r"C:\\Users\\Anurag\\OneDrive\\Desktop\\PROJECTS\\Review_sentiment\\sentiment_analysis\\analysis\\models\\vector.pkl")
    model = joblib.load(r"C:\\Users\\Anurag\\OneDrive\\Desktop\\PROJECTS\\Review_sentiment\\sentiment_analysis\\analysis\\models\\model.jbl")
    clean_text = preprocessing(text)
    vect_text = vect_or.transform([clean_text])
    prediction = model.predict(vect_text)
    return 'Positive' if prediction == 1 else 'Negative'
