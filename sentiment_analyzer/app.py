from flask import Flask, render_template, request
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from googletrans import Translator

# Initialize the Flask app
app = Flask(__name__)

# Initialize the VADER sentiment analyzer
nltk.download('vader_lexicon')  # Ensure the VADER lexicon is downloaded
analyzer = SentimentIntensityAnalyzer()

# Initialize the Google Translator
translator = Translator()

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze the text submitted via the form."""
    text_input = request.form.get('text_input')  # Get the text from the form

    sentiment_result = None
    expert_insight = None
    analysis = ""
    language = ""

    if text_input:
        # Detect language
        try:
            if len(text_input.strip()) < 5:  # Minimum length for reliable detection
                language = "Unknown (too short)"
            else:
                language = detect(text_input)
        except:
            language = "Unknown"

        # Translate non-English text to English
        if language != "en" and language != "Unknown":
            try:
                translated_text = translator.translate(text_input, src=language, dest='en').text
            except:
                translated_text = text_input  # Fallback to original text if translation fails
        else:
            translated_text = text_input

        # Perform sentiment analysis on the translated text
        scores = analyzer.polarity_scores(translated_text)
        compound_score = scores['compound']

        # Debugging output
        print(f"Translated Text: {translated_text}")
        print(f"Compound Score: {compound_score}")

        # Determine sentiment based on compound score
        if compound_score > 0.05:  # Adjusted threshold for positive sentiment
            analysis = "Positive"
        elif compound_score < -0.05:  # Adjusted threshold for negative sentiment
            analysis = "Negative"
        else:
            analysis = "Neutral"

        sentiment_result = {
            'text': text_input,
            'translated_text': translated_text,
            'analysis': analysis,
            'compound_score': compound_score,
            'language': language
        }

        # Expert Insights
        if analysis == "Positive":
            if compound_score > 0.5:
                expert_insight = "Your text conveys strong positivity! Keep spreading the good vibes and inspiring others."
            else:
                expert_insight = "Your text conveys positivity! It's great to see optimism in your words."
        elif analysis == "Negative":
            if compound_score < -0.5:
                expert_insight = "Your text conveys strong negativity. Consider reflecting on the cause and seeking support if needed."
            else:
                expert_insight = "Your text seems to express some negativity. Take a moment to focus on things that bring you joy."
        else:  # Neutral
            if -0.02 < compound_score < 0.02:  # Narrower range for true neutrality
                expert_insight = "Your text appears neutral. If you're feeling uncertain, try focusing on what brings you clarity or joy."
            else:
                expert_insight = "Your text is balanced, with no strong emotions detected. Keep expressing yourself!"

        # Debugging output
        print(f"Analysis: {analysis}")
        print(f"Expert Insight Sent to Template: {expert_insight}")

    # Render the same template, passing the results back to display
    return render_template('index.html', result=sentiment_result, expert_insight=expert_insight)

if __name__ == '__main__':
    app.run(debug=True)