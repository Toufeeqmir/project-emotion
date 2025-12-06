from textblob import TextBlob

class SentimentEngine:
    def analyze_multimodal(self, face_emotion, spoken_text):
        # 1. Text Analysis
        if not spoken_text:
            text_sent = "Neutral"
            score = 0
        else:
            analysis = TextBlob(spoken_text)
            score = analysis.sentiment.polarity
            if score > 0.1: text_sent = "Positive"
            elif score < -0.1: text_sent = "Negative"
            else: text_sent = "Neutral"

        # 2. Decision Logic
        if face_emotion == "happy" and text_sent == "Positive":
            return "High Engagement", "Great job! Your energy matches your words."
        elif face_emotion in ["angry", "sad", "fear"] and text_sent == "Negative":
            return "High Stress", "You seem upset. Take a moment to breathe."
        elif face_emotion == "happy" and text_sent == "Negative":
            return "Sarcasm/Nervousness", "Mismatch detected. Are you hiding stress?"
        elif face_emotion == "neutral" and text_sent == "Positive":
            return "Low Energy", "Your words are great, but your face is flat. Smile more!"
        else:
            return "Analyzing...", f"Face: {face_emotion}, Speech: {text_sent}"