from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from fuzzywuzzy import process


app = Flask(__name__)
CORS(app)

# Example FAQ data
faq_data = {
    # General Business Info
    "What are your operating hours?": "We are open from 9 AM to 9 PM, Monday to Saturday.",
    "Are you open on weekends?": "Yes, we are open on Saturdays. However, we are closed on Sundays.",
    "Where are you located?": "im curently at Raja ananmalaipuram chennai",
    "How can I contact customer support?": "You can reach our customer support team at support@example.com or call us at (123) 456-7890.",
    "Do you have parking facilities?": "Yes, we provide free parking for all our customers.",
    "hi":"Hi!, This is Akash how can i assist you",
    "hello":"Hello!, This is Akash how can i assist you",

    # Services
    "What services do you offer?": "We offer dine-in, takeaway, delivery services, and catering for events.",
    "Do you offer delivery?": "Yes, we offer delivery services. Delivery fees may vary depending on the location.",
    "Can I book a table online?": "Yes, you can book a table through our website or mobile app.",
    "Do you provide catering services?": "Yes, we offer catering for corporate events, parties, and gatherings.",

    # Payment and Pricing
    "What payment methods do you accept?": "We accept cash, credit/debit cards, and digital wallets like Google Pay and Apple Pay.",
    "Do you have any ongoing discounts or promotions?": "Yes, we regularly have discounts and promotions. Check our website or social media pages for the latest offers.",
    "Do you offer refunds?": "Refunds are available for cancellations made within 24 hours of purchase. Terms and conditions apply.",
    "Is there a minimum order value for delivery?": "Yes, the minimum order value for delivery is $15.",

    # Products and Availability
    "What are your most popular items?": "Our bestsellers are the Classic Cheeseburger, Pepperoni Pizza, and Chocolate Lava Cake.",
    "Are your products vegetarian-friendly?": "Yes, we have a variety of vegetarian and vegan-friendly options available.",
    "Do you offer gluten-free options?": "Yes, we provide gluten-free alternatives for select items on our menu.",
    "Can I customize my order?": "Yes, you can customize your order. Please let us know your preferences while placing the order.",

    # Technical Issues and Support
    "I forgot my account password. What should I do?": "Click on the 'Forgot Password' link on the login page to reset your password.",
    "How can I track my order?": "You can track your order through the link sent to your email or in the 'Orders' section of our app.",
    "I am facing issues with the website. What should I do?": "Please clear your browser cache and try again. If the issue persists, contact us at support@example.com.",

    # Policies
    "What is your cancellation policy?": "You can cancel your order within 15 minutes of placing it. After that, cancellations are not allowed.",
    "Do you charge for cancellations?": "No, cancellations made within the allowed time frame are free of charge.",
    "What is your privacy policy?": "We value your privacy and do not share your personal information with third parties. Please visit our Privacy Policy page for details.",
    "What is your return policy?": "We accept returns for defective or damaged items within 7 days of purchase. Contact customer support for assistance.",

    # General Inquiries
    "Do you have a mobile app?": "Yes, our app is available on both iOS and Android platforms. Download it from the App Store or Google Play.",
    "Are your services available internationally?": "Currently, our services are available only within the United States.",
    "Can I subscribe to your newsletter?": "Yes, you can subscribe to our newsletter for updates on offers, new products, and more."
}


# Load HuggingFace conversational model
chatbot = pipeline("text-generation", model="microsoft/DialoGPT-medium")

def get_response(user_input):
    # Fuzzy matching for FAQs
    question, score = process.extractOne(user_input, faq_data.keys())
    if score >= 70:  # If confidence score is high enough
        return faq_data[question]

    # Fallback response using GPT
    try:
        response = chatbot(user_input, max_length=100, num_return_sequences=1)
        return response[0]['generated_text']
    except Exception as e:
        return f"Error: {str(e)}"


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    response = get_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
