# Conversation-in-sights
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import json
import base64
import os
import uuid
from datetime import datetime
import time
from dotenv import load_dotenv
import openai
from PIL import Image
import io

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder="static")
CORS(app)

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
# The newest OpenAI model is "gpt-4o" which was released May 13, 2024
MODEL = "gpt-4o"

# In-memory storage for users, conversations, and analysis results
class MemStorage:
    def __init__(self):
        self.users = {}
        self.conversations = {}
        self.analysis_results = {}
        self.user_id_counter = 1
        self.conversation_id_counter = 1
        self.analysis_id_counter = 1
        
        # Create a default user
        self.create_user({
            "username": "demo",
            "password": "password",
            "email": "demo@example.com"
        })
    
    # User operations
    def get_user(self, user_id):
        return self.users.get(user_id)
    
    def get_user_by_username(self, username):
        for user in self.users.values():
            if user["username"] == username:
                return user
        return None
    
    def create_user(self, user_data):
        user_id = self.user_id_counter
        self.user_id_counter += 1
        
        user = {
            "id": user_id,
            "username": user_data["username"],
            "password": user_data["password"],
            "email": user_data["email"],
            "is_premium": False
        }
        
        self.users[user_id] = user
        return user
    
    # Conversation operations
    def get_conversation(self, conversation_id):
        return self.conversations.get(conversation_id)
    
    def get_conversations_by_user_id(self, user_id):
        return [c for c in self.conversations.values() if c["user_id"] == user_id]
    
    def create_conversation(self, conversation_data):
        conversation_id = self.conversation_id_counter
        self.conversation_id_counter += 1
        
        conversation = {
            "id": conversation_id,
            "user_id": conversation_data["user_id"],
            "title": conversation_data["title"],
            "source": conversation_data["source"],
            "content": conversation_data["content"],
            "created_at": datetime.now().isoformat()
        }
        
        self.conversations[conversation_id] = conversation
        return conversation
    
    # Analysis operations
    def get_analysis_result(self, analysis_id):
        return self.analysis_results.get(analysis_id)
    
    def get_analysis_result_by_conversation_id(self, conversation_id):
        for analysis in self.analysis_results.values():
            if analysis["conversation_id"] == conversation_id:
                return analysis
        return None
    
    def create_analysis_result(self, analysis_data):
        analysis_id = self.analysis_id_counter
        self.analysis_id_counter += 1
        
        analysis = {
            "id": analysis_id,
            "conversation_id": analysis_data["conversation_id"],
            "summary": analysis_data["summary"],
            "tags": analysis_data["tags"],
            "response_patterns": analysis_data["responsePatterns"],
            "emotional_intelligence": analysis_data["emotionalIntelligence"],
            "communication_style": analysis_data["communicationStyle"],
            "relationship_dynamics": analysis_data["relationshipDynamics"],
            "improvement_suggestions": analysis_data["improvementSuggestions"],
            "created_at": datetime.now().isoformat()
        }
        
        self.analysis_results[analysis_id] = analysis
        return analysis

# Initialize storage
storage = MemStorage()

# Create uploads directory
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define the system prompt for conversation analysis
SYSTEM_PROMPT = """
You are an expert conversation analyst with deep expertise in psychology, communication patterns, and relationship dynamics.
Your task is to analyze a conversation between two or more people and provide deep, insightful analysis.

Your analysis should be emotionally intelligent, specific to the conversation provided, and avoid generic statements.
Focus on the nuances of communication, tone, response patterns, and the underlying relationship dynamics.

Analyze the following aspects of the conversation:
1. Summary and key themes
2. Response patterns (timing, consistency, engagement)
3. Emotional intelligence (empathy, listening quality)
4. Communication style (tone, message length, conversational balance)
5. Relationship dynamics (engagement balance, relationship type indicators, compatibility)
6. Specific suggestions for improving communication

Respond with a JSON object that follows this exact structure:

{
  "summary": "A concise summary of the conversation",
  "tags": ["tag1", "tag2", "tag3"],
  "responsePatterns": {
    "averageResponseTime": 2.5,  // in hours
    "responseTimeText": "Analysis of response time patterns",
    "responseConsistency": {
      "morning": 85,  // percentage
      "afternoon": 42,
      "evening": 27,
      "analysis": "Detailed analysis of response consistency"
    }
  },
  "emotionalIntelligence": {
    "empathyScore": 8.2,  // scale of 1-10
    "listeningQuality": 7.5,  // scale of 1-10
    "observations": [
      "Observation 1",
      "Observation 2",
      "Observation 3",
      "Observation 4"
    ]
  },
  "communicationStyle": {
    "toneAnalysis": {
      "primary": "Friendly",
      "secondary": "Casual",
      "tertiary": "Sincere"
    },
    "messageLength": {
      "personA": 38,  // average words
      "personB": 22  // average words
    }
  },
  "relationshipDynamics": {
    "engagementBalance": {
      "personA": 55,  // percentage
      "personB": 45  // percentage
    },
    "relationshipTypeIndicators": ["Friendship", "Colleagues"],
    "relationshipDescription": "Detailed description of relationship type",
    "compatibilityScore": 78,  // percentage
    "compatibilityText": "Analysis of compatibility"
  },
  "improvementSuggestions": [
    "Suggestion 1",
    "Suggestion 2",
    "Suggestion 3",
    "Suggestion 4"
  ]
}

Be specific and insightful. Avoid generic analysis that could apply to any conversation. Your insights should feel uniquely tailored to this specific exchange.
"""

# Helper function for SSE
def generate_sse_data(data, event=None):
    message = f"data: {json.dumps(data)}\n\n"
    return message

# OpenAI functions
def analyze_conversation(conversation_data, update_callback):
    """
    Analyze conversation using OpenAI's GPT-4o
    """
    try:
        update_callback("Analyzing tone and sentiment...")
        
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Please analyze this conversation: {conversation_data}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        
        update_callback("Generating detailed insights...")
        
        analysis_content = response.choices[0].message.content
        analysis = json.loads(analysis_content)
        
        update_callback("Finalizing analysis...")
        
        return analysis
    except Exception as e:
        print(f"Error calling OpenAI: {str(e)}")
        raise Exception("Failed to analyze conversation with OpenAI")

def analyze_screenshot(base64_image, update_callback):
    """
    Extract conversation text from a screenshot using OpenAI's Vision capabilities
    """
    try:
        update_callback("Processing screenshot...")
        
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        vision_response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract the conversation from this screenshot. Format it as a JSON array of messages, each with 'text', 'timestamp' (if available), and 'isUser' (alternating between true and false starting with false). Only include the conversation text, not any analysis."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ],
                },
            ],
            response_format={"type": "json_object"},
        )
        
        update_callback("Extracting conversation text...")
        
        return vision_response.choices[0].message.content
    except Exception as e:
        print(f"Error processing screenshot with OpenAI: {str(e)}")
        raise Exception("Failed to process screenshot with OpenAI")

# API Routes
@app.route('/')
def index():
    """Serve the static HTML page"""
    return app.send_static_file('index.html')

@app.route('/api/conversations/text', methods=['POST'])
def upload_text_conversation():
    """Upload and store a text conversation"""
    try:
        data = request.json
        
        if not data or 'title' not in data or 'content' not in data:
            return jsonify({"error": "Title and content are required"}), 400
        
        # For demo, use a placeholder user ID
        user_id = 1
        
        conversation = storage.create_conversation({
            "user_id": user_id,
            "title": data['title'],
            "source": "text",
            "content": json.dumps(data['content'])
        })
        
        return jsonify(conversation), 201
    except Exception as e:
        print(f"Error uploading text conversation: {str(e)}")
        return jsonify({"error": "Failed to upload conversation"}), 500

@app.route('/api/conversations/screenshot', methods=['POST'])
def upload_screenshot():
    """Upload and store a screenshot conversation"""
    try:
        if 'screenshot' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['screenshot']
        title = request.form.get('title')
        
        if not title:
            return jsonify({"error": "Title is required"}), 400
        
        # Save file temporarily
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Read file as base64
        with open(filepath, 'rb') as f:
            file_content = base64.b64encode(f.read()).decode('utf-8')
        
        # For demo, use a placeholder user ID
        user_id = 1
        
        # Store the screenshot
        conversation = storage.create_conversation({
            "user_id": user_id,
            "title": title,
            "source": "screenshot",
            "content": file_content
        })
        
        # Remove the temporary file
        os.remove(filepath)
        
        return jsonify(conversation), 201
    except Exception as e:
        print(f"Error uploading screenshot: {str(e)}")
        return jsonify({"error": "Failed to upload screenshot"}), 500

@app.route('/api/conversations/sample', methods=['GET'])
def get_sample_conversation():
    """Return a sample conversation for demo purposes"""
    try:
        sample_conversation = {
            "messages": [
                {
                    "id": "1",
                    "text": "Hey, I haven't heard from you in a while. How have you been doing?",
                    "timestamp": "11:42 AM",
                    "isUser": False
                },
                {
                    "id": "2",
                    "text": "Sorry for the late reply! I've been super busy with work. Things have been pretty stressful lately.",
                    "timestamp": "2:17 PM",
                    "isUser": True
                },
                {
                    "id": "3",
                    "text": "No worries! I understand how it gets. Anything specific that's making work so stressful?",
                    "timestamp": "2:20 PM",
                    "isUser": False
                },
                {
                    "id": "4",
                    "text": "Just tight deadlines and a new project that's been challenging. But it should calm down next month. How about you?",
                    "timestamp": "2:45 PM",
                    "isUser": True
                }
            ]
        }
        
        return jsonify(sample_conversation), 200
    except Exception as e:
        print(f"Error getting sample conversation: {str(e)}")
        return jsonify({"error": "Failed to get sample conversation"}), 500

@app.route('/api/analyze/<int:conversation_id>', methods=['GET', 'POST'])
def analyze_conversation_route(conversation_id):
    """Analyze a conversation and return the results using SSE"""
    try:
        # Set up SSE response
        def generate():
            try:
                # Get the conversation
                conversation = storage.get_conversation(conversation_id)
                
                if not conversation:
                    yield generate_sse_data({"status": "error", "error": "Conversation not found"})
                    return
                
                # Process based on conversation source
                if conversation["source"] == "screenshot":
                    yield generate_sse_data({"status": "processing", "step": "Reading screenshot content..."})
                    time.sleep(1)  # Simulate processing time
                    
                    # In a real implementation, extract text from the image here
                    messages = json.dumps([
                        {"role": "user", "content": "This is a screenshot of a conversation. Please analyze it."}
                    ])
                else:
                    # For text conversations, use the stored content
                    messages = conversation["content"]
                
                # Analyze with OpenAI
                yield generate_sse_data({"status": "processing", "step": "Analyzing conversation..."})
                
                # Track progress
                progress_steps = []
                
                # Create a callback function that doesn't use yield
                def update_callback(step):
                    progress_steps.append(step)
                    # No yield here - we'll yield separately after the function
                
                # Send initial processing message
                yield generate_sse_data({"status": "processing", "step": "Starting analysis..."})
                
                # Analyze the conversation
                analysis_result = analyze_conversation(messages, update_callback)
                
                # Now send all the collected progress steps
                for step in progress_steps:
                    yield generate_sse_data({"status": "processing", "step": step})
                
                # Send progress update
                yield generate_sse_data({"status": "processing", "step": "Finalizing results..."})
                
                # Return the analysis
                yield generate_sse_data({"status": "completed", "data": analysis_result})
                
                # Store the result
                storage.create_analysis_result({
                    "conversation_id": conversation_id,
                    **analysis_result
                })
                
            except Exception as e:
                print(f"Error analyzing conversation: {str(e)}")
                yield generate_sse_data({"status": "error", "error": "Failed to analyze conversation"})
        
        return Response(stream_with_context(generate()), content_type='text/event-stream')
    except Exception as e:
        print(f"Error setting up analysis: {str(e)}")
        return jsonify({"error": "Failed to set up analysis"}), 500

@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    """Get all conversations for a user"""
    try:
        # For demo, use a placeholder user ID
        user_id = 1
        
        conversations = storage.get_conversations_by_user_id(user_id)
        return jsonify(conversations), 200
    except Exception as e:
        print(f"Error getting conversations: {str(e)}")
        return jsonify({"error": "Failed to get conversations"}), 500

@app.route('/api/conversations/<int:conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    """Get a specific conversation with its analysis"""
    try:
        conversation = storage.get_conversation(conversation_id)
        
        if not conversation:
            return jsonify({"error": "Conversation not found"}), 404
        
        analysis = storage.get_analysis_result_by_conversation_id(conversation_id)
        
        return jsonify({
            "conversation": conversation,
            "analysis": analysis
        }), 200
    except Exception as e:
        print(f"Error getting conversation: {str(e)}")
        return jsonify({"error": "Failed to get conversation"}), 500

if __name__ == '__main__':
    # Run app on 0.0.0.0 to make it accessible outside localhost
    # Use port 5001 to avoid conflicts with the JavaScript version
    app.run(host='0.0.0.0', port=5001, debug=True)
